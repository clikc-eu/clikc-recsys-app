import random
from typing import List
from lightfm import LightFM
import pandas as pd
from .training import check_trained_model, train_model
from .dataset import Dataset
from .load_store import load_data
from ..util.mapping import map_id_external_to_internal, get_external_ids
from ..constants import DataSource, FilePath, MappingType
import numpy as np
from lightfm.data import Dataset as LightDataset
from fastapi import HTTPException, status
from ..repository import lm_learning_unit as lm_lu_repository, learning_unit as lu_repository, user as user_repository
from sqlalchemy.orm import Session


'''
    This functionality performs a prediction for
    a known user given its ID.
    "last_item_id" represents the id of the last Learning Unit
    viewed by the user. -1 means that we have to provide recommendations
    after the self-assessment phase.
    The first 3 predictions are returned
    sorted by their score.
    When random_mode == True, recommendations are given randomly.
'''


def predict_for_user(user_id: int, random_mode: bool, db: Session):

    if random_mode == True:
        # Recommendations made randomly

        item_data = lu_repository.get_all(db)

        # Items as Python dictionary
        items = pd.DataFrame.from_records(
            [item.dict() for item in item_data]).to_dict('records')

        # Labour market items as Python dictionary
        lm_items = lm_lu_repository.get_all(db)

        # User as Python dictionary if not None
        user = user_repository.get_one(user_id, db)

        # Check if user exists
        check_valid_user(user_id, user)

        # Determine last_item_id and if it was a labour marker LU or not
        is_last_lm, last_item_id = determine_last_item(user)

        # Check if last item id is a valid id. -1 is allowed (user with zero interactions)
        check_valid_item(is_last_lm=is_last_lm, last_item_id=last_item_id, items=items, lm_items=lm_items, user=user)

        if int(last_item_id) != -1 and is_last_lm == False:
            # Given the eqf level and the cluster number of
            # the last learning unit completed by the user
            # check if it is possible to increase the eqf_level
            # of the user for that cluster.
            # This happens only when all the learning units of a cluster have
            # been completed for a given eqf level.
            
            increase, skill, cluster_number = check_eqf_level_completed(
                user=user, last_item_id=int(last_item_id), items=items)

            if increase == True:
                # eqf level must be increased if it is possible (eqf_level < 4)
                user_eqf = int(user.get('eqf_levels')[
                               int(skill) - 1][int(cluster_number) - 1])
                if user_eqf < 4:
                    user_eqf += 1
                    user_repository.update_eqf(user['id'], skill, cluster_number, str(user_eqf), db)
                    # If no DB exception has occured update user local state
                    user['eqf_levels'][int(skill) - 1][int(cluster_number) - 1] = str(user_eqf)

        # Check if it is necessary to recommend labour market Learning Unit
        if len(user['completed_lus']) % 5 == 0 and len(user['completed_lus']) > 0 and is_last_lm == False:
            lm_lu_ids = get_lm_recommendations(user, lm_items)
            if len(lm_lu_ids) > 0:
                return True, lm_lu_ids[0:1]


        # Get items the user has not interacted with - shape: [{"lu_id": "370", "result": 0.7775328675422801}, ...]
        item_with_no_interaction_ids = get_item_with_no_interaction_ids(
            items, user)

        item_with_no_interaction = list(
            filter(lambda i: i['id'] in item_with_no_interaction_ids, items))

        random.shuffle(item_with_no_interaction)

        # Use Learning Path rules (eqf, etc..)
        res = apply_filter_two(user=user, items=item_with_no_interaction)
        # ids are obtained here
        result = [item['id'] for item in res]

        # If after self assessment phase, select items
        # from favourite clusters
        if int(last_item_id) == -1:
            result = list(filter(lambda i: i['id'] in result, item_with_no_interaction))
            result = filter_by_fav_clusters(user=user, items=result)

        # Return firs three elements
        return False, result[0:3]
    else:
        # Check if model has been trained
        if check_trained_model() == False:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail='Model Not Already Trained.')

        # Assume model and dataset stored as pickle file
        dataset = Dataset(data_source=DataSource.PICKLE)
        model = load_data(FilePath.TRAINED_MODEL_PICKLE_PATH)

        items = dataset.items_list

        # Labour market items as Python dictionary
        lm_items = lm_lu_repository.get_all(db)

        # User as Python dictionary if not None
        user = user_repository.get_one(user_id, db)

        # Check if user is not None
        check_valid_user(user_id, user)

        # Determine last_item_id and if it was a labour marker LU or not
        is_last_lm, last_item_id = determine_last_item(user)

        # Check if last item id is a valid id. -1 is allowed (user with zero interactions)
        check_valid_item(is_last_lm=is_last_lm, last_item_id=int(last_item_id), items=items, lm_items=lm_items, user=user)

        if int(last_item_id) != -1 and is_last_lm == False:
            # Given the eqf level and the cluster number of
            # the last learning unit completed by the user
            # check if it is possible to increase the eqf_level
            # of the user for that cluster.
            # This happens only when all the learning units of a cluster have
            # been completed for a given eqf level.
            increase, skill, cluster_number = check_eqf_level_completed(
                user=user, last_item_id=int(last_item_id), items=items)

            if increase == True:
                # eqf level must be increased if it is possible (eqf_level < 4)
                user_eqf = int(user.get('eqf_levels')[
                               int(skill) - 1][int(cluster_number) - 1])
                
                if user_eqf < 4:
                    user_eqf += 1
                    user_repository.update_eqf(user['id'], skill, cluster_number, str(user_eqf), db)
                    # If no DB exception has occured update user local state
                    user['eqf_levels'][int(skill) - 1][int(cluster_number) - 1] = str(user_eqf)

        # Check if it is necessary to recommend labour market Learning Unit
        if len(user['completed_lus']) % 5 == 0 and len(user['completed_lus']) > 0 and is_last_lm == False:
            lm_lu_ids = get_lm_recommendations(user, lm_items)
            if len(lm_lu_ids) > 0:
                return True, lm_lu_ids[0:1]


        user_in_model: bool = check_user_in_model(user_id, dataset.users_list)

        # If recommendations are for a user which is not
        # in the model dataset, re-build dataset and train the model
        if user_in_model == False:
            train_model()

        if is_last_lm == True:
            # We need to get last standard Learning Unit id in order
            # to maintain continuity for the similarity
            last_item_id = get_last_lu_id_by_timestamp(user=user)

        # Get recommendations from pipeline
        return False, get_from_pipeline(model=model, dataset=dataset, user=user, last_item_id=int(last_item_id))


'''
This function gets (not viewed) labour market learning
unit recommendations for a given user.
'''
def get_lm_recommendations(user, lm_items):
    
    user_lm_ids = list(map(lambda i: i['id'], user['completed_lm_lus']))
    lm_ids = list(map(lambda i: i['id'], lm_items))

    random.shuffle(lm_ids)

    return list(filter(lambda i: i not in user_lm_ids, lm_ids))

'''
This function is a wrapper function for the
prediction pipeline.
Pipeline is composed of path A, B and C.
We get one result from each path.
'''
def get_from_pipeline(model, dataset, user, last_item_id):

    # For prediction
    num_items = len(dataset.items_list)

    # Map external user id to internal dataset id
    internal_user_id = map_id_external_to_internal(
        dataset=dataset.dataset, external_id=user['id'], id_type=MappingType.USER_ID_TYPE)

    # Get items the user has not interacted with - shape: [{"lu_id": "370", "result": 0.7775328675422801}, ...]
    # Use updated online data
    item_with_no_interaction_ids = get_item_with_no_interaction_ids(
        dataset.items_list, user)

    path_a_results = get_from_path_a(model=model, user=user, internal_user_id=internal_user_id, num_items=num_items,
                                     dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids, last_item_id=last_item_id)

    # If after self assessment phase, select items
    # from favourite clusters
    if int(last_item_id) == -1:
        result = list()
        for id in path_a_results:
            result.append(list(filter(lambda i: i['id'] == id, dataset.items_list))[0])
        
        path_a_results = filter_by_fav_clusters(user=user, items=result)

    path_b_results = list()
    if int(last_item_id) != -1:
        path_b_results = get_from_path_b(model=model, user=user,last_item_id=last_item_id, path_a_results=path_a_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids, use_likes=True)

    path_c_results = list()
    if int(last_item_id) != -1:
        path_c_results = get_from_path_c(model=model, user=user, last_item_id=last_item_id, previous_path_results=path_a_results + path_b_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids)

    return path_a_results + path_b_results + path_c_results


'''
This function filters items by the eqf level
for a specific cluster for a given user.
It returns the ids.
'''
def apply_filter_two(user, items):

    user_eqf = user['eqf_levels']

    filtered = list(filter(lambda item:
                           (user_eqf[0][0] == item['eqf_level'] and item['skill'] == "1" and item['cluster_number'] == "1") or
                           (user_eqf[0][1] == item['eqf_level'] and item['skill'] == "1" and item['cluster_number'] == "2") or
                           (user_eqf[0][2] == item['eqf_level'] and item['skill'] == "1" and item['cluster_number'] == "3") or
                           (user_eqf[1][0] == item['eqf_level'] and item['skill'] == "2" and item['cluster_number'] == "1") or
                           (user_eqf[1][1] == item['eqf_level'] and item['skill'] == "2" and item['cluster_number'] == "2") or
                           (user_eqf[1][2] == item['eqf_level'] and item['skill'] == "2" and item['cluster_number'] == "3") or
                           (user_eqf[2][0] == item['eqf_level'] and item['skill'] == "3" and item['cluster_number'] == "1") or
                           (user_eqf[2][1] == item['eqf_level'] and item['skill'] == "3" and item['cluster_number'] == "2") or
                           (user_eqf[2][2] == item['eqf_level'] and item['skill'] == "3" and item['cluster_number'] == "3") or
                           (user_eqf[3][0] == item['eqf_level'] and item['skill'] == "4" and item['cluster_number'] == "1") or
                           (user_eqf[3][1] == item['eqf_level'] and item['skill'] == "4" and item['cluster_number'] == "2") or
                           (user_eqf[3][2] == item['eqf_level'] and item['skill'] == "4" and item['cluster_number'] == "3"), items))

    return filtered


'''
This function filters items (already filtered by eqf_level) 
by user selected favourite
clusters (in self assessment phase).
This function takes 1 item for each favourite cluster
'''
def filter_by_fav_clusters(user, items):
    user_fav_clusters = user['fav_clusters']

    filtered = list()

    # Take 1 item for each favourite cluster
    for fav in user_fav_clusters:
        found = list(filter(lambda item: item['skill'] == fav['skill'] and item['cluster_number'] == fav['cluster'], items))
        if len(found) > 0:
            filtered.append(found[0])

    filtered = [item['id'] for item in filtered]

    return filtered


'''
This function represents Path A of the recommendation pipeline.
It uses a LightFM model.
'''
def get_from_path_a(model, user, internal_user_id, num_items, dataset, item_with_no_interaction_ids, last_item_id):

    predictions = model.predict(internal_user_id, np.arange(num_items), user_features=dataset.uf_matrix,
                                item_features=dataset.if_matrix)

    # Returns a dictionary of items
    # identifier field is now 'id'
    path_a_results = sort_predictions(predictions=predictions, dataset=dataset, id_type=MappingType.ITEM_ID_TYPE,
                                      item_with_no_interaction_ids=item_with_no_interaction_ids)

    # Filter 2: Filter by skill, cluster, eqf level
    path_a_res = apply_filter_two(user=user, items=path_a_results)
    path_a_result = [item['id'] for item in path_a_res]

    # If after self-assessment phase return all ids
    if int(last_item_id) == -1:
        return path_a_result

    # Filter 3a: Take top score recommendation
    return path_a_result[0:1]


'''
This function represents Path B of the recommendation pipeline.
It uses cosine similarity to get the most similar item
to previous one.
'''
def get_from_path_b(model, user, last_item_id, path_a_results: List, dataset, item_with_no_interaction_ids, use_likes):
    
    internal_item_id = map_id_external_to_internal(
        dataset=dataset.dataset, external_id=last_item_id, id_type=MappingType.ITEM_ID_TYPE)

    # Get latent representations
    (_, item_representations) = model.get_item_representations(
        dataset.if_matrix)

    # Cosine similarity
    scores = item_representations.dot(
        item_representations[internal_item_id, :])
    item_norms = np.linalg.norm(item_representations, axis=1)
    scores /= item_norms
    normalized_scores = scores/item_norms[internal_item_id]

    # Returns a dictionary of items
    # identifier field is now 'id'
    path_b_results = sort_predictions(predictions=normalized_scores, dataset=dataset,
        id_type=MappingType.ITEM_ID_TYPE, item_with_no_interaction_ids=item_with_no_interaction_ids)

    # Filter 2: Filter by skill, cluster, eqf level
    path_b_res = apply_filter_two(user=user, items=path_b_results)
    
    # Filter 3b:  If user liked latest item take top score. 
    # Else take top score <= of 0.5 (of similarity), 
    # different from the one taken in pipeline A.
    # If we have no top score item with similarity <= 0.5
    # use top score element (standard)
    # IMPORTANT: keep sorted elements here!
    path_b_results = list(filter(lambda i: i['id'] not in path_a_results, path_b_res))

    if use_likes == True:
        # If user liked last element everything is ok (take top score). Check for dislike only.
        last_item = list(filter(lambda i: i['lu_id'] == last_item_id, user['completed_lus']))[0]

        if last_item['liked'] == False:
            # User disliked last item.
            dis_path_b_results = list(filter(lambda i: i['score'] <= 0.5, path_b_results))
            if len(dis_path_b_results) > 0:
                path_b_results = dis_path_b_results

    path_b_result = [item['id'] for item in path_b_results]
    return path_b_result[0:1]

'''
This function represents Path C of the recommendation pipeline.
It uses cosine similarity to get the most similar item
to the ones where the user got best results (>= 60%).
'''
def get_from_path_c(model, user, last_item_id, previous_path_results: List, dataset, item_with_no_interaction_ids):
    
    path_c_result = list()

    completed_items = user['completed_lus']

    # If only one completed apply path B again excluding previous result
    if len(completed_items) == 1:
        return get_from_path_b(model=model, user=user, last_item_id=last_item_id, path_a_results=previous_path_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids, use_likes=False)

    good_items = list(filter(lambda item: item['result'] >= 0.6, completed_items))

    # Here we have more than one item completed.
    # If we get no good results re-apply path B again.
    if len(good_items) == 0:
        return get_from_path_b(model=model, user=user, last_item_id=last_item_id, path_a_results=previous_path_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids, use_likes=False)

    # Here we have at least one good result.
    # For a random element, taken among the elements the user obtained a result >= 60%,
    # get the most similar item the user has not completed
    # and different from path A and path B results (Filter 3c).

    # Randomly pick one item
    random.shuffle(good_items)

    good_item = good_items[0]

    path_c_result += get_from_path_b(model=model, user=user, last_item_id=good_item['lu_id'], path_a_results=previous_path_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids, use_likes=False)

    return path_c_result[0:1]


'''
This function checks if it is necessary to update (returns True)
the eqf level of a cluster for a given user.
This check is performed each time a user completes
a Learning Unit.
'''
def check_eqf_level_completed(user, last_item_id, items):
    # Get current item
    lu = list(
        filter(lambda item: item['id'] == last_item_id, items))[0]

    # Get items for a specific skill, cluster and eqf levels
    cluster_eqf_items = list(filter(lambda item: item['skill'] == lu['skill'] and item['cluster_number']
                             == lu['cluster_number'] and item['eqf_level'] == lu['eqf_level'], items))

    # Get items for a specific skill, cluster and eqf levels
    # completed by a user
    all_completed_lus_ids = [l['lu_id'] for l in user['completed_lus']]
    cluster_eqf_completed_lus = list(filter(lambda item: item['id'] in all_completed_lus_ids and item['skill']
                                     == lu['skill'] and item['cluster_number'] == lu['cluster_number'] and item['eqf_level'] == lu['eqf_level'], items))

    return len(cluster_eqf_completed_lus) == len(cluster_eqf_items), lu['skill'], lu['cluster_number']


'''
This function get the last Learning Unit id the user has seen.
'''
def get_last_lu_id_by_timestamp(user):
    completed_lus = sorted(user['completed_lus'], key=lambda d: d['timestamp'], reverse=True)
    return completed_lus[0]["lu_id"]


'''
This function checks if user information is stored
both in model and dataset.
'''
def check_user_in_model(user_id, dataset_users):
    user_list = list(
        filter(lambda user: user['id'] == user_id, dataset_users))

    if len(user_list) == 0:
        return False

    return True


'''
This function returns the item ids the
user has not interacted with.
'''
def get_item_with_no_interaction_ids(items, user):
    item_with_interaction_ids = user.get('completed_lus')

    item_with_interaction_ids = set(
        [lu["lu_id"] for lu in item_with_interaction_ids])

    all_item_ids = set([item['id']
                        for item in items])

    item_with_no_interaction_ids = list(
        all_item_ids - item_with_interaction_ids)

    return item_with_no_interaction_ids

'''
This function checks if user with user_id is not None.
If not it triggers an exception.
'''
def check_valid_user(user_id, user):
    if user == None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'User With ID={user_id} Not Found.')

    if len(user['fav_clusters']) == 0 or len(user['eqf_levels']) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'User With ID={user_id} Not Valid.')

'''
This function checks if last_lu_id is valid: 
- the id must be found in the set of all the items.
- -1 value is allowed for the moment after self-assessment phase (user has never completed learning units)
- the id must not be found in the user's consumed learning unit list.
If last_item_id is not valid it triggers an exception.
'''
def check_valid_item(is_last_lm, last_item_id, items, lm_items, user):

    # Check for existing item if it is from labour market
    if is_last_lm == True and int(last_item_id) not in list(map(lambda i: i['id'], lm_items)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Item With ID={last_item_id} Not Found.')

    
    # Check if is_last_lm is a valid value.
    # Every 5 LU (counter)
    # If user history does not match this constraint
    # an exception will be triggered
    if is_last_lm == True and len(user['completed_lus']) % 5 != 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')
    

    # Check for existing item
    if is_last_lm == False and int(last_item_id) != -1 and len(list(filter(lambda item: item['id'] == last_item_id, items))) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Item With ID={last_item_id} Not Found.')

'''
This function sorts items the user has not interacted with by obtained
score.
'''
def sort_predictions(predictions, dataset: Dataset, id_type: MappingType, item_with_no_interaction_ids=None):

    ids = get_external_ids(dataset.dataset, id_type)

    df = pd.DataFrame(data={'id': ids, 'score': predictions})

    # Rename 'identifier' in 'id'
    df_items = pd.DataFrame(dataset.items_list)
    df = df.join(df_items.set_index('id'), on='id')

    # Filter 1: remove items the user has already interacted with
    if item_with_no_interaction_ids != None:
        df = df[df.id.isin(item_with_no_interaction_ids)]

    top_x = df.nlargest(df.shape[0], 'score')

    return top_x.to_dict('records')


'''
This function determines the last learning unit id (either from labour marker or not)
and if it comes from labour market set or not.
'''
def determine_last_item(user):
    completed_lus = user.get('completed_lus')
    completed_lm_lus = user.get('completed_lm_lus')

    # Case where the user has not completed any LUs yet.
    if len(completed_lus) == 0:
        return False, -1

    max_ts_lu = list(filter(lambda lu: lu['timestamp'] == max([e['timestamp'] for e in completed_lus]), completed_lus))[0]

    max_ts_lm_lu = list(filter(lambda lu: lu['timestamp'] == max([e['timestamp'] for e in completed_lm_lus]), completed_lm_lus))
    
    if len(max_ts_lm_lu) == 0:
        return False, max_ts_lu['lu_id']

    max_ts_lm_lu = max_ts_lm_lu[0]

    # Last LU was a standard Learning unit
    if max_ts_lu['timestamp'] > max_ts_lm_lu['timestamp']:
        return False, max_ts_lu['lu_id']
    
    # Last LU was a labour market Learning unit
    if max_ts_lm_lu['timestamp'] > max_ts_lu['timestamp']:
        return True, max_ts_lm_lu['id']

    # Non realistic case where both timestamps are equal
    return False, -1

    
    
