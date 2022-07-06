import random
from typing import List
from lightfm import LightFM
import pandas as pd
from .training import check_trained_model, train_model
from .dataset import Dataset
from .load_store import load_data
from ..util.mapping import map_id_external_to_internal, map_id_internal_to_external, get_external_ids, get_user_feature_mapping
from ..constants import DataSource, FilePath, MappingType
import numpy as np
from lightfm.data import Dataset as LightDataset
from fastapi import HTTPException, status
from ..repository import lm_learning_unit as lm_lu_repository, learning_unit as lu_repository, user as user_repository
from ..schemas import CompletedLearningUnit, LMLearningUnit
from datetime import datetime

'''
    This functionality performs a prediction for
    a known user given its ID.
    "last_lu_id" represents the id of the last Learning Unit
    viewed by the user. -1 means that we have to provide recommendations
    after the self-assessment phase.
    The first 3 predictions are returned
    sorted by their score.
    When random_mode == True, recommendations are given randomly.
'''


def predict_for_user(user_id: int, is_last_lm: bool, last_item_id: str, result: float, random_mode: bool):

    if random_mode == True:
        # Recommendations made randomly

        item_data = lu_repository.get_all()

        # Items as Python dictionary
        items = pd.DataFrame.from_records(
            [item.dict() for item in item_data]).to_dict('records')

        # Labour market items as Python dictionary
        lm_items = lm_lu_repository.get_all()

        # Users as Python dictionary
        # TODO: get users from online DB
        users = user_repository.get_all()

        # Check if user_id is valid
        user = check_valid_user(user_id, users)

        # Check if last item id is a valid id. -1 is allowed (user with zero interactions)
        check_valid_item(is_last_lm=is_last_lm, last_item_id=last_item_id, items=items, lm_items=lm_items, user=user)

        # Check if result value is valid
        check_valid_result(result)

        if int(last_item_id) != -1 and is_last_lm == False:
            # Update user Learning Unit history with last_item_id
            user = user_repository.update_history(
                user['id'], CompletedLearningUnit(lu_id=last_item_id, result=result, timestamp=datetime.now().timestamp()))

            # Given the eqf level and the cluster number of
            # the last learning unit completed by the user
            # check if it is possible to increase the eqf_level
            # of the user for that cluster.
            # This happens only when all the learning units of a cluster have
            # been completed for a given eqf level.
            increase, skill, cluster_number = check_eqf_level_completed(
                user=user, last_item_id=last_item_id, items=items)

            if increase == True:
                # eqf level must be increased if it is possible (eqf_level < 4)
                user_eqf = int(user.get('eqf_levels')[
                               int(skill) - 1][int(cluster_number) - 1])
                if user_eqf < 4:
                    user_eqf += 1
                    user = user_repository.update_eqf(user['id'], int(
                        skill) - 1, int(cluster_number) - 1, str(user_eqf))

        elif int(last_item_id) != -1 and is_last_lm == True:
            # Update user labour market Learning Unit history with last_item_id
            user = user_repository.update_lm_history(
                user['id'], LMLearningUnit(identifier=last_item_id))

            # Get standard last_item_id
            last_item_id = get_last_lu_id_by_timestamp(user)

        # Check if it is necessary to recommend labour market Learning Unit
        if user['lu_counter'] >= 5:
            lm_lu_ids = get_lm_recommendations(user, lm_items)
            if len(lm_lu_ids) > 0:
                return True, lm_lu_ids[0:1]


        # Get items the user has not interacted with - shape: [{"lu_id": "370", "result": 0.7775328675422801}, ...]
        item_with_no_interaction_ids = get_item_with_no_interaction_ids(
            items, user)

        item_with_no_interaction = list(
            filter(lambda i: i['identifier'] in item_with_no_interaction_ids, items))

        random.shuffle(item_with_no_interaction)

        # Rename identifier field in id
        for i in item_with_no_interaction:
            i['id'] = i.get('identifier')
            del i['identifier']

        # Use Learning Path rules (eqf, etc..)
        # ids are obtained here
        result = apply_filter_two(user=user, items=item_with_no_interaction)

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

        # Since dataset is built on training request: users, items
        # and user history may be different from
        # online db. Use data from online db for checks
        item_data = lu_repository.get_all()

        # Items as Python dictionary
        items = pd.DataFrame.from_records(
            [item.dict() for item in item_data]).to_dict('records')

        # Labour market items as Python dictionary
        lm_items = lm_lu_repository.get_all()

        users = user_repository.get_all()

        # Check if user_id is valid
        user = check_valid_user(user_id, users)

        # Check if last item id is a valid id. -1 is allowed (user with zero interactions)
        check_valid_item(is_last_lm=is_last_lm, last_item_id=last_item_id, items=items, lm_items=lm_items, user=user)

        # Check if result value is valid
        check_valid_result(result)

        # Update user Learning Unit history with last_item_id
        if int(last_item_id) != -1 and is_last_lm == False:
            user = user_repository.update_history(
                user['id'], CompletedLearningUnit(lu_id=last_item_id, result=result, timestamp=datetime.now().timestamp()))

            # Given the eqf level and the cluster number of
            # the last learning unit completed by the user
            # check if it is possible to increase the eqf_level
            # of the user for that cluster.
            # This happens only when all the learning units of a cluster have
            # been completed for a given eqf level.
            increase, skill, cluster_number = check_eqf_level_completed(
                user=user, last_item_id=last_item_id, items=items)

            if increase == True:
                # eqf level must be increased if it is possible (eqf_level < 4)
                user_eqf = int(user.get('eqf_levels')[
                               int(skill) - 1][int(cluster_number) - 1])
                if user_eqf < 4:
                    user_eqf += 1
                    user = user_repository.update_eqf(user['id'], int(
                        skill) - 1, int(cluster_number) - 1, str(user_eqf))

        elif int(last_item_id) != -1 and is_last_lm == True:
            # Update user labour market Learning Unit history with last_item_id
            user = user_repository.update_lm_history(
                user['id'], LMLearningUnit(identifier=last_item_id))

            # Get standard last_item_id
            last_item_id = get_last_lu_id_by_timestamp(user)

        # Check if it is necessary to recommend labour market Learning Unit
        if user['lu_counter'] >= 5:
            lm_lu_ids = get_lm_recommendations(user, lm_items)
            if len(lm_lu_ids) > 0:
                return True, lm_lu_ids[0:1]


        user_in_model: bool = check_user_in_model(user_id, dataset.users_list)

        # If recommendations are for a user which is not
        # in the model dataset, re-build dataset and train the model
        if user_in_model == False:
            train_model()

        print("lastt " + last_item_id)

        # Get recommendations from pipeline
        return False, get_from_pipeline(model=model, dataset=dataset, user=user, last_item_id=last_item_id)


def get_last_lu_id_by_timestamp(user):
    completed_lus = sorted(user['completed_lus'], key=lambda d: d['timestamp'], reverse=True)
    return completed_lus[0]["lu_id"]

'''
This function gets (not viewed) labour market learning
unit recommendations for a given user.
'''
def get_lm_recommendations(user, lm_items):
    
    user_lm_ids = list(map(lambda i: i['identifier'], user['completed_lm_lus']))
    lm_ids = list(map(lambda i: i['identifier'], lm_items))

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
        dataset=dataset.dataset, external_id=str(user['id']), id_type=MappingType.USER_ID_TYPE)

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
            result.append(list(filter(lambda i: i['identifier'] == id, dataset.items_list))[0])
        # Rename identifier field in id
        for i in result:
            i['id'] = i.get('identifier')
            del i['identifier']
        path_a_results = filter_by_fav_clusters(user=user, items=result)

    path_b_results = list()
    if int(last_item_id) != -1:
        path_b_results = get_from_path_b(model=model, user=user,last_item_id=last_item_id, path_a_results=path_a_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids)

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


    filtered = [item['id'] for item in filtered]

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
        filtered.append(list(filter(lambda item: item['skill'] == fav['skill'] and item['cluster_number'] == fav['cluster'], items))[0])

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
    path_a_result = apply_filter_two(user=user, items=path_a_results)

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
def get_from_path_b(model, user, last_item_id, path_a_results: List, dataset, item_with_no_interaction_ids):
    
    internal_item_id = map_id_external_to_internal(
        dataset=dataset.dataset, external_id=str(last_item_id), id_type=MappingType.ITEM_ID_TYPE)

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
    path_b_results = apply_filter_two(user=user, items=path_b_results)

    # Filter 3b: Take top score recommendation and exclude
    # the element taken in path A
    # IMPORTANT: keep sorted elements here!
    path_b_result = list(filter(lambda i: i not in path_a_results, path_b_results))

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
        return get_from_path_b(model=model, user=user, last_item_id=last_item_id, path_a_results=previous_path_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids)

    good_items = list(filter(lambda item: item['result'] >= 0.6, completed_items))

    # Here we have more than one item completed.
    # If we get no good results re-apply path B again.
    if len(good_items) == 0:
        return get_from_path_b(model=model, user=user, last_item_id=last_item_id, path_a_results=previous_path_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids)

    # Here we have at least one good result.
    # For a random element, taken among the elements the user obtained a result >= 60%,
    # get the most similar item the user has not completed
    # and different from path A and path B results (Filter 3c).

    # Randomly pick one item
    random.shuffle(good_items)

    good_item = good_items[0]

    path_c_result += get_from_path_b(model=model, user=user, last_item_id=good_item['lu_id'], path_a_results=previous_path_results, dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids)

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
        filter(lambda item: item['identifier'] == last_item_id, items))[0]

    # Get items for a specific skill, cluster and eqf levels
    cluster_eqf_items = list(filter(lambda item: item['skill'] == lu['skill'] and item['cluster_number']
                             == lu['cluster_number'] and item['eqf_level'] == lu['eqf_level'], items))

    # Get items for a specific skill, cluster and eqf levels
    # completed by a user
    all_completed_lus_ids = [l['lu_id'] for l in user['completed_lus']]
    cluster_eqf_completed_lus = list(filter(lambda item: item['identifier'] in all_completed_lus_ids and item['skill']
                                     == lu['skill'] and item['cluster_number'] == lu['cluster_number'] and item['eqf_level'] == lu['eqf_level'], items))

    return len(cluster_eqf_completed_lus) == len(cluster_eqf_items), lu['skill'], lu['cluster_number']


'''
This function checks if user information is stored
both in model and dataset.
'''
def check_user_in_model(user_id, dataset_users):
    user_list = list(
        filter(lambda user: user['id'] == str(user_id), dataset_users))

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

    all_item_ids = set([item['identifier']
                        for item in items])

    item_with_no_interaction_ids = list(
        all_item_ids - item_with_interaction_ids)

    return item_with_no_interaction_ids


'''
This function checks if last Learning Unit result
is valid, where valid means a value in the range
0.0 - 1.0.
'''
def check_valid_result(result: float):
    if result < 0.0 or result > 1.0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Result With Value {result} Is Not Valid.')


'''
This function checks if user_id is in users list.
If not it triggers an exception.
'''
def check_valid_user(user_id, users):
    user_list = list(filter(lambda user: user['id'] == str(user_id), users))
    if len(user_list) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'User With ID={user_id} Not Found.')

    return user_list[0]


'''
This function checks if last_lu_id is valid: 
- the id must be found in the set of all the items.
- -1 value is allowed for the moment after self-assessment phase (user has never completed learning units)
- the id must not be found in the user's consumed learning unit list.
If last_item_id is not valid it triggers an exception.
'''
def check_valid_item(is_last_lm, last_item_id, items, lm_items, user):

    # Check for existing item if it is from labour market
    if is_last_lm == True and last_item_id not in list(map(lambda i: i['identifier'], lm_items)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Item With ID={last_item_id} Not Found.')

    
    # Check if is_last_lm is a valid value.
    if is_last_lm == True and user['lu_counter'] != 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')

    # Check if the user has already seen the labour market item
    if is_last_lm == True and int(last_item_id) in [int(lu['identifier']) for lu in user['completed_lm_lus']]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')

    
    user_lm_ids = list(map(lambda i: i['identifier'], user['completed_lm_lus']))
    lm_ids = list(map(lambda i: i['identifier'], lm_items))

    if is_last_lm == False and (user['lu_counter'] < 0 or user['lu_counter'] >= 5) and len(user_lm_ids) < len(lm_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')

    # Check for existing item
    if is_last_lm == False and int(last_item_id) != -1 and len(list(filter(lambda item: item['identifier'] == last_item_id, items))) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Item With ID={last_item_id} Not Found.')

    # Check if received last_item_id == -1 but
    # the user has already seen some items
    if is_last_lm == False and int(last_item_id) == -1 and len(user.get('completed_lus')) != 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')

    # Check if the user has already seen the item
    if is_last_lm == False and int(last_item_id) in [int(lu['lu_id']) for lu in user.get('completed_lus')]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')


'''
    This functions sorts items the user has not interacted with by obtained
    score.
'''
def sort_predictions(predictions, dataset: Dataset, id_type: MappingType, item_with_no_interaction_ids=None):

    ids = get_external_ids(dataset.dataset, id_type)

    df = pd.DataFrame(data={'id': ids, 'score': predictions})

    # Rename 'identifier' in 'id'
    df_items = pd.DataFrame(dataset.items_list)
    df_items.rename(columns={'identifier': 'id'}, inplace=True)
    df = df.join(df_items.set_index('id'), on='id')

    # Filter 1: remove items the user has already interacted with
    if item_with_no_interaction_ids != None:
        df = df[df.id.isin(item_with_no_interaction_ids)]

    top_x = df.nlargest(df.shape[0], 'score')

    return top_x.to_dict('records')
