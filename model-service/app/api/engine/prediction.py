import random
from typing import List
from lightfm import LightFM
import pandas as pd
from .training import check_trained_model, train_model
from .dataset import Dataset
from .load_store import load_data
from ..util.mapping import map_id_external_to_internal, map_id_internal_to_external, get_external_ids, get_user_feature_mapping
from ..constants import DataSource, FilePath, MappingType, PredictionType
import numpy as np
from lightfm.data import Dataset as LightDataset
from fastapi import HTTPException, status
from ..repository import learning_unit as lu_repository, user as user_repository
from ..schemas import CompletedLearningUnit

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


def predict_for_user(user_id: int, last_item_id: str, result: float, random_mode: bool):

    if random_mode == True:
        # Recommendations made randomly

        item_data = lu_repository.get_all()

        # Items as Python dictionary
        items = pd.DataFrame.from_records(
            [item.dict() for item in item_data]).to_dict('records')

        # Users as Python dictionary
        # TODO: get users from online DB
        users = user_repository.get_all()

        # Check if user_id is valid
        user = check_valid_user(user_id, users)

        # Check if last item id is a valid id. -1 is allowed (user with zero interactions)
        check_valid_item(last_item_id, items, user)

        # Check if result value is valid
        check_valid_result(result)

        if int(last_item_id) != -1:
            # Update user Learning Unit history with last_item_id
            user = user_repository.update_history(
                user['id'], CompletedLearningUnit(lu_id=last_item_id, result=result))

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
        result = apply_filter_two(user=user, items=item_with_no_interaction)

        # Return firs three elements
        return result[0:3]
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

        users = user_repository.get_all()

        # Check if user_id is valid
        user = check_valid_user(user_id, users)

        # Check if last item id is a valid id. -1 is allowed (user with zero interactions)
        check_valid_item(last_item_id, items, user)

        # Check if result value is valid
        check_valid_result(result)

        # Update user Learning Unit history with last_item_id
        if int(last_item_id) != -1:
            user = user_repository.update_history(
                user['id'], CompletedLearningUnit(lu_id=last_item_id, result=result))

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

        user_in_model: bool = check_user_in_model(user_id, dataset.users_list)

        # If recommendations are for a user which is not
        # in the model dataset, re-build dataset and train the model
        if user_in_model == False:
            train_model()

        # Get recommendations from pipeline
        return get_from_pipeline(model=model, dataset=dataset, user=user)


'''
This function is a wrapper function for the
prediction pipeline.
Pipeline is composed of path A, B and C.
We get one result from each path.
'''


def get_from_pipeline(model, dataset, user):

    # For prediction
    num_items = len(dataset.items_list)

    # Map external user id to internal dataset id
    internal_user_id = map_id_external_to_internal(
        dataset=dataset.dataset, external_id=str(user['id']), id_type=MappingType.USER_ID_TYPE)

    # Map external item id to internal dataset id
    # if int(last_item_id) != -1:
    #     last_item_internal_id = map_id_external_to_internal(
    #         dataset=dataset.dataset, external_id=last_item_id, id_type=MappingType.ITEM_ID_TYPE)

    # Get items the user has not interacted with - shape: [{"lu_id": "370", "result": 0.7775328675422801}, ...]
    # Use updated online data
    item_with_no_interaction_ids = get_item_with_no_interaction_ids(
        dataset.items_list, user)

    path_a_results = get_from_path_a(model=model, user=user, internal_user_id=internal_user_id, num_items=num_items,
                                     dataset=dataset, item_with_no_interaction_ids=item_with_no_interaction_ids)

    path_b_results = []

    # TODO: Filter 2: Filter by skill, cluster, eqf level
    # TODO: Filter 3b: Take top score recommendation and exclude
    # the element taken in path A

    # TODO: To be discussed
    path_c_results = []

    return path_a_results


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
This function represents Path A of the recommendation pipeline.
It uses a LightFM model.
'''


def get_from_path_a(model, user, internal_user_id, num_items, dataset, item_with_no_interaction_ids):

    predictions = model.predict(internal_user_id, np.arange(num_items), user_features=dataset.uf_matrix,
                                item_features=dataset.if_matrix)

    # Returns a dictionary of items
    # identifier field is now 'id'
    path_a_results = sort_predictions(predictions=predictions, dataset=dataset, id_type=MappingType.ITEM_ID_TYPE,
                                      prediction_type=PredictionType.ITEMS_FOR_USER, item_with_no_interaction_ids=item_with_no_interaction_ids)

    # Filter 2: Filter by skill, cluster, eqf level
    path_a_result = apply_filter_two(user=user, items=path_a_results)

    # Filter 3a: Take top score recommendation
    return path_a_result[0:1]


'''
This function represents Path B of the recommendation pipeline.
It uses cosine similarity to get the most similar item
to previous one.
'''


def get_from_path_b():
    pass


'''
This function represents Path C of the recommendation pipeline.
It uses cosine similarity to get the most similar item
to the ones where the user got best results.
'''


def get_from_path_c():
    pass


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


def check_valid_item(last_item_id, items, user):

    # Check for existing item
    if int(last_item_id) != -1 and len(list(filter(lambda item: item['identifier'] == last_item_id, items))) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Item With ID={last_item_id} Not Found.')

    # Check if received last_item_id == -1 but
    # the user has already seen some items
    if int(last_item_id) == -1 and len(user.get('completed_lus')) != 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')

    # Check if the user has already seen the item
    if int(last_item_id) in [int(lu['lu_id']) for lu in user.get('completed_lus')]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'Item With ID={last_item_id} Not Accepted.')


'''
    TODO: TO BE REMOVED
    This functionality performs predictions for a user with zero interactions
    given the input features.
    fake_features_generation generates this list randomly (for test purposes).
'''


def predict_for_new_user(user_features: List[str], num_pred: int, fake_features_generation: bool = False):

    # Check if model has been trained
    if check_trained_model() == False:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail='Model Not Trained Yet.')

    # Check for num_pred >0
    if num_pred < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='\'num_pred\' Must Be > 0.')

    # Check for at least one feature
    if len(user_features) == 0 and fake_features_generation == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Empty List of Features.')

    if fake_features_generation == False:
        # Remove duplicates from list
        user_features = list(dict.fromkeys(user_features))

    # Assume model and dataset stored as pickle file
    dataset = Dataset(data_source=DataSource.PICKLE)
    model: LightFM = load_data(FilePath.TRAINED_MODEL_PICKLE_PATH)

    if fake_features_generation == False and set(user_features).issubset(set(dataset.user_features)) == False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Not Valid List of Features')

    # Check if num_pred is less than the overall number of items
    if num_pred > len(dataset.items_list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Number Of Requested Predictions Exceeds The Number Of Items')

    # Generate fake user features if needed
    if fake_features_generation == True:
        user_features = dataset.build_fake_new_user_features(
            num_features=1)

    # Get user-feature mappings
    user_feature_map = get_user_feature_mapping(dataset.dataset)

    new_user_features = dataset.format_new_user_input(
        user_feature_map, user_features)

    num_items = len(dataset.items_list)

    predictions = model.predict(0, np.arange(
        num_items), user_features=new_user_features, item_features=dataset.if_matrix)

    return sort_predictions(predictions=predictions, num_pred=num_pred, dataset=dataset, id_type=MappingType.ITEM_ID_TYPE, prediction_type=PredictionType.ITEMS_FOR_UNKNOWN_USER)


'''
    TODO: TO BE MERGED IN PIPELINE
    This functionality performs predictions for a given item (items similar to this item)
    via Cosine Similarity.
    The first 'num_pred' predictions are returned
    sorted by their score (default is 100).
'''


def predict_items_for_known_item(item_id: int, num_pred: int):

    # Check if model has been trained
    if check_trained_model() == False:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail='Model Not Trained Yet.')

    # Check for num_pred >0
    if num_pred < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='\'num_pred\' must be > 0.')

    # Assume model and dataset stored as pickle file
    dataset = Dataset(data_source=DataSource.PICKLE)
    model = load_data(FilePath.TRAINED_MODEL_PICKLE_PATH)

    # Check if item_id is valid
    if len(list(filter(lambda item: item['identifier'] == str(item_id), dataset.items_list))) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f'Item With ID={item_id} Not Found.')

    # Check if num_pred is less than the overall number of items
    if num_pred > len(dataset.items_list):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail='Number Of Requested Predictions Exceeds The Number Of Items')

    internal_item_id = map_id_external_to_internal(
        dataset=dataset.dataset, external_id=str(item_id), id_type=MappingType.ITEM_ID_TYPE)

    # Get latent representations
    (_, item_representations) = model.get_item_representations(
        dataset.if_matrix)

    # Cosine similarity
    scores = item_representations.dot(
        item_representations[internal_item_id, :])
    item_norms = np.linalg.norm(item_representations, axis=1)
    scores /= item_norms
    normalized_scores = scores/item_norms[internal_item_id]

    # it is possible to remove the object for which we are requesting similarity
    return sort_predictions(predictions=normalized_scores, dataset=dataset, num_pred=num_pred,
                            id_type=MappingType.ITEM_ID_TYPE, prediction_type=PredictionType.ITEMS_FOR_KNOWN_ITEM)


'''
    This functions sorts items the user has not interacted with by obtained
    score.
'''


def sort_predictions(predictions, dataset: Dataset, id_type: MappingType, prediction_type: PredictionType, item_with_no_interaction_ids=None):

    ids = get_external_ids(dataset.dataset, id_type)

    df = pd.DataFrame(data={'id': ids, 'score': predictions})

    # Rename 'identifier' in 'id'
    if prediction_type == PredictionType.ITEMS_FOR_USER or prediction_type == PredictionType.ITEMS_FOR_UNKNOWN_USER:
        df_items = pd.DataFrame(dataset.items_list)
        df_items.rename(columns={'identifier': 'id'}, inplace=True)
        df = df.join(df_items.set_index('id'), on='id')

    # Filter 1: remove items the user has already interacted with
    if item_with_no_interaction_ids != None:
        df = df[df.id.isin(item_with_no_interaction_ids)]

    top_x = df.nlargest(df.shape[0], 'score')

    return top_x.to_dict('records')
