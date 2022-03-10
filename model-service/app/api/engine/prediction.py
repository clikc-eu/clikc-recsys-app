import pandas as pd
from constants import DataSource, FilePath
from .training import check_trained_model
from .dataset import Dataset
from .load_store import load_data
from util.mapping import map_id_external_to_internal, map_id_internal_to_external, get_external_ids, get_user_feature_mapping
from constants import DataSource, MappingType, PredictionType
import numpy as np
from lightfm.data import Dataset as LightDataset


def predict_for_user(user_id: int, num_pred: int):
    '''
        This functionality performs a prediction for
        a known user given its ID.
        The first 'num_pred' predictions are returned
        sorted by their score (default is 10).
    '''

    if check_trained_model() == False:
        return

    # Assume model and dataset stored as pickle file
    dataset = Dataset(data_source=DataSource.PICKLE)
    model = load_data(FilePath.TRAINED_MODEL_PICKLE_PATH)

    num_items = len(dataset.items_list)

    # Map external id to internal dataset id
    internal_user_id = map_id_external_to_internal(
        dataset=dataset.dataset, external_id=user_id, id_type=MappingType.USER_ID_TYPE)

    # Get items the user has not interacted with
    item_with_interaction_ids = set(list(filter(
        lambda user: user['user_id'] == user_id, dataset.users_list))[0].get('interactions'))

    all_item_ids = set([item['item_id']
                        for item in dataset.items_list])

    item_with_no_interaction_ids = list(
        all_item_ids - item_with_interaction_ids)

    predictions = model.predict(user_id, np.arange(num_items), user_features=dataset.user_features_matrix,
                                item_features=dataset.item_features_matrix)

    return sort_predictions(predictions=predictions, dataset=dataset.dataset, id_type=MappingType.ITEM_ID_TYPE, prediction_type=PredictionType.ITEMS_FOR_USER, item_with_no_interaction_ids=item_with_no_interaction_ids)


def predict_for_unknown_user(user_features_list: list, fake_features_generation: bool = False):
    '''
        This functionality performs predictions for a user with zero interactions
        given its declared tastes.
        fake_features_generation generates this list randomly (for test purposes).
    '''

    if check_trained_model() == False:
        return

    # Assume model and dataset stored as pickle file
    dataset = Dataset(data_source=DataSource.PICKLE)
    model = load_data(FilePath.TRAINED_MODEL_PICKLE_PATH)

    # Generate fake user features if needed
    if fake_features_generation == True:
        user_features_list = dataset.build_fake_new_user_features(
            num_features=4)

    # Get user-feature mappings
    user_feature_map = get_user_feature_mapping(dataset.dataset)

    new_user_features = dataset.format_new_user_input(
        user_feature_map, user_features_list)

    num_items = len(dataset.items_list)

    predictions = model.predict(0, np.arange(
        num_items), user_features=dataset.user_features_matrix, item_features=dataset.item_features_matrix)

    return sort_predictions(predictions=predictions, dataset=dataset, id_type=MappingType.ITEM_ID_TYPE, prediction_type=PredictionType.ITEMS_FOR_UNKNOWN_USER)


def predict_items_for_known_item(item_id: int, num_pred: int):
    '''
        This functionality performs predictions for a given item (items similar to this item)
        via Cosine Similarity.
        The first 'num_pred' predictions are returned
        sorted by their score (default is 10).
    '''
    if check_trained_model() == False:
        return

    # Assume model and dataset stored as pickle file
    dataset = Dataset(data_source=DataSource.PICKLE)
    model = load_data(FilePath.TRAINED_MODEL_PICKLE_PATH)

    internal_item_id = map_id_external_to_internal(
        dataset=dataset.dataset, external_id=item_id, id_type=MappingType.ITEM_ID_TYPE)

    # Get latent representations
    (_, item_representations) = model.get_item_representations(
        dataset.item_features_matrix)

    # Cosine similarity
    scores = item_representations.dot(
        item_representations[internal_item_id, :])
    item_norms = np.linalg.norm(item_representations, axis=1)
    scores /= item_norms
    normalized_scores = scores/item_norms[internal_item_id]

    return sort_predictions(predictions=normalized_scores, id_type=MappingType.ITEM_ID_TYPE, prediction_type=PredictionType.ITEMS_FOR_KNOWN_ITEM)


def sort_predictions(predictions, dataset: Dataset, id_type: MappingType, prediction_type: PredictionType, num_pred: int = 100, item_with_no_interaction_ids=None):
    '''
        This functions sorts items the user has not interacted with
        and return the 'num_pred' number of items.
    '''

    ids = get_external_ids(dataset.dataset, id_type)

    df = pd.DataFrame(data={'id': ids, 'score': predictions})

    if prediction_type == PredictionType.ITEMS_FOR_USER or prediction_type == PredictionType.ITEMS_FOR_UNKNOWN_USER:
        df_items = pd.DataFrame(dataset.items_list)
        df_items.rename(columns={'item_id': 'id'}, inplace=True)
        df = df.join(df_items.set_index('id'), on='id')

    if item_with_no_interaction_ids != None:
        df = df[df.id.isin(item_with_no_interaction_ids)]

    top_x = df.nlargest(df.shape[0], 'score')

    return top_x['id'].head(num_pred).tolist()
