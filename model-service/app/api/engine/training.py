import os
from lightfm import LightFM
from .dataset import Dataset
from .load_store import store_data, move_data
from ..constants import DataSource, FilePath
from ..util.logger import logger


def train_model(data_source: DataSource = DataSource.LOCAL_JSON):
    '''
    This method is needed in order to offer training functionality
    to the recommender system.
    '''
    logger.info("Training request.")

    dataset = Dataset(data_source=data_source)

    logger.info("Training model from scratch.")

    model = train(
        dataset.interactions, dataset.uf_matrix, dataset.if_matrix)

    logger.info("Model trained. Storing on disk.")

    # Save model and dataset (from temporary file to the definitive one)
    store_data(model, FilePath.TRAINED_MODEL_PICKLE_PATH)
    move_data(source=FilePath.TEMP_DATASET_PICKLE_PATH, dest=FilePath.DATASET_PICKLE_PATH)

    logger.info("Model stored on disk.")


def check_trained_model():
    return os.path.isfile(FilePath.TRAINED_MODEL_PICKLE_PATH)


def train(interactions, user_features_matrix, item_features_matrix):
    '''
    This function trains and returns a model.
    '''

    # Build the model with using hyper-parameters
    # https://making.lyst.com/lightfm/docs/lightfm.html
    model = LightFM(
        no_components=50,
        learning_schedule="adadelta",
        rho=0.95,
        max_sampled=50,
        epsilon=1e-06,
        loss="warp",
        learning_rate=0.05,
        item_alpha=0.0,
        user_alpha=0.0,
        random_state=2022
    )

    model.fit(
        # (no_users, no_items) sparse matrix (with 1s denoting positive, and -1s negative interactions)
        interactions=interactions,
        # (no_users, no_user_features) sparse matrix
        user_features=user_features_matrix,
        # (no_items, no_items_features) sparse matrix
        item_features=item_features_matrix,
        epochs=50,
        num_threads=os.cpu_count(),
        verbose=True
    )

    return model
