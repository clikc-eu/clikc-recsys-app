# Folders
from enum import Enum

# File paths
class FilePath():

    __LOCAL_DATASET_FOLDER_PATH = "local_dataset/"

    # Dataset pickle file
    DATASET_PICKLE_PATH = "dataset.pickle"

    # Trained model pickle file
    TRAINED_MODEL_PICKLE_PATH = "trained_model.pickle"

    # Local dataset items and users files
    ITEM_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "items.json"
    USER_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "users.json"

    # Log file
    LOG_PATH = "main.log"


# Data source
class DataSource(Enum):
    LOCAL_JSON = 1
    PICKLE = 2
    ONLINE_DB = 3
    LOCAL_USER_JSON = 4
    LOCAL_ON_DEMAND_USER = 5

# Load/Store Dataset State
class DatasetState():
    INTERACTIONS = "interactions"
    TEST_INTERACTIONS = "test_interactions"
    USER_FEATURES_MATRIX = "user_features_matrix"
    ITEM_FEATURES_MATRIX = "item_features_matrix"
    USER_FEATURES = "user_features"
    ITEM_FEATURES = "item_features"
    USERS_LIST = "users_list"
    ITEMS_LIST = "items_list"
    DATASET = "dataset"

