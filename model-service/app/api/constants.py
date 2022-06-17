# Folders
from enum import Enum

# File paths
class FilePath():

    __LOCAL_DATASET_FOLDER_PATH = "local_dataset/"

    # Dataset pickle file
    DATASET_PICKLE_PATH = "dataset.pickle"
    TEMP_DATASET_PICKLE_PATH = "temp_dataset.pickle"

    # Trained model pickle file
    TRAINED_MODEL_PICKLE_PATH = "trained_model.pickle"

    # Local dataset items and users files
    ITEM_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "items.json"
    USER_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "users.json"
    ITEM_SCHEMA_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "items_schema.json"
    USER_SCHEMA_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "users_schema.json"

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

# Mapping of user and items
class MappingType(Enum):
    USER_ID_TYPE = 6
    ITEM_ID_TYPE = 7

class PredictionType(Enum):
    ITEMS_FOR_USER = 8
    ITEMS_FOR_UNKNOWN_USER = 9
    ITEMS_FOR_KNOWN_ITEM = 10

class TrainingJob():
    JOB_ID = 'model_training_job'
    JOB_NAME = 'Train model each day'

'''
List of dynamic field types used in the translations of the Learning Units
'''
class DynamicFieldType():
    PARAGRAPH = 'paragraph'
    MEMORY_BOX = 'memory_box'
    REFERENCE = 'reference'
    LANGUAGE_POINT = 'language_point'
