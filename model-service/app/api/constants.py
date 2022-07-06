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

    # TODO: TEMPORARY - Local dataset items and users files
    __BASE_DATA_PATH = "base_data/"
    BASE_LU_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + __BASE_DATA_PATH + "items.json"
    BASE_USER_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + __BASE_DATA_PATH + "users.json"
    LM_LU_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "lm_learning_units_json.json"
    LU_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "learning_units_json.json"
    USER_JSON_PATH = __LOCAL_DATASET_FOLDER_PATH + "users_json.json"
    LM_LU_PICKLE_PATH = __LOCAL_DATASET_FOLDER_PATH + "lm_learning_units_pickle.pickle"
    LU_PICKLE_PATH = __LOCAL_DATASET_FOLDER_PATH + "learning_units_pickle.pickle"
    USER_PICKLE_PATH = __LOCAL_DATASET_FOLDER_PATH + "users_pickle.pickle"

    # Log file
    LOG_PATH = "main.log"


# Data source
class DataSource(Enum):
    PICKLE = 2
    ONLINE_DB = 3
    LOCAL_ON_DEMAND_USER = 5
    LOCAL_DB_PICKLE = 6

# Load/Store Dataset State
class DatasetState():
    INTERACTIONS = "interactions"
    USER_FEATURES_MATRIX = "user_features_matrix"
    ITEM_FEATURES_MATRIX = "item_features_matrix"
    USER_FEATURES = "user_features"
    ITEM_FEATURES = "item_features"
    USERS_LIST = "users_list"
    ITEMS_LIST = "items_list"
    LM_ITEMS_LIST = "lm_items_list"
    DATASET = "dataset"

# Mapping of user and items
class MappingType(Enum):
    USER_ID_TYPE = 6
    ITEM_ID_TYPE = 7

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


'''
List of configuration keys contained in "configuration.json" file
'''
class JsonConfig():
    RANDOM_MODE_NAME = 'random_mode' # Used to start the microservice with random recommendations
    API_KEY_NAME = 'access-token'    # Used to get the api key used in the authentication header