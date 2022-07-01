import os
import pickle
from ..constants import FilePath
from ..util.logger import logger
from ..util.lu_generator import generate_learning_units
from ..engine.load_store import load_data

'''
This method reads learning units from a local pickle file.
In case the pickle file does not exist it builds the pickle file
starting from a basic json file.
'''
def get_all():
        
    lus_data = list()

    if os.path.isfile(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH):
        logger.info("Loading learning units from pickle file.")
        lus_data = load_data(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH)
    else:
        logger.info("Generating learning units from scratch.")

        lus_data = generate_learning_units()

        # save as pickle
        with open(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH, 'wb') as fle:
            pickle.dump(lus_data, fle, protocol=pickle.HIGHEST_PROTOCOL)

    return lus_data