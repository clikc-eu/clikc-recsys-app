import json
import os
import pickle

import pandas as pd
from ..constants import FilePath
from ..util.logger import logger
from ..util.lm_lu_generator import generate_lm_learning_units
from ..engine.load_store import load_data

'''
This method reads labour market learning units from a local pickle file.
'''
def get_all():
        
    lm_lus_data = list()

    if os.path.isfile(os.getcwd() + '/' + FilePath.LM_LU_PICKLE_PATH):
        logger.info("Loading learning units from pickle file.")
        lus_dict = load_data(os.getcwd() + '/' + FilePath.LM_LU_PICKLE_PATH)
    else:
        logger.info("Generating learning units from scratch.")

        lm_lus_data = generate_lm_learning_units()

        lm_lus_df = pd.DataFrame.from_records([lu.dict() for lu in lm_lus_data])
        lus_dict = lm_lus_df.to_dict('records')

        # save as pickle
        with open(os.getcwd() + '/' + FilePath.LM_LU_PICKLE_PATH, 'wb') as fle:
            pickle.dump(lus_dict, fle, protocol=pickle.HIGHEST_PROTOCOL)

        # save as json
        with open(os.getcwd() + '/' + FilePath.LM_LU_JSON_PATH, 'w') as f:
            json.dump(lus_dict, f)


    return lus_dict