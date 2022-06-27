'''
This method reads user data from a local pickle file.
In case the pickle file does not exist it builds the pickle file
starting from a basic json file.
'''
import json
import os
import pickle

import pandas as pd
from ..constants import FilePath
from ..util.logger import logger
from ..util.user_generator import generate_users
from ..engine.load_store import load_data


def get_all():

    user_json = list()
    user_data = list()

    if os.path.isfile(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH):
        logger.info("Loading users from pickle file.")
        user_data = load_data(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)
        user_json = pd.DataFrame.from_records([user.dict() for user in user_data]).to_dict('records')
    else:
        logger.info("Generating users from scratch.")

        user_json, user_data = generate_users()
        # save as json
        with open(os.getcwd() + '/' + FilePath.USER_JSON_PATH, 'w') as f:
            json.dump(user_json, f)

        # save as pickle
        with open(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH, 'wb') as fle:
            pickle.dump(user_data, fle, protocol=pickle.HIGHEST_PROTOCOL)

    return user_json