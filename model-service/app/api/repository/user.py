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
from ..engine.load_store import load_data, store_data, store_json
from ..schemas import CompletedLearningUnit, LMLearningUnit

'''
This function gets all users from the database (pickle file for now).
'''
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
        store_json(user_json, os.getcwd() + '/' + FilePath.USER_JSON_PATH)

        # save as pickle
        store_data(user_data, os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)

    return user_json


'''
This function updates user data into the database (pickle file for now)
'''
def update_history(user_id: str, completed_lu: CompletedLearningUnit):
    
    # Load users as dictionary
    user_data = load_data(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)


    # user index
    i = 0

    # Find user index in data
    for u in user_data:
        if u.id == user_id:
            break

        i = i + 1

    # Append Learning Unit to user history
    user_data[i].completed_lus.append(completed_lu)
    user_data[i].lu_counter = user_data[i].lu_counter + 1

    user_json = pd.DataFrame.from_records([user.dict() for user in user_data]).to_dict('records')


    # save as json
    store_json(user_json, os.getcwd() + '/' + FilePath.USER_JSON_PATH)

    # save as pickle
    store_data(user_data, os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)

    # return updated user
    return user_json[i]

'''
This function updates the eqf level of a cluster
for a given user.
'''
def update_eqf(user_id: str, skill: int, cluster: int, eqf: str):
    # Load users as dictionary
    user_data = load_data(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)

    # user index
    i = 0

    # Find user index in data
    for u in user_data:
        if u.id == user_id:
            break

        i = i + 1

    # Append Learning Unit to user history
    user_data[i].eqf_levels[skill][cluster] = eqf

    user_json = pd.DataFrame.from_records([user.dict() for user in user_data]).to_dict('records')


    # save as json
    store_json(user_json, os.getcwd() + '/' + FilePath.USER_JSON_PATH)

    # save as pickle
    store_data(user_data, os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)

    # return updated user
    return user_json[i]

'''
This function updates labour market LU user data into the database (pickle file for now)
'''
def update_lm_history(user_id: str, completed_lu: LMLearningUnit):
    
    # Load users as dictionary
    user_data = load_data(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)


    # user index
    i = 0

    # Find user index in data
    for u in user_data:
        if u.id == user_id:
            break

        i = i + 1

    # Append Learning Unit to user history
    user_data[i].completed_lm_lus.append(completed_lu)
    user_data[i].lu_counter = 0

    user_json = pd.DataFrame.from_records([user.dict() for user in user_data]).to_dict('records')


    # save as json
    store_json(user_json, os.getcwd() + '/' + FilePath.USER_JSON_PATH)

    # save as pickle
    store_data(user_data, os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)

    # return updated user
    return user_json[i]


