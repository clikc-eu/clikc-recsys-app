'''
This method reads user data from a local pickle file.
In case the pickle file does not exist it builds the pickle file
starting from a basic json file.
'''
from datetime import datetime
import os
import pickle
import time
from typing import List

import pandas as pd
from sqlalchemy import TIMESTAMP
from ..constants import FilePath
from ..util.logger import logger
from ..engine.load_store import load_data, store_data, store_json
from . import models
from ..schemas import CompletedLMLearningUnit, CompletedLearningUnit, User, Cluster
from ..util import mapping
from sqlalchemy.orm import Session

def get_all(db: Session):

    users_data = list()

    users_data = db.query(models.User).where(models.User.user_cluster_skill != None)

    users: List[User] = list()

    for user in users_data:
        u = build_user(user)
        users.append(u)

    # TODO: remove when switching to online users
    # save as pickle
    with open(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH, 'wb') as fle:
        pickle.dump(users, fle, protocol=pickle.HIGHEST_PROTOCOL)

    user_json = pd.DataFrame.from_records([user.dict() for user in users]).to_dict('records')

    # save as json
    store_json(user_json, os.getcwd() + '/' + FilePath.USER_JSON_PATH)

    return user_json
    
def get_one():
    pass


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
This function converts user's learning units history from database format
to recommender system format.
'''
def build_user(user: models.User) -> User:
    
    # Build user eqf levels and favorite clusters
    eqf_levels: List[List[str]] = [['2', '2', '2'], ['2', '2', '2'], ['2', '2', '2'], ['2', '2', '2']]
    fav_clusters: List[Cluster] = list()
    # Max 3 favorite clusters
    i = 3
    for c in user.user_cluster_skill:
        # Get eqf level number for each cluster
        if int(c.cluster_id) >=1 and int(c.cluster_id) <=3:
            skill = '1'
        elif int(c.cluster_id) >=4 and int(c.cluster_id) <=6:
            skill = '2'
        elif int(c.cluster_id) >=7 and int(c.cluster_id) <=9:
            skill = '3'
        elif int(c.cluster_id) >=10 and int(c.cluster_id) <=12:
            skill = '4'

        cluster = mapping.cluster_mapper(id=c.id, cluster_number=str(c.cluster_id), skill=str(skill), to_db=False)

        eqf_levels[int(skill) - 1][int(cluster) - 1] = str(c.skill_value)

        # Check if cluster is a favorite one (max 3). In case add it to user's favorite clusters
        if i > 0 and c.use_for_startup == 1:
            fav_clusters.append(Cluster(skill=skill, cluster=cluster))
            i -= 1

    # Build user completed Learning Units
    completed_lus: List[CompletedLearningUnit] = list()
    completed_lm_lus: List[CompletedLMLearningUnit] = list()

    for lu in user.user_learning_unit:
        if (lu.test_completed_on != None and lu.liked != 0 and lu.learning_unit_id != 0 and lu.learning_unit_id != None) and (lu.learning_unit_labour_market_id == 0 or lu.learning_unit_labour_market_id == None):
            # Standard learning unit
            # Compute average result for all completed tests
            # Get completed test
            tests = list(filter(lambda t: t.test.translation.learning_unit_id == lu.learning_unit_id and t.submitted_on != None, user.user_test))
            # Map each test to its accuracy
            tests = list(map(lambda t: t.accuracy, tests))
            result = 0
            if(len(tests) > 0):
                result = sum(tests) / len(tests)
            if lu.liked == 1:
                liked = True
            elif lu.liked == -1:
                liked = False

            # Convert datetime timestamp to float timestamp
            timestamp = convert_timestamp_to_float(ts=str(lu.test_completed_on))
            completed_lus.append(CompletedLearningUnit(lu_id=lu.learning_unit_id, result=result, liked=liked, timestamp=timestamp))

        elif (lu.test_completed_on != None and lu.learning_unit_labour_market_id != 0 and lu.learning_unit_labour_market_id != None) and (lu.learning_unit_id == 0 or lu.learning_unit_id == None):
            # Labour market learning unit
            timestamp = convert_timestamp_to_float(ts=str(lu.test_completed_on))
            completed_lm_lus.append(CompletedLMLearningUnit(id=lu.learning_unit_labour_market_id, timestamp=timestamp))

    return User(id=user.id, eqf_levels=eqf_levels, fav_clusters=fav_clusters, completed_lus=completed_lus, completed_lm_lus=completed_lm_lus)

'''
This function converts the timestamp in str format to a float number
'''
def convert_timestamp_to_float(ts: str) -> float:
    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    return time.mktime(dt.timetuple()) + dt.microsecond/1e6