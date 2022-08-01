from datetime import datetime
import os
import pickle
import time
from typing import List
from fastapi import HTTPException, status

import pandas as pd
from sqlalchemy import TIMESTAMP, exc
import sqlalchemy
from ..constants import FilePath
from ..util.logger import logger
from ..engine.load_store import load_data, store_data, store_json
from . import models
from ..schemas import CompletedLMLearningUnit, CompletedLearningUnit, User, Cluster
from ..util import mapping
from sqlalchemy.orm import Session

'''
This function reads users stored in the online DB.
'''
def get_all(db: Session):

    users_data = list()

    users_data = db.query(models.User).where(models.User.user_cluster_skill != None)

    users: List[User] = list()

    for user in users_data:
        u = build_user(user)
        users.append(u)

    # Enable just for debug purposes
    # save as pickle
    # with open(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH, 'wb') as fle:
    #     pickle.dump(users, fle, protocol=pickle.HIGHEST_PROTOCOL)

    user_json = pd.DataFrame.from_records([user.dict() for user in users]).to_dict('records')

    # save as json
    # store_json(user_json, os.getcwd() + '/' + FilePath.USER_JSON_PATH)

    return user_json


'''
This function reads a certain user, stored in the online DB, given its id.
'''
def get_one(user_id: int, db: Session):
    
    # Returns the user in dictionary format or None
    user_data: models.User = db.query(models.User).where(models.User.id == user_id).first()

    if user_data != None:
        user: User = build_user(user_data)
        user_json = pd.DataFrame.from_records([user.dict()]).to_dict('records')[0]

        # Not valid user. Missing data about eqf levels for each cluster
        if user_data.user_cluster_skill == None:
            user_json['eqf_levels'] = []
            user_json['fav_clusters'] = []

        return user_json

    # Case the user is None (not found in DB)
    return user_data


'''
This function updates the eqf level of a cluster
for a given user stored on Database.
'''
def update_eqf(user_id: int, skill: str, cluster: str, eqf: str, db: Session):

    # Map cluster number to db format
    c = mapping.cluster_mapper(user_id, cluster, skill, to_db=True)

    user_cluster_entry = db.query(models.UserClusterSkill).filter(models.UserClusterSkill.user_id == user_id, models.UserClusterSkill.cluster_id == c)

    if user_cluster_entry.first() == None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='Element not found.')

    user_cluster_entry.update({
        models.UserClusterSkill.skill_value: eqf,
    })

    try:
        db.commit()
    except exc.SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='An Internal Error Has Occurred.')


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
            tests = list(filter(lambda t: t.test.translation.learning_unit_id == lu.learning_unit_id and t.submitted_on != None and t.used_for_recap_test == 0, user.user_test))
            
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

        elif (lu.completed_on != None and lu.learning_unit_labour_market_id != 0 and lu.learning_unit_labour_market_id != None) and (lu.learning_unit_id == 0 or lu.learning_unit_id == None):
            # Labour market learning unit
            timestamp = convert_timestamp_to_float(ts=str(lu.completed_on))
            completed_lm_lus.append(CompletedLMLearningUnit(id=lu.learning_unit_labour_market_id, timestamp=timestamp))

    return User(id=user.id, eqf_levels=eqf_levels, fav_clusters=fav_clusters, completed_lus=completed_lus, completed_lm_lus=completed_lm_lus)

'''
This function converts the timestamp in str format to a float number
'''
def convert_timestamp_to_float(ts: str) -> float:
    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    return time.mktime(dt.timetuple()) + dt.microsecond/1e6