import json
import os
import pickle
import random
from datetime import datetime
import pandas as pd
from ..schemas import User, CompletedLearningUnit, Cluster, LearningUnit, CompletedLMLearningUnit
from typing import List
from ..constants import FilePath

'''
This is an helper function that builds a users pickle
file starting from a json file. It uses the already generated learning unit data.
'''
def generate_users():

    users: List[User] = list()
    users_json = list()
    lm_lus = list(range(1, 33))
    lm_lus = list(map(lambda i: str(i), lm_lus))
    available_eqf_levels = ["2", "3", "4"]
    available_clusters = [
        Cluster(skill="1", cluster="1"),
        Cluster(skill="1", cluster="2"),
        Cluster(skill="1", cluster="3"),
        Cluster(skill="2", cluster="1"),
        Cluster(skill="2", cluster="2"),
        Cluster(skill="2", cluster="3"),
        Cluster(skill="3", cluster="1"),
        Cluster(skill="3", cluster="2"),
        Cluster(skill="3", cluster="3"),
        Cluster(skill="4", cluster="1"),
        Cluster(skill="4", cluster="2"),
        Cluster(skill="4", cluster="3"),
    ]

    with open(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH, "rb") as inp:
        item_data: List[LearningUnit] = pickle.load(inp)

    if os.path.isfile(os.getcwd() + '/' + FilePath.BASE_USER_JSON_PATH):
        with open(os.getcwd() + '/' + FilePath.BASE_USER_JSON_PATH, 'r') as f:
            # Read json file as python dictionary
            basic_user_data = json.load(f)


    i = 30 # 30 users after self assessment
    j = 30 # 30 users missing last LU for a cluster in order to increase eqf level
    for basic_user in basic_user_data:

        # Pick eqf levels
        eqf_levels = list(list())
        eqf_levels.append(random.sample(available_eqf_levels, 3))
        eqf_levels.append(random.sample(available_eqf_levels, 3))
        eqf_levels.append(random.sample(available_eqf_levels, 3))
        eqf_levels.append(random.sample(available_eqf_levels, 3))

        # Pick 3 favourite clusters
        fav_clusters = list()
        fav_clusters = random.sample(available_clusters, 3)

        completed_lus: List[CompletedLearningUnit] = list()
        completed_lm_lus: List[CompletedLMLearningUnit] = list()
        lu_counter = 0

        if i > 0:
            # Users with no interactions after self assessment
            i = i - 1
        elif j > 0:
            # Users missing last LU for a cluster in order to increase eqf level + 3 interactions for other clusters
            j = j - 1
            # Pick a random skill
            skill_index = random.sample([0, 1, 2, 3], 1)[0]
            # Pick a random cluster from previous skill
            cluster_index = random.sample([0, 1, 2], 1)[0]
            # Pick correct eqf level
            eqf_level = eqf_levels[skill_index][cluster_index]

            # Extract 9 lus (max is 10 for a cluster of a specific eqf level) given skill_index, cluster_index and eqf_level
            ids = list(filter(lambda lu: lu.skill==str(skill_index + 1) and lu.cluster_number==str(cluster_index + 1) and lu.eqf_level==eqf_level, item_data))
            random.shuffle(ids)
            k = 9
            for id in ids:
                if lu_counter == 5:
                    user_lm_is = list(map(lambda lm_i: lm_i.id, completed_lm_lus))
                    lm_is = list(filter(lambda lm_i: lm_i not in user_lm_is, lm_lus))
                    
                    if len(lm_is) > 0:
                        completed_lm_lus.append(CompletedLMLearningUnit(id=lm_is[0], timestamp=datetime.now().timestamp()))

                    lu_counter = 0
                
                completed_lus.append(CompletedLearningUnit(lu_id=id.id, result=random.uniform(0, 1), timestamp=datetime.now().timestamp(), liked=True))  # results randomly generated
                lu_counter += 1
                k = k - 1
                if k == 0:
                    break

            # Extract 30 lus (max is 10 for a cluster of a specific eqf level) for other clusters given skill_index, cluster_index and eqf_level
            ids = list(filter(lambda lu: lu not in ids, item_data))
            random.shuffle(ids)
            ids = list(filter(lambda lu: lu.eqf_level<=eqf_levels[int(lu.skill) - 1][int(lu.cluster_number) - 1], ids))
            k = 9
            for id in ids:
                if lu_counter == 5:
                    user_lm_is = list(map(lambda lm_i: lm_i.id, completed_lm_lus))
                    lm_is = list(filter(lambda lm_i: lm_i not in user_lm_is, lm_lus))
                    if len(lm_is) > 0:
                        completed_lm_lus.append(CompletedLMLearningUnit(id=lm_is[0], timestamp=datetime.now().timestamp()))
                    lu_counter = 0

                completed_lus.append(CompletedLearningUnit(lu_id=id.id, result=random.uniform(0, 1), timestamp=datetime.now().timestamp(), liked=False))  # results randomly generated
                lu_counter += 1
                k = k - 1
                if k == 0:
                    break
        else:
            # Remaining users interact with lu having eqf_level lower than the one registered for a specific user
            ids = item_data
            random.shuffle(ids)
            ids = list(filter(lambda lu: lu.eqf_level<=eqf_levels[int(lu.skill) - 1][int(lu.cluster_number) - 1], ids))
            k = 10
            for id in ids:

                if lu_counter == 5:
                    user_lm_is = list(map(lambda lm_i: lm_i.id, completed_lm_lus))
                    lm_is = list(filter(lambda lm_i: lm_i not in user_lm_is, lm_lus))
                    if len(lm_is) > 0:
                        completed_lm_lus.append(CompletedLMLearningUnit(id=lm_is[0], timestamp=datetime.now().timestamp()))
                    lu_counter = 0

                completed_lus.append(CompletedLearningUnit(lu_id=id.id, result=random.uniform(0, 1), timestamp=datetime.now().timestamp(), liked=True))  # results randomly generated
                lu_counter += 1
                k = k - 1
                if k == 0:
                    break

        user = User(id=str(basic_user.get('id')),
                    first_name=basic_user.get('first_name'),
                    last_name=basic_user.get('last_name'),
                    username=basic_user.get('username'),
                    email=basic_user.get('email'),
                    gender=basic_user.get('gender'),
                    eqf_levels=eqf_levels,
                    fav_clusters=fav_clusters,
                    completed_lus=completed_lus,
                    completed_lm_lus=completed_lm_lus,
                    lu_counter=lu_counter
        )

        users.append(user)
        users_json.append(user.json())

        users_df = pd.DataFrame.from_records([user.dict() for user in users])

    return users_df.to_dict('records'), users