import os
import pickle
from ..constants import FilePath
from ..util.logger import logger
from ..engine.load_store import load_data
from ..util.mapping import cluster_mapper
from sqlalchemy.orm import Session
from . import models
from ..schemas import LearningUnit


'''
This function reads learning units from a local pickle file.
In case the pickle file does not exist it builds the pickle file
starting from a basic json file.
'''
def get_all(db: Session):

    lus_data = list()

    lus_data = db.query(models.LearningUnit).where(models.LearningUnit.cluster_number > '0', models.LearningUnit.skill > '0',  models.LearningUnit.eqf_level > '1')

    lus_data = list(map(lambda i: LearningUnit.from_orm(i), lus_data))

    for elem in lus_data:
        elem.cluster_number = cluster_mapper(id=elem.id,
            cluster_number=elem.cluster_number, skill=elem.skill, to_db=False)

    # TODO: remove when switching to online users
    # save as pickle
    with open(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH, 'wb') as fle:
        pickle.dump(lus_data, fle, protocol=pickle.HIGHEST_PROTOCOL)

    return lus_data
