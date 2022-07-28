import json
import os
import pickle

import pandas as pd
from ..constants import FilePath
from ..util.logger import logger
from ..engine.load_store import load_data
from ..util.mapping import cluster_mapper
from sqlalchemy.orm import Session
from . import models
from ..schemas import LMLearningUnit


'''
This function reads labour market learning units from the online db.
'''
def get_all(db: Session):

    lm_lus_data = list()

    lm_lus_data = db.query(models.LearningUnitLabourMarket).all()

    lm_lus_data = list(map(lambda i: LMLearningUnit.from_orm(i), lm_lus_data))

    lus_dict = pd.DataFrame.from_records([lu.dict() for lu in lm_lus_data]).to_dict('records')

    # Enable just for debug purposes
    # save as pickle
    # with open(os.getcwd() + '/' + FilePath.LM_LU_PICKLE_PATH, 'wb') as fle:
    #     pickle.dump(lus_dict, fle, protocol=pickle.HIGHEST_PROTOCOL)

    # # save as json
    # with open(os.getcwd() + '/' + FilePath.LM_LU_JSON_PATH, 'w') as f:
    #     json.dump(lus_dict, f)

    return lus_dict