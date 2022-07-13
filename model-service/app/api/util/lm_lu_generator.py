
import pandas as pd
from ..schemas import LMLearningUnit
from typing import List
from ..constants import FilePath

'''
This is an helper function that builds a labour market learning units pickle file.
'''
def generate_lm_learning_units():

    lm_learning_units: List[LMLearningUnit] = list()

    ids = list(range(1, 33))

    ids = list(map(lambda id: str(id), ids))

    for id in ids:
        lm_learning_units.append(LMLearningUnit(identifier=id))

    return lm_learning_units
