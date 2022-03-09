import json
import os
import pickle
from constants import FilePath


def store_fake_users_json(data, sorting_key: str):

    sorted_data = sorted(data, key=lambda x: x[sorting_key])

    with open(FilePath.USER_JSON_PATH, 'w') as fout:
        json.dump(sorted_data, fout)


def store_data(data, filename):
    '''
    Save data as pickle file
    '''
    with open(filename, 'wb') as fle:
        pickle.dump(data, fle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    '''
    Load data from pickle file
    '''
    with open(filename, "rb") as inp:
        data = pickle.load(inp)
    return data
