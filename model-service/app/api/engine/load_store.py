import json
import os
import pickle
import shutil


def store_data(data, filename):
    '''
    Save data as pickle file
    '''
    with open(filename, 'wb') as fle:
        pickle.dump(data, fle, protocol=pickle.HIGHEST_PROTOCOL)


def store_json(json_data, filename):
    '''
    Save json file on disk
    '''
    with open(filename, 'w') as f:
            json.dump(json_data, f)


def load_data(filename):
    '''
    Load data from pickle file
    '''
    with open(filename, "rb") as inp:
        data = pickle.load(inp)
    return data


def move_data(dest: str, source: str):
    '''
    Move data from 'source' to 'dest'
    '''
    shutil.move(os.path.join('./', source), os.path.join('./', dest))
