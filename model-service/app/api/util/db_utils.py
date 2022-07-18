import json
import os
from .logger import logger
from ..constants import JsonConfig

'''
This function gets DB parameters from 'configuration.json'.
'''

def get_db_params():

    db_name = str()
    db_user = str()
    db_pw = str()
    db_ip = str()
    db_port = str()

    if os.path.exists("configuration.json") and os.path.isfile("configuration.json"):
        with open("configuration.json", "r") as configuration:
            try:
                configuration = json.load(configuration)
                db_name = configuration.get(JsonConfig.DB_NAME_NAME)
                db_user = configuration.get(JsonConfig.DB_USER_NAME)
                db_pw = configuration.get(JsonConfig.DB_PW_NAME)
                db_ip = configuration.get(JsonConfig.DB_IP_NAME)
                db_port = configuration.get(JsonConfig.DB_PORT_NAME)
                if db_name == None or db_user == None or db_pw == None or db_ip == None or db_port == None:
                    raise KeyError()
            except json.JSONDecodeError as json_decode_error:
                logger.error(f"JSONDecodeError triggered at 'configuration.json' file loading: {json_decode_error}")
                exit()
            except KeyError as key_error:
                logger.error(f"KeyError triggered at 'configuration.json' file access: {key_error}")
                exit()

    return db_name, db_user, db_pw, db_ip, db_port
