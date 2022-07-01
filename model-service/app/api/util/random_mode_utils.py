import json
import os
from .logger import logger
from ..constants import JsonConfig

'''
This function checks if random recommendations mode
has been enabled via 'configuration.json' using
key named 'random_mode'.
'''
def check_random_mode():

    random_mode = False

    if os.path.exists("configuration.json") and os.path.isfile("configuration.json"):
        with open("configuration.json", "r") as configuration:
            try:
                configuration = json.load(configuration)
                random_mode = configuration.get(JsonConfig.RANDOM_MODE_NAME)
                if random_mode == None:
                    raise KeyError()
            except json.JSONDecodeError as json_decode_error:
                logger.error(f"JSONDecodeError triggered at 'configuration.json' file loading: {json_decode_error}")
                exit()
            except KeyError as key_error:
                logger.error(f"KeyError triggered at 'configuration.json' file access: {key_error}")
                exit()

    return random_mode
