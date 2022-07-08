import json
import os
from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from .service import ModelService
from .schemas import StatusOut, RecommendOut, StatusTrainingOut, UserFeaturesIn
from .util import logger
from .constants import JsonConfig


'''
Authentication
'''
API_KEY = str()

api_key_header = APIKeyHeader(name=JsonConfig.API_KEY_NAME, auto_error=False)

model = APIRouter()


'''
This function loads a configuration file at startup
'''
@model.on_event("startup")
async def startup():
    global API_KEY

    if os.path.exists("configuration.json") and os.path.isfile("configuration.json"):
        with open("configuration.json", "r") as configuration:
            try:
                configuration = json.load(configuration)
                API_KEY = configuration.get(JsonConfig.API_KEY_NAME)
                if not API_KEY:
                    raise KeyError()
            except json.JSONDecodeError as json_decode_error:
                logger.error(f"JSONDecodeError triggered at 'configuration.json' file loading: {json_decode_error}")
                exit()
            except KeyError() as key_error:
                logger.error(f"KeyError triggered at 'configuration.json' file access: {key_error}")
                exit()


async def authentication(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return True
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized. Not Valid Credential.")



'''
This endpoint gives us some informations about Model Service status:
- Server running
- Model trained
'''
@model.get('/status', response_model=StatusOut, status_code=status.HTTP_200_OK)
def get_status(api_key: APIKey = Depends(authentication)):
    
    return ModelService().get_model_status()

'''
    This endpoint gives us some to manually trigger a model training without interfer
    with the default scheduling. Simply, next training is perfomed with this call.
'''
@model.post('/train', response_model=StatusTrainingOut, status_code=status.HTTP_202_ACCEPTED)
def train_model(api_key: APIKey = Depends(authentication)):

    return ModelService().train_model()


'''
This endpoint allows us to obtain recommendations for a given user given its id 'user_id'.
It is possible to specify the last Learning Unit id via query parameter 'last_lu_id'.
Default value is -1. It means that the first Learning Unit, after the self assessment phase,
must be recommended.
'''
@model.get('/recommendations/user/{user_id}', response_model=RecommendOut, status_code=status.HTTP_200_OK)
def get_recommendations_for_user(user_id: int, is_last_lm: int, last_lu_id: int = -1, result: float = 1.0, liked: int = 1, api_key: APIKey = Depends(authentication)):

    # Switch from integers to bools where necessary
    if is_last_lm < 0 or is_last_lm > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'is_last_lm={is_last_lm} Value Is Not Valid.')

    last_lm = False
    if is_last_lm == 1:
        last_lm = True

    if liked < 0 or liked > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f'liked={liked} Value Is Not Valid.')

    is_liked = False
    if liked == 1:
        is_liked = True

    return ModelService().get_recommendations_for_user(user_id=user_id, is_last_lm=last_lm, last_lu_id=str(last_lu_id), result=result, liked=is_liked)