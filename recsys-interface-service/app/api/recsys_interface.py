from fastapi import HTTPException
import json
import os
from aiohttp import ClientSession
from fastapi import APIRouter, Depends, Security, status
from .schemas import StatusOut, StatusTrainingOut, RecommendOut
from .service import RecsysInterfaceService
from starlette.requests import Request
from fastapi.security.api_key import APIKeyHeader, APIKey
from .util import logger


# Authentication
API_KEY = str()
API_KEY_NAME = "access-token"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# Model Service Authentication
MODEL_SERVICE_API_KEY = str()
MODEL_SERVICE_API_KEY_NAME = "model-token"

recsys_interface = APIRouter()

'''
This function loads a configuration file at startup
'''
@recsys_interface.on_event("startup")
async def startup():
    global API_KEY
    global API_KEY_NAME
    global MODEL_SERVICE_API_KEY
    global MODEL_SERVICE_API_KEY_NAME

    if os.path.exists("configuration.json") and os.path.isfile("configuration.json"):
        with open("configuration.json", "r") as configuration:
            try:
                configuration = json.load(configuration)
                API_KEY = configuration.get(API_KEY_NAME)
                MODEL_SERVICE_API_KEY = configuration.get(MODEL_SERVICE_API_KEY_NAME)
                if (not API_KEY) or (not MODEL_SERVICE_API_KEY):
                    raise KeyError()
            except json.JSONDecodeError as json_decode_error:
                logger.error(f"JSONDecodeError triggered at 'configuration.json' file loading: {json_decode_error}")
                exit()
            except KeyError() as key_error:
                logger.error(f"KeyError triggered at 'configuration.json' file access: {key_error}")
                exit()


def client_session_dep(request: Request) -> ClientSession:
    return request.app.state.client_session


async def authentication(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return True
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized. Not Valid Credential.")



@recsys_interface.get('/status', response_model=StatusOut, status_code=status.HTTP_200_OK)
async def get_status(client_session: ClientSession = Depends(client_session_dep), api_key: APIKey = Depends(authentication)):
    '''
    This endpoint gives us some informations from Model Service about its status:
        \n- If Model Service is running or not
        \n- If model of the recommender system has been trained or not
        \n- If Model Service has been launched in "random mode"
    '''
    return await RecsysInterfaceService().get_model_service_status(client_session=client_session)


@recsys_interface.post('/train', response_model=StatusTrainingOut, status_code=status.HTTP_202_ACCEPTED)
async def train_model(client_session: ClientSession = Depends(client_session_dep), api_key: APIKey = Depends(authentication)):

    '''
    This endpoint gives us the possibility to manually trigger a model training.
    '''
    return await RecsysInterfaceService().train_model(client_session=client_session)


@recsys_interface.get('/recommendations/user/{user_id}', response_model=RecommendOut, status_code=status.HTTP_200_OK)
async def get_recommendations_for_user(user_id: int, client_session: ClientSession = Depends(client_session_dep), api_key: APIKey = Depends(authentication)):

    '''
    This endpoint allows us to obtain recommendations for a given user given its id 'user_id'.
    '''

    return await RecsysInterfaceService().get_recommendations_for_user(user_id=user_id, client_session=client_session)