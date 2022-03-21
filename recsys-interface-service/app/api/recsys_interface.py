from aiohttp import ClientSession
from fastapi import APIRouter, Depends, status
import asyncio
from .models import StatusOut, StatusTrainingOut, RecommendOut, UserFeaturesIn
from .service import RecsysInterfaceService
from starlette.requests import Request

recsys_interface = APIRouter()

def client_session_dep(request: Request) -> ClientSession:
    return request.app.state.client_session


@recsys_interface.get('/status', response_model=StatusOut, status_code=status.HTTP_200_OK)
async def get_status(client_session: ClientSession = Depends(client_session_dep)):
    '''
        This endpoint gives us some informations from Model Service about its status:
            - Model Service running
            - Model trained
    '''
    return await RecsysInterfaceService().get_model_service_status(client_session=client_session)


@recsys_interface.post('/train', response_model=StatusTrainingOut, status_code=status.HTTP_202_ACCEPTED)
async def train_model(client_session: ClientSession = Depends(client_session_dep)):
    '''
        This endpoint gives us some to manually trigger a model training without interfer
        with the default scheduling. Simply, next training is perfomed with this call.
    '''
    return await RecsysInterfaceService().train_model(client_session=client_session)


@recsys_interface.post('/recommendations/user/features', response_model=RecommendOut, status_code=status.HTTP_200_OK)
async def get_recommendations_for_new_user(user_features: UserFeaturesIn, num_pred: int = 100, client_session: ClientSession = Depends(client_session_dep)):
    '''
        This endpoint allows us to obtain recommendations for a new user (with zero interactions) given some features.
        Features must be sent as a list of strings and must belong to the already existing (into the recommender) set of features.
        It is possible to specify the number of predictions to obtain via query parameter 'num_pred'.
    '''
    return await RecsysInterfaceService().get_recommendations_for_new_user(user_features=user_features.user_features, num_pred=num_pred, client_session=client_session)    


@recsys_interface.get('/recommendations/user/{user_id}', response_model=RecommendOut, status_code=status.HTTP_200_OK)
async def get_recommendations_for_user(user_id: int, num_pred: int = 100, client_session: ClientSession = Depends(client_session_dep)):
    '''
        This endpoint allows us to obtain recommendations for a given user given its id 'user_id'.
        It is possible to specify the number of predictions to obtain via query parameter 'num_pred'.
    '''
    return await RecsysInterfaceService().get_recommendations_for_user(user_id=user_id, num_pred=num_pred, client_session=client_session)


@recsys_interface.get('/recommendations/item/{item_id}', response_model=RecommendOut, status_code=status.HTTP_200_OK)
async def get_similar_items(item_id: int, num_pred: int = 100, client_session: ClientSession = Depends(client_session_dep)):
    '''
        This endpoint allows us to obtain recommendations (similar items) for a given item given its id 'item_id'.
        Similarity is given by cosine similarity.
        It is possible to specify the number of predictions to obtain via query parameter 'num_pred'.
    '''
    return await RecsysInterfaceService().get_similar_items(item_id=item_id, num_pred=num_pred, client_session=client_session)
