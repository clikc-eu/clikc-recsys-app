from typing import List
from aiohttp import ClientConnectionError, ClientResponse, ClientSession
from .schemas import StatusOut
from .constants import ModelServiceUrls
import asyncio
from fastapi import HTTPException, status
from . import recsys_interface

'''
This is the service class in charge of contacting the
Model microservice. It uses AIOHTTP library in order
to reach the endpoints.
'''
class RecsysInterfaceService():

    async def get_model_service_status(self, client_session: ClientSession):
        headers = {'access-token': recsys_interface.MODEL_SERVICE_API_KEY}
        try:
            async with client_session.get(
                url=ModelServiceUrls.STATUS_URL,
                headers=headers
            ) as response:
                res = await response.json()
                if response.status != status.HTTP_200_OK:
                    raise HTTPException(status_code=response.status, detail=res['detail'])
                return await response.json()
        except ClientConnectionError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Not Available.")


    async def train_model(self, client_session: ClientSession):
        headers = {'access-token': recsys_interface.MODEL_SERVICE_API_KEY}
        try:
            async with client_session.post(
                url=ModelServiceUrls.TRAIN_URL,
                headers=headers
            ) as response:
                res = await response.json()
                if response.status != status.HTTP_202_ACCEPTED:
                    raise HTTPException(status_code=response.status, detail=res['detail'])
                return await response.json()
        except ClientConnectionError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Not Available.")

    '''
    This is the microservice method in charge of contacting the Model microservice
    in order to forward recommendation requests for a given user.
    '''
    async def get_recommendations_for_user(self, user_id: int, client_session: ClientSession):

        headers = {'access-token': recsys_interface.MODEL_SERVICE_API_KEY}
        try:
            async with client_session.get(
                url=ModelServiceUrls.RECS_USER_URL + str(user_id),
                headers=headers
            ) as response:
                res = await response.json()
                if response.status != status.HTTP_200_OK:
                    raise HTTPException(status_code=response.status, detail=res['detail'])
                return res
        except ClientConnectionError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Not Available.")
