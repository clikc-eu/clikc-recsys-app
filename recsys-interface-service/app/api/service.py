from typing import List
from aiohttp import ClientConnectionError, ClientResponse, ClientSession
from .models import StatusOut
from .constants import ModelServiceUrls
import asyncio
from fastapi import HTTPException, status
from . import recsys_interface


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



    async def get_recommendations_for_new_user(self, user_features: List[str], num_pred: int, client_session: ClientSession):

        payload = {
            'user_features': user_features
        }

        params = {
            'num_pred': num_pred
        }
        headers = {'access-token': recsys_interface.MODEL_SERVICE_API_KEY}
        try:
            async with client_session.post(
                url=ModelServiceUrls.RECS_USER_FEATURES_URL,
                json=payload,
                params=params,
                headers=headers
            ) as response:
                res = await response.json()
                if response.status != status.HTTP_200_OK:
                    raise HTTPException(status_code=response.status, detail=res['detail'])
                return res
        except ClientConnectionError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Not Available.")



    async def get_recommendations_for_user(self, user_id: int, num_pred: int, client_session: ClientSession):

        params = {
            'num_pred': num_pred
        }
        headers = {'access-token': recsys_interface.MODEL_SERVICE_API_KEY}
        try:
            async with client_session.get(
                url=ModelServiceUrls.RECS_USER_URL + str(user_id),
                params=params,
                headers=headers
            ) as response:
                res = await response.json()
                if response.status != status.HTTP_200_OK:
                    raise HTTPException(status_code=response.status, detail=res['detail'])
                return res
        except ClientConnectionError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Not Available.")



    async def get_similar_items(self, item_id: int, num_pred: int, client_session: ClientSession):
        
        params = {
            'num_pred': num_pred
        }
        headers = {'access-token': recsys_interface.MODEL_SERVICE_API_KEY}
        try:
            async with client_session.get(
                url=ModelServiceUrls.RECS_ITEM_URL + str(item_id),
                params=params,
                headers=headers
            ) as response:
                res = await response.json()
                if response.status != status.HTTP_200_OK:
                    raise HTTPException(status_code=response.status, detail=res['detail'])
                return res
        except ClientConnectionError:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service Not Available.")

