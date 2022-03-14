from aiohttp import ClientSession
from fastapi import APIRouter, Depends, status
import asyncio
from .models import StatusOut
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