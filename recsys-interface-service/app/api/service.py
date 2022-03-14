from aiohttp import ClientSession
from .models import StatusOut
from .constants import ModelServiceUrls
import asyncio


class RecsysInterfaceService():

    async def get_model_service_status(self, client_session: ClientSession):
        async with client_session.get(
            url=ModelServiceUrls.STATUS_URL,
            raise_for_status=True
        ) as response:
            return await response.json()