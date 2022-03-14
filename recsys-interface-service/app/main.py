from __future__ import annotations


from typing import Final
from fastapi import FastAPI
import asyncio
from .api.recsys_interface import recsys_interface
from aiohttp import ClientSession



app: Final = FastAPI(openapi_url="/api/v1/recsys-interface/openapi.json", docs_url="/api/v1/recsys-interface/docs")

app.include_router(recsys_interface, prefix='/api/v1/recsys-interface', tags=['recsys-interface'])


@app.on_event("startup")
async def startup_event():
    setattr(app.state, "client_session", ClientSession(raise_for_status=False))


@app.on_event("shutdown")
async def shutdown_event():
    await asyncio.wait((app.state.client_session.close()), timeout=5.0)
