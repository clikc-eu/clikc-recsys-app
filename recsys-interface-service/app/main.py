from fastapi import FastAPI
from .api.recsys_interface import recsys_interface


app = FastAPI(openapi_url="/api/v1/recsys-interface/openapi.json", docs_url="/api/v1/recsys-interface/docs")

app.include_router(recsys_interface, prefix='/api/v1/recsys-interface', tags=['recsys-interface'])