from fastapi import FastAPI
from .api.model import model

app = FastAPI(openapi_url="/api/v1/model/openapi.json", docs_url="/api/v1/model/docs")

app.include_router(model, prefix='/api/v1/model', tags=['model'])
