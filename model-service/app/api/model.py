from fastapi import APIRouter, status
from .service import ModelService
from .models import StatusOut


model = APIRouter()

@model.get('/status', response_model=StatusOut, status_code=status.HTTP_200_OK)
def get_status():
    return ModelService().get_model_status()