from fastapi import APIRouter, HTTPException, status

model = APIRouter()

@model.get('/status', status_code=status.HTTP_200_OK)
def status():
    return {'status': 'running'}
