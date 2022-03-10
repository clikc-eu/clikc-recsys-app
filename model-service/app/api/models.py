from typing import List
from pydantic import BaseModel


class StatusOut(BaseModel):
    server_on: bool
    model_trained: bool


class RecommendOut(BaseModel):
    ids: List[int]