from typing import List
from pydantic import BaseModel

class UserFeaturesIn(BaseModel):
    user_features: List[str]


class StatusOut(BaseModel):
    server_on: bool
    model_trained: bool


class RecommendOut(BaseModel):
    ids: List[int]