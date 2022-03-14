from pydantic import BaseModel
from typing import List

class UserFeaturesIn(BaseModel):
    user_features: List[str]


class StatusOut(BaseModel):
    server_on: bool
    model_trained: bool


class StatusTrainingOut(BaseModel):
    training_triggered: bool

class RecommendOut(BaseModel):
    ids: List[int]