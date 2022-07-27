from pydantic import BaseModel
from typing import List

class UserFeaturesIn(BaseModel):
    user_features: List[str]


class StatusOut(BaseModel):
    server_on: bool
    model_trained: bool
    random_mode_on: bool


class StatusTrainingOut(BaseModel):
    training_completed: bool

class RecommendOut(BaseModel):
    is_labour_market: bool
    ids: List[int]