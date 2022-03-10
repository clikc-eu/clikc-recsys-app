from typing import List
from .engine import training, prediction
from .models import RecommendOut, StatusOut

class ModelService():

    def get_model_status(self):
        return StatusOut(server_on=True, model_trained=training.check_trained_model())


    def get_recommendations_for_user(self, user_id: int, num_pred: int):
        return RecommendOut(ids=prediction.predict_for_user(user_id=user_id, num_pred=num_pred))


    def get_similar_items(self, item_id: int, num_pred: int):
        return RecommendOut(ids=prediction.predict_items_for_known_item(item_id=item_id, num_pred=num_pred))