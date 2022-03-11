from fastapi import APIRouter, status
from .service import ModelService
from .models import StatusOut, RecommendOut, UserFeaturesIn


model = APIRouter()


@model.get('/status', response_model=StatusOut, status_code=status.HTTP_200_OK)
def get_status():
    '''
        This endpoint gives us some informations about Model Service status:
            - Server running
            - Model trained
    '''
    return ModelService().get_model_status()


@model.post('/recommendations/user/features', response_model=RecommendOut, status_code=status.HTTP_200_OK)
def get_recommendations_for_new_user(user_features: UserFeaturesIn, num_pred: int = 100):
    '''
        This endpoint allows us to obtain recommendations for a new user (with zero interactions) given some features.
        Features must be sent as a list of strings and must belong to the already existing (into the recommender) set of features.
        It is possible to specify the number of predictions to obtain via query parameter 'num_pred'.
    '''
    return ModelService().get_recommendations_for_new_user(user_features=user_features.user_features, num_pred=num_pred)    


@model.get('/recommendations/user/{user_id}', response_model=RecommendOut, status_code=status.HTTP_200_OK)
def get_recommendations_for_user(user_id: int, num_pred: int = 100):
    '''
        This endpoint allows us to obtain recommendations for a given user given its id 'user_id'.
        It is possible to specify the number of predictions to obtain via query parameter 'num_pred'.
    '''
    return ModelService().get_recommendations_for_user(user_id=user_id, num_pred=num_pred)



@model.get('/recommendations/item/{item_id}', response_model=RecommendOut, status_code=status.HTTP_200_OK)
def get_similar_items(item_id: int, num_pred: int = 100):
    '''
        This endpoint allows us to obtain recommendations (similar items) for a given item given its id 'item_id'.
        Similarity is given by cosine similarity.
        It is possible to specify the number of predictions to obtain via query parameter 'num_pred'.
    '''
    return ModelService().get_similar_items(item_id=item_id, num_pred=num_pred)