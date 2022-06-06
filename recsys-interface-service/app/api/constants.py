class ModelServiceUrls():
    STATUS_URL = "http://model_service:8000/api/v1/model/status"
    TRAIN_URL = "http://model_service:8000/api/v1/model/train"
    RECS_USER_URL = "http://model_service:8000/api/v1/model/recommendations/user/"
    RECS_USER_FEATURES_URL = "http://model_service:8000/api/v1/model/recommendations/user/features"
    RECS_ITEM_URL = "http://model_service:8000/api/v1/model/recommendations/item/"


class FilePath():
    # Log file
    LOG_PATH = "main.log"