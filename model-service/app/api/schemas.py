from typing import List
from pydantic import BaseModel

'''
Model of the features used by a new user
'''
class UserFeaturesIn(BaseModel):
    user_features: List[str]

'''
Output status of the Model microservice
'''
class StatusOut(BaseModel):
    server_on: bool
    model_trained: bool
    random_mode_on: bool

'''
Output status of training phase: it shows if training has been triggered
'''
class StatusTrainingOut(BaseModel):
    training_triggered: bool

'''
List of the ids of the recommendations given as output to the user
'''
class RecommendOut(BaseModel):
    is_labour_market: bool
    ids: List[int]

'''
It contains texts and links to multimedia elements. Here only text fields are used.
'''
class DynamicField(BaseModel):
    type: str
    content: str

    class Config():
        orm_mode = True

'''
Model of one of the possible translations of a Learning Unit.
It contains the didactic content.
'''
class Translation(BaseModel):
    language_name: str  # language code
    title: str
    subtitle: str
    #keywords: List[str]       # list of keywords
    keywords: str       # list of keywords, comma separated
    introduction: str
    text_area: str
    dynamic_fields: List[DynamicField]

    class Config():
        orm_mode = True


'''
Model of a single Learning Unit
'''
class LearningUnit(BaseModel):
    id: int
    identifier: str
    cluster_number: str
    skill: str
    eqf_level: str
    translations: List[Translation]

    class Config():
        orm_mode = True

'''
Model of a single labour market Learning Unit
'''
class LMLearningUnit(BaseModel):
    id: int

class CompletedLMLearningUnit(LMLearningUnit):
    timestamp: float  # Timestamp when labour market learning unit was registered as completed

'''
Cluster model
'''
class Cluster(BaseModel):
    skill: str
    cluster: str

'''
Completed Learning Unit model
'''
class CompletedLearningUnit(BaseModel):
    lu_id: int # Learning Unit identifier
    result: float   # Result in percentage
    timestamp: float  # Timestamp when learning unit was registered as completed
    liked: bool       # Assume like/dislike must be mandatory. If liked -> True. If not liked -> False.

'''
User model
'''
class User(BaseModel):
    id: str
    first_name: str
    last_name: str
    username: str
    email: str
    gender: str
    eqf_levels: List[List[str]] # 3 clusters for each skill
    fav_clusters: List[Cluster] # 3 favourite clusters from assessment phase
    completed_lus: List[CompletedLearningUnit]  # List of completed Learning Units
    completed_lm_lus: List[CompletedLMLearningUnit]      # List of completet labour market Learning Units
    lu_counter: int     # Counter for viewed Learning Unit (LU). Every 5 LU is set to zero and
                        # a LU from labour market is recommended
