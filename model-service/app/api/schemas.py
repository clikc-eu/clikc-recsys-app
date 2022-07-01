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
    ids: List[int]

'''
It contains texts and links to multimedia elements. Here only text fields are used.
'''
class DynamicField(BaseModel):
    type: str
    content: str

'''
Model of one of the possible translations of a Learning Unit.
It contains the didactic content.
'''
class Translation(BaseModel):
    language_name: str  # language code
    title: str
    subtitle: str
    keywords: List[str]       # list of keywords
    introduction: str
    text_area: str
    dynamic_fields: List[DynamicField]


'''
Model of a single Learning Unit
'''
class LearningUnit(BaseModel):
    identifier: str
    cluster_number: str
    skill: str
    eqf_level: str
    translations: List[Translation]

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
    lu_id: str # Learning Unit identifier
    result: float   # Result in percentage

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
