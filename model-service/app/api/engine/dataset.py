import json
import os
import random
from typing import List
from lightfm.data import Dataset as LightDataset
import pandas as pd
from bs4 import BeautifulSoup
from ..constants import DynamicFieldType, FilePath, DataSource, DatasetState, SkillClusterEqf
from ..util.logger import logger
from .load_store import store_data, load_data
import en_core_web_sm
from ..schemas import LearningUnit
from ..repository import lm_learning_unit as lm_lu_repository, learning_unit as lu_repository, user as user_repository, database

class Dataset():
    '''
    This dataset class has to be used in order to train the recommender.

    - use 'build_from_local_db_pickle()' in order to build the dataset from local pickle file(s).
    - use 'build_from_online_db()' in order to use data fetched from the
    remote database.
    '''

    def __init__(self, data_source: int = DataSource.ONLINE_DB):
        '''
        The Constructor of Dataset class internally uses the Dataset class provided by LightFM: https://making.lyst.com/lightfm/docs/lightfm.data.html
        '''

        # List of users and items
        self.users_list = None
        self.items_list = None
        self.lm_items_list = None

        # LightFM dataset 'LightDataset' object
        # https://making.lyst.com/lightfm/docs/lightfm.data.html
        self.dataset = LightDataset()

        # Interactions object for interactions between users and items
        # A row represents a user and a column represents an item
        self.interactions = None

        # Features for each user
        # A row represents a user, a column represents a feature listed
        # in in self.user_features List
        self.uf_matrix = None

        # Features for each item
        # A row represents an item, a column represents a feature in self.item_features
        self.if_matrix = None

        # List of user features
        self.user_features = None

        # List of item features
        self.item_features = None

        # On Dataset Building delete old dataset and trained model pickle files if necessary
        # depending on data_source
        if data_source == DataSource.ONLINE_DB:
            if os.path.isfile(FilePath.TEMP_DATASET_PICKLE_PATH):
                os.remove(FilePath.TEMP_DATASET_PICKLE_PATH)

            self.__build_from_online_db()

        elif data_source == DataSource.PICKLE:

            self.__load_dataset()

        logger.info("Dataset ready.")

    '''
    This method build a dataset starting from online DB
    '''
    def __build_from_online_db(self):
        # TODO: Load from db using repository methods
        logger.info("Preparing learning units online dump.")

        # Load learning units data from online DB
        items_dump = lu_repository.get_all(next(database.get_db()))

        logger.info("Extracting keywords to enrich learning units local dump.")

        # Upgrade learning units dump with extracted keywords
        # Some learning units might not have any keywords

        # items_dump is now a dictionary
        items_dump = self.__extract_keywords(items_dump)

        logger.info("Keywords successfully extracted. Learning Units ready.")

        logger.info("Preparing labour market learning units local dump.")
        lm_items_dump = lm_lu_repository.get_all()

        # Load users data from pickle file
        # users_dump is a dictionary
        logger.info("Preparing users local dump.")
        users_dump = user_repository.get_all()

        logger.info("Users successfully extracted.")

        # Build list of all features by using skill:cluster:eqf (encoded) and keywords
        skill_cluster_eqf = self.__add_required_skill_cluster_eqf_triplets()
        keywords = []
        for lu in items_dump:
            if lu.get('extracted_keywords'):
                keywords.extend(lu.get('extracted_keywords'))
            if lu.get('translations')[0].get('keywords'):
                kws = lu.get('translations')[0].get('keywords').split(",")
                keywords.extend(kws)
            
        keywords = list(set(keywords))

        # User features are the eqf levels registered for each cluster of each skill
        self.user_features = skill_cluster_eqf
        # Learning unit features are a mix of hand written keywords, extracted keywords and the assigned skill/cluster/eqf
        self.item_features = skill_cluster_eqf + keywords

        # Drop duplicates
        self.user_features = list(dict.fromkeys(self.user_features))
        self.item_features = list(dict.fromkeys(self.item_features))

        self.items_list = items_dump
        self.users_list = users_dump
        self.lm_items_list = lm_items_dump

        # Dataset creation
        # It builds the ID mappings: https://making.lyst.com/lightfm/docs/examples/dataset.html#building-the-id-mappings
        #
        # We have to create a mapping between user and item ids
        # of our input data and the indices that will be used internally by our model.
        # Also mappings for user and item features are created.
        self.dataset.fit(
            ([user["id"] for user in self.users_list]),
            ([item["id"] for item in self.items_list]),
            user_features=self.user_features,
            item_features=self.item_features
        )

        # Building the interaction matrix
        #
        # parameter: (iterable of (user_id, item_id) or (user_id, item_id, weight))
        #
        # https://making.lyst.com/lightfm/docs/lightfm.data.html#lightfm.data.Dataset.build_interactions
        self.interactions = self.__build_interactions_matrix(self.users_list)

        # Building the items and users feature matrixes
        # format: [(user1 , [feature1, feature2, ...]), ..]
        users_features_list = list()
        for user in self.users_list:
            user_skill_cluster_eqf = list()
            for skill in range(4):
                for cluster in range(3):
                    try:
                        user_skill_cluster_eqf.append(f"skill:{str(skill + 1)}-cluster:{str(cluster + 1)}-eqf:{str(user['eqf_levels'][skill][cluster])}")
                    except IndexError:
                        logger.error(f"User {user['id']} has no eqf_levels registered")
            
            users_features_list.append(
                (user["id"], user_skill_cluster_eqf))

        # format: [(item1 , [feature1, feature2, ...]), ..]
        items_features_list = list()
        for item in self.items_list:

            kwds = []

            if item.get("extracted_keywords"):
                kwds += item["extracted_keywords"]

            if item.get('translations')[0].get('keywords'):
                kws = item.get('translations')[0].get('keywords').split(",")
                kwds.extend(kws)    
            
            items_features_list.append(
                (item["id"], list(kwds) + [f"skill:{item['skill']}-cluster:{item['cluster_number']}-eqf:{item['eqf_level']}"]))

        self.uf_matrix = self.dataset.build_user_features(users_features_list)
        logger.info("Users features matrix has been built: %s" %
                    repr(self.uf_matrix))

        self.if_matrix = self.dataset.build_item_features(items_features_list)
        logger.info("Items (Learning Units) features matrix has been built: %s" %
                    repr(self.if_matrix))

        logger.info("Dataset with IDs mappings has been built from local data.")

        # store produced data on disk
        self.__store_dataset()





    def __build_interactions_matrix(self, users_with_items_interactions: list):
        '''
        Building the interaction matrix

        parameter: (iterable of (user_id, item_id) or (user_id, item_id, weight))

        https://making.lyst.com/lightfm/docs/lightfm.data.html#lightfm.data.Dataset.build_interactions
        '''
        '''
        Creates a list of (user_id, item_id) tuples starting from the two lists of users and their interactions.
        '''
        user_item_interactions = set()
        for user in users_with_items_interactions:
            for lu in user["completed_lus"]:
                user_item_interactions.add((user["id"], lu["lu_id"]))

        (interactions, _) = self.dataset.build_interactions(
            user_item_interactions)

        logger.info("Interaction matrix built starting from %s user-item interaction tuples: %s" %
                    (len(user_item_interactions), repr(interactions)))
        return interactions

    '''
    This method prepares a dictionary named "state", containing
    all the dataset data built, and stores it on disk in order
    to be loaded later.
    '''
    def __store_dataset(self):
        state = {
            DatasetState.INTERACTIONS: self.interactions,
            DatasetState.USER_FEATURES_MATRIX: self.uf_matrix,
            DatasetState.ITEM_FEATURES_MATRIX: self.if_matrix,
            DatasetState.USER_FEATURES: self.user_features,
            DatasetState.ITEM_FEATURES: self.item_features,
            DatasetState.USERS_LIST: self.users_list,
            DatasetState.ITEMS_LIST: self.items_list,
            DatasetState.LM_ITEMS_LIST: self.lm_items_list,
            DatasetState.DATASET: self.dataset
        }
        store_data(state, FilePath.TEMP_DATASET_PICKLE_PATH)

    def __load_dataset(self):
        state = load_data(FilePath.DATASET_PICKLE_PATH)
        self.interactions = state[DatasetState.INTERACTIONS]
        self.uf_matrix = state[DatasetState.USER_FEATURES_MATRIX]
        self.if_matrix = state[DatasetState.ITEM_FEATURES_MATRIX]
        self.user_features = state[DatasetState.USER_FEATURES]
        self.item_features = state[DatasetState.ITEM_FEATURES]
        self.users_list = state[DatasetState.USERS_LIST]
        self.items_list = state[DatasetState.ITEMS_LIST]
        self.lm_items_list = state[DatasetState.LM_ITEMS_LIST]
        self.dataset = state[DatasetState.DATASET]

    '''
    This method enriches learning units dump with extracted
    keywords via named entity recognition
    '''
    def __extract_keywords(self, lus_dump: List[LearningUnit]):

        # Load proper model
        model = en_core_web_sm.load()

        # For each learning unit get extracted keywords
        lu_keywords = []
        logger.info(f"Starting extraction of {len(lus_dump)} documents.")
        for lu in lus_dump:
            # Take just english translation
            translation = list(filter(lambda t: t.language_name=="en", lu.translations))[0]
            plain_text = translation.title
            plain_text += " " + translation.subtitle
            plain_text += " " + translation.introduction
            plain_text += " " + translation.text_area

            # Take all dynamic fields of type paragraph, memory box, reference and language point
            dynamic_fields = list(filter(lambda f: f.type==DynamicFieldType.PARAGRAPH or f.type==DynamicFieldType.MEMORY_BOX or f.type==DynamicFieldType.REFERENCE or f.type==DynamicFieldType.LANGUAGE_POINT, translation.dynamic_fields))
            for field in dynamic_fields:
                plain_text += " " + field.content

            # Remove HTML tags, if any
            cleaned_text = BeautifulSoup(plain_text)
            cleaned_text = cleaned_text.get_text()
            
            keywords = list(set((self.__get_keywords(model, cleaned_text))))
            lu_keywords.append(keywords)

        lus_df = pd.DataFrame.from_records([lu.dict() for lu in lus_dump])
        lus_df['extracted_keywords'] = lu_keywords
        lus_dict = lus_df.to_dict('records')

        # ENABLE TO DEBUG: save as json
        with open(os.getcwd() + '/' + FilePath.LU_JSON_PATH, 'w') as f:
            json.dump(lus_dict, f)

        return lus_dict

    '''
    This method gets keywords via named entity recognition from a plain text
    '''
    def __get_keywords(self, nlp_model, plain_text):
        res = []
        doc = nlp_model(plain_text)

        # Recognized entities
        if doc.ents:
            for entity in doc.ents:
                res.append(entity.text.lower())
                
        return res


    '''
    This method returns all possible combinations (triplets) of skills, clusters and eqfs
    '''
    def __add_required_skill_cluster_eqf_triplets(self):
        skill_cluster_eqf = list()
        skill_cluster_eqf.extend([
            # Skill 1
            SkillClusterEqf.SCE_1_1_2,
            SkillClusterEqf.SCE_1_1_3,
            SkillClusterEqf.SCE_1_1_4,
            SkillClusterEqf.SCE_1_2_2,
            SkillClusterEqf.SCE_1_2_3,
            SkillClusterEqf.SCE_1_2_4,
            SkillClusterEqf.SCE_1_3_2,
            SkillClusterEqf.SCE_1_3_3,
            SkillClusterEqf.SCE_1_3_4,
            # Skill 2
            SkillClusterEqf.SCE_2_1_2,
            SkillClusterEqf.SCE_2_1_3,
            SkillClusterEqf.SCE_2_1_4,
            SkillClusterEqf.SCE_2_2_2,
            SkillClusterEqf.SCE_2_2_3,
            SkillClusterEqf.SCE_2_2_4,
            SkillClusterEqf.SCE_2_3_2,
            SkillClusterEqf.SCE_2_3_3,
            SkillClusterEqf.SCE_2_3_4,
            # Skill 3
            SkillClusterEqf.SCE_3_1_2,
            SkillClusterEqf.SCE_3_1_3,
            SkillClusterEqf.SCE_3_1_4,
            SkillClusterEqf.SCE_3_2_2,
            SkillClusterEqf.SCE_3_2_3,
            SkillClusterEqf.SCE_3_2_4,
            SkillClusterEqf.SCE_3_3_2,
            SkillClusterEqf.SCE_3_3_3,
            SkillClusterEqf.SCE_3_3_4,
            # Skill 4
            SkillClusterEqf.SCE_4_1_2,
            SkillClusterEqf.SCE_4_1_3,
            SkillClusterEqf.SCE_4_1_4,
            SkillClusterEqf.SCE_4_2_2,
            SkillClusterEqf.SCE_4_2_3,
            SkillClusterEqf.SCE_4_2_4,
            SkillClusterEqf.SCE_4_3_2,
            SkillClusterEqf.SCE_4_3_3,
            SkillClusterEqf.SCE_4_3_4,
        ])

        return skill_cluster_eqf

