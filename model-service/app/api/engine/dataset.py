import json
import os
import pickle
import random
from typing import List
from lightfm.data import Dataset as LightDataset
import pandas as pd
from ..constants import DynamicFieldType, FilePath, DataSource, DatasetState
from ..util.logger import logger
from .load_store import store_fake_users_json, store_data, load_data
import numpy as np
from scipy import sparse
import it_core_news_lg
import spacy
import jsonschema
from jsonschema import validate
from collections import Counter
from string import punctuation
from ..util.lu_generator import generate_learning_units
from ..util.user_generator import generate_users
from ..schemas import DynamicField, LearningUnit, Translation



class Dataset():
    '''
    This dataset class has to be used in order to train the recommender.

    - TO BE REMOVED: use 'build_from_local_json()' in order to use data from local json file(s).
    - use 'build_from_local_db_pickle()' in order to build the dataset from local pickle file(s).
    - use 'build_from_online_db()' in order to use data fetched from the
    remote database.
    '''

    def __init__(self, data_source: int = DataSource.LOCAL_JSON):
        '''
        The Constructor of Dataset class internally uses the Dataset class provided by LightFM: https://making.lyst.com/lightfm/docs/lightfm.data.html
        '''

        # List of users and items
        self.users_list = None
        self.items_list = None

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

        # Test interactions to be used for evaluation purposes
        self.test_interactions = None

        # On Dataset Building delete old dataset and trained model pickle files if necessary
        # depending on data_source
        if data_source == DataSource.LOCAL_DB_PICKLE:
            # Load users, learning units and interactions
            # from local pickle files

            # Delete the old dataset if exists
            if os.path.isfile(FilePath.TEMP_DATASET_PICKLE_PATH):
                os.remove(FilePath.TEMP_DATASET_PICKLE_PATH)
            
            logger.info("Building dataset from local dump.")

            self.__build_from_local_db_pickle()

        elif data_source == DataSource.LOCAL_JSON:
            # TODO: Old - To be removed after dataset switch
            if os.path.isfile(FilePath.TEMP_DATASET_PICKLE_PATH):
                os.remove(FilePath.TEMP_DATASET_PICKLE_PATH)

            self.__build_from_local_json(
                user_data_source=DataSource.LOCAL_USER_JSON)

        elif data_source == DataSource.ONLINE_DB:
            if os.path.isfile(FilePath.TEMP_DATASET_PICKLE_PATH):
                os.remove(FilePath.TEMP_DATASET_PICKLE_PATH)

            self.__build_from_online_db()

        elif data_source == DataSource.PICKLE:

            self.__load_dataset()

        logger.info("Dataset ready.")

    '''
    This method builds a dataset starting from local pickle files
    '''
    def __build_from_local_db_pickle(self):
        
        logger.info("Preparing learning units local dump.")

        # Load learning units data from pickle file
        items_dump = self.__load_lus()

        logger.info("Extracting keywords to enrich learning units local dump.")

        # Upgrade learning units dump with extracted keywords
        # Some learning units might not have any keywords

        # items_dump is now a dictionary
        items_dump = self.__extract_keywords(items_dump)

        logger.info("Keywords successfully extracted. Learning Units ready.")

        # Load users data from pickle file
        # users_dump is a dictionary
        logger.info("Preparing users local dump.")
        users_dump = self.__load_users()

        logger.info("Users successfully extracted.")

        # Build list of all features by using skill:cluster:eqf (encoded) and keywords
        skill_cluster_eqf = list({'skill:' + lu.get('skill') + '-' + 'cluster:' + lu.get('cluster_number') + '-' + 'eqf:' + lu.get('eqf_level') for lu in items_dump})
        keywords = []
        for lu in items_dump:
            if lu.get('extracted_keywords'):
                keywords.extend(lu.get('extracted_keywords'))
            if lu.get('translations')[0].get('keywords'):
                keywords.extend(lu.get('translations')[0].get('keywords'))
            
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

        # Dataset creation
        # It builds the ID mappings: https://making.lyst.com/lightfm/docs/examples/dataset.html#building-the-id-mappings
        #
        # We have to create a mapping between user and item ids
        # of our input data and the indices that will be used internally by our model.
        # Also mappings for user and item features are created.
        self.dataset.fit(
            ([user["id"] for user in self.users_list]),
            ([item["identifier"] for item in self.items_list]),
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
                    user_skill_cluster_eqf.append(f"skill:{str(skill + 1)}-cluster:{str(cluster + 1)}-eqf:{str(user['eqf_levels'][skill][cluster])}")
            
            users_features_list.append(
                (user["id"], user_skill_cluster_eqf))

        # format: [(item1 , [feature1, feature2, ...]), ..]
        items_features_list = list()
        for item in self.items_list:

            kwds = []

            if item.get("extracted_keywords"):
                kwds += item["extracted_keywords"]

            if item.get('translations')[0].get('keywords'):
                kwds.extend(item.get('translations')[0].get('keywords'))    
            
            items_features_list.append(
                (item["identifier"], list(kwds) + [f"skill:{item['skill']}-cluster:{item['cluster_number']}-eqf:{item['eqf_level']}"]))

        self.uf_matrix = self.dataset.build_user_features(users_features_list)
        logger.info("Users features matrix has been built: %s" %
                    repr(self.uf_matrix))

        self.if_matrix = self.dataset.build_item_features(items_features_list)
        logger.info("Items (Learning Units) features matrix has been built: %s" %
                    repr(self.if_matrix))

        logger.info("Dataset with IDs mappings has been built from local data.")

        # store produced data on disk
        self.__store_dataset()

    '''
    This method build a dataset starting from online DB
    '''
    def __build_from_online_db(self):
        # TODO: Load from db using repository methods
        pass

    '''
    TO BE REMOVED: This method builds a dataset starting from local pickle files
    '''
    def __build_from_local_json(self, user_data_source=DataSource.LOCAL_ON_DEMAND_USER):
        '''
        By default generate new users. JSON usage must be specified
        '''

        # Load items from json file
        items_dump = self.__load_items()


        logger.info("Extracting keywords from local dump.")

        # Upgrade items_dump with extracted keywords
        # Some items might not have any keywords
        items_dump = self.__extract_keywords(items_dump)

        logger.info("Keywords successfully extracted.")

        # Load users from json file if necessary
        if user_data_source == DataSource.LOCAL_USER_JSON:
            users_dump = self.__load_users()

        # Build list of features names
        themes_names = list({item.get("theme") for item in items_dump})
        places_names = list({item.get("place") for item in items_dump})
        keywords = []
        for item in items_dump:
            if item.get('extracted_keywords'):
                keywords.extend(item.get('extracted_keywords'))
        keywords = list(set(keywords))

        # User features are a mix of themes, places and national
        self.user_features = themes_names + places_names + ["national"]
        # Item features are a mix of themes, places, national and keywords
        self.item_features = themes_names + places_names + ["national"] + keywords
        # Drop duplicates
        self.user_features = list(dict.fromkeys(self.user_features))
        self.item_features = list(dict.fromkeys(self.item_features))

        # Extract items with valid structure
        self.items_list = items_list = self.__extract_items_from_local_dump(
            items_dump)

        # Generate users if necessary
        if user_data_source == DataSource.LOCAL_ON_DEMAND_USER:
            self.users_list = self.__build_fake_users_list(
                items_list=items_list, themes_names=themes_names, places_names=places_names, num_interactions=200)
        else:
            self.users_list = users_dump

        # Dataset creation
        # It builds the ID mappings: https://making.lyst.com/lightfm/docs/examples/dataset.html#building-the-id-mappings
        #
        # We have to create a mapping between user and item ids
        # of our input data and the indices that will be used internally by our model.
        # Also mappings for user and item features are created.
        self.dataset.fit(
            ([user["user_id"] for user in self.users_list]),
            ([item["item_id"] for item in self.items_list]),
            user_features=self.user_features,
            item_features=self.item_features
        )

        # Building the interaction matrix
        #
        # parameter: (iterable of (user_id, item_id) or (user_id, item_id, weight))
        #
        # https://making.lyst.com/lightfm/docs/lightfm.data.html#lightfm.data.Dataset.build_interactions
        self.interactions = self.__build_interactions_matrix(self.users_list)

        (self.test_interactions, _) = self.dataset.build_interactions(
            self.__build_fake_test_interactions(self.users_list, self.items_list)
        )

        # Building the items and users feature matrixes
        # format: [(user1 , [feature1, feature2, ...]), ..]
        users_features_list = list()
        for user in self.users_list:
            users_features_list.append(
                (user["user_id"], user["fav_themes"] + user["fav_places"]))

        # format: [(item1 , [feature1, feature2, ...]), ..]
        items_features_list = list()
        for item in self.items_list:
            kwds = []
            if item.get("extracted_keywords"):
                kwds += item["extracted_keywords"]
            items_features_list.append(
                (item["item_id"], list(dict.fromkeys([item["theme"], item["place"]] + kwds))))

        self.uf_matrix = self.dataset.build_user_features(users_features_list)
        logger.info("Users features matrix has been built: %s" %
                    repr(self.uf_matrix))

        self.if_matrix = self.dataset.build_item_features(items_features_list)
        logger.info("Items features matrix has been built: %s" %
                    repr(self.if_matrix))

        logger.info("Dataset with IDs mappings has been built from local data.")

        # store produced data on disk
        self.__store_dataset()

    '''
    This method reads learning units from a local pickle file.
    In case the pickle file does not exist it builds the pickle file
    starting from a basic json file.
    '''
    def __load_lus(self):
        
        lus_data = list()

        if os.path.isfile(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH):
            logger.info("Loading learning units from pickle file.")
            lus_data = load_data(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH)
        else:
            logger.info("Generating learning units from scratch.")

            lus_data = generate_learning_units()

            # save as pickle
            with open(os.getcwd() + '/' + FilePath.LU_PICKLE_PATH, 'wb') as fle:
                pickle.dump(lus_data, fle, protocol=pickle.HIGHEST_PROTOCOL)

        return lus_data


    '''
    TO BE REMOVED: This method reads items from a local json file
    '''
    def __load_items(self):
        item_data = list()

        # Get item json schema
        json_schema = self.__get_json_schema(FilePath.ITEM_SCHEMA_JSON_PATH)

        # Open json file
        if os.path.isfile(os.getcwd() + '/' + FilePath.ITEM_JSON_PATH):
            with open(os.getcwd() + '/' + FilePath.ITEM_JSON_PATH, 'r') as f:
                # Read json file as python dictionary
                try:
                    item_data = json.load(f)
                    if not item_data:
                        raise KeyError()
                except KeyError as ke:
                    logger.error(f"Error when accessing source `{FilePath.ITEM_JSON_PATH}` file: {ke}")
                    exit()
                except json.JSONDecodeError as jde:
                    logger.error(f"Error when decoding `{FilePath.ITEM_JSON_PATH}` file: {jde}")
                    exit()

        try:
            validate(instance=item_data, schema=json_schema)
        except jsonschema.ValidationError as ve:
                logger.error(f"Error when validating json `{FilePath.ITEM_JSON_PATH}`: {ve}")
                exit()

        return item_data

    '''
    This method reads user data from a local pickle file.
    In case the pickle file does not exist it builds the pickle file
    starting from a basic json file.
    '''
    def __load_users(self):

        user_json = list()
        user_data = list()

        if os.path.isfile(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH):
            logger.info("Loading users from pickle file.")
            user_data = load_data(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH)
            user_json = pd.DataFrame.from_records([user.dict() for user in user_data]).to_dict('records')
        else:
            logger.info("Generating users from scratch.")

            user_json, user_data = generate_users()
            # save as json
            with open(os.getcwd() + '/' + FilePath.USER_JSON_PATH, 'w') as f:
                json.dump(user_json, f)

            # save as pickle
            with open(os.getcwd() + '/' + FilePath.USER_PICKLE_PATH, 'wb') as fle:
                pickle.dump(user_data, fle, protocol=pickle.HIGHEST_PROTOCOL)

        return user_json

    # TODO: To be removed - old method to load users from json file
    # def __load_users(self):
    #     user_data = list()

    #     # Get item json schema
    #     json_schema = self.__get_json_schema(FilePath.USER_SCHEMA_JSON_PATH)

    #     # Open json  file
    #     if os.path.isfile(os.getcwd() + '/' + FilePath.USER_JSON_PATH):
    #         with open(os.getcwd() + '/' + FilePath.USER_JSON_PATH, 'r') as f:
    #             # Read json file as python dictionary
    #             try:
    #                 user_data = json.load(f)
    #                 if not user_data:
    #                     raise KeyError()
    #             except KeyError as ke:
    #                 logger.error(f"Error when accessing source `{FilePath.USER_JSON_PATH}` file: {ke}")
    #                 exit()
    #             except json.JSONDecodeError as jde:
    #                 logger.error(f"Error when decoding `{FilePath.USER_JSON_PATH}` file: {jde}")
    #                 exit()

    #     try:
    #         validate(instance=user_data, schema=json_schema)
    #     except jsonschema.ValidationError as ve:
    #             logger.error(f"Error when validating json `{FilePath.USER_JSON_PATH}`: {ve}")
    #             exit()

    #     return user_data

    # def __get_json_schema(self, filename: str) -> dict:
    #     if os.path.isfile(os.getcwd() + '/' + filename):
    #         with open(os.getcwd() + '/' + filename, 'r') as f_schema:
    #             try:
    #                 json_schema = json.load(f_schema)
    #                 if not json_schema:
    #                     raise KeyError()
    #             except KeyError as ke:
    #                 logger.error(f"Error when accessing source `{filename}` file: {ke}")
    #                 exit()
    #             except json.JSONDecodeError as jde:
    #                 logger.error(f"Error when decoding `{filename}` file: {jde}")
    #                 exit()

    #     return json_schema


    def __extract_items_from_local_dump(self, local_items_dump):
        '''
        Reads imported items and returns the ones having the correct strutcture as a list.
        '''
        items_list = []

        for item_dict in local_items_dump:
            if item_dict.get("entities") and item_dict.get("theme") and item_dict.get("place"):
                items_list.append(item_dict)

        logger.info(f"Number of dropped item: {len(local_items_dump) - len(items_list)}")

        return items_list

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

    def __build_fake_test_interactions(self, users_list, items_list):
        '''
        Creates a list of (user_id, item_id) tuples starting from the two lists of users and all the available items.
        '''
        interactions = set()

        for user in users_list:
            num_items_to_sel = 30  # select 30 random items for the user to use as test interactions
            random.shuffle(items_list)
            for item in items_list:
                if item["item_id"] not in user["interactions"] and item["theme"] in user["fav_themes"] and item["place"] in user["fav_places"]:
                    interactions.add((user["user_id"], item["item_id"]))

                num_items_to_sel -= 1
                if num_items_to_sel == 0:
                    break

        return interactions

    def __build_fake_users_list(self, items_list, themes_names, places_names, num_users: int = 10000, num_fav_themes: int = 3, num_fav_places: int = 2, num_interactions: int = 100) -> list:
        '''
        Builds a fake users list with interactions populated from items_list
        '''
        users_list = list()

        num_items = len(items_list)
        num_themes = len(themes_names)
        num_places = len(places_names)

        while num_users > 0:
            user_dict = dict()

            # Fake user id
            user_dict["user_id"] = num_users
            user_dict["fav_themes"] = set(
                random.sample(themes_names, num_fav_themes))
            user_dict["fav_places"] = set(
                random.sample(places_names, num_fav_places))
            # always add national items
            user_dict["fav_places"].add("national")
            user_dict["fav_themes"] = list(user_dict["fav_themes"])
            user_dict["fav_places"] = list(user_dict["fav_places"])

            random.shuffle(items_list)
            interactions = [item["item_id"] for item in items_list if item["theme"]
                            in user_dict["fav_themes"] and item["place"] in user_dict["fav_places"] and item is not None]
            user_dict["interactions"] = interactions[:num_interactions]

            users_list.append(user_dict)
            num_users = num_users - 1

        store_fake_users_json(users_list, sorting_key='user_id')
        logger.info(
            "\nJson file containing fake generated users stored on disk!")

        return users_list


    '''
    This method prepares a dictionary named "state", containing
    all the dataset data built, and stores it on disk in order
    to be loaded later.
    '''
    def __store_dataset(self):
        state = {
            DatasetState.INTERACTIONS: self.interactions,
            DatasetState.TEST_INTERACTIONS: self.test_interactions,
            DatasetState.USER_FEATURES_MATRIX: self.uf_matrix,
            DatasetState.ITEM_FEATURES_MATRIX: self.if_matrix,
            DatasetState.USER_FEATURES: self.user_features,
            DatasetState.ITEM_FEATURES: self.item_features,
            DatasetState.USERS_LIST: self.users_list,
            DatasetState.ITEMS_LIST: self.items_list,
            DatasetState.DATASET: self.dataset
        }
        store_data(state, FilePath.TEMP_DATASET_PICKLE_PATH)

    def __load_dataset(self):
        state = load_data(FilePath.DATASET_PICKLE_PATH)
        self.interactions = state[DatasetState.INTERACTIONS]
        self.test_interactions = state[DatasetState.TEST_INTERACTIONS]
        self.uf_matrix = state[DatasetState.USER_FEATURES_MATRIX]
        self.if_matrix = state[DatasetState.ITEM_FEATURES_MATRIX]
        self.user_features = state[DatasetState.USER_FEATURES]
        self.item_features = state[DatasetState.ITEM_FEATURES]
        self.users_list = state[DatasetState.USERS_LIST]
        self.items_list = state[DatasetState.ITEMS_LIST]
        self.dataset = state[DatasetState.DATASET]

    '''
    This method enriches learning units dump with extracted
    keywords via named entity recognition
    '''
    def __extract_keywords(self, lus_dump: List[LearningUnit]):

        # Load proper model
        model = it_core_news_lg.load()

        # For each learning unit get extracted keywords
        lu_keywords = []
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
            
            keywords = list(set((self.__get_keywords(model, plain_text))))
            lu_keywords.append(keywords)

        lus_df = pd.DataFrame.from_records([lu.dict() for lu in lus_dump])
        lus_df['extracted_keywords'] = lu_keywords
        lus_dict = lus_df.to_dict('records')

        # save as json
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


    def build_fake_new_user_features(self, num_features: int = 10) -> list:
        '''
        Builds a fake new user features list from the overall set of features.
        '''
        new_user_features = []

        new_user_features = list(set(
            random.sample(self.user_features, num_features)))

        return new_user_features


    def format_new_user_input(self, user_feature_map, user_feature_list):
        num_features = len(user_feature_list)
        normalised_val = 1.0
        target_indices = []

        for feature in user_feature_list:
            try:
                target_indices.append(user_feature_map[feature])
            except KeyError:
                logger.info("New user feature encountered '{}'".format(feature))
                pass

        new_user_features = np.zeros(len(user_feature_map.keys()))
        for i in target_indices:
            new_user_features[i] = normalised_val
        new_user_features = sparse.csr_matrix(new_user_features)
        return(new_user_features)
