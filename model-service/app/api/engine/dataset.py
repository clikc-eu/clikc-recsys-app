import json
import os
import random
from lightfm.data import Dataset as LightDataset
import pandas as pd
from ..constants import FilePath, DataSource, DatasetState
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



class Dataset():
    '''
    This dataset class has to be used in order to train the recommender.

    - use 'build_from_local_json()' in order to use data from local json file(s).
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
        if data_source == DataSource.LOCAL_JSON:

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


    def __build_from_online_db():
        # TODO: Load from db using repository methods
        pass


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

    def __load_users(self):
        user_data = list()

        # Get item json schema
        json_schema = self.__get_json_schema(FilePath.USER_SCHEMA_JSON_PATH)

        # Open json  file
        if os.path.isfile(os.getcwd() + '/' + FilePath.USER_JSON_PATH):
            with open(os.getcwd() + '/' + FilePath.USER_JSON_PATH, 'r') as f:
                # Read json file as python dictionary
                try:
                    user_data = json.load(f)
                    if not user_data:
                        raise KeyError()
                except KeyError as ke:
                    logger.error(f"Error when accessing source `{FilePath.USER_JSON_PATH}` file: {ke}")
                    exit()
                except json.JSONDecodeError as jde:
                    logger.error(f"Error when decoding `{FilePath.USER_JSON_PATH}` file: {jde}")
                    exit()

        try:
            validate(instance=user_data, schema=json_schema)
        except jsonschema.ValidationError as ve:
                logger.error(f"Error when validating json `{FilePath.USER_JSON_PATH}`: {ve}")
                exit()

        return user_data

    def __get_json_schema(self, filename: str) -> dict:
        if os.path.isfile(os.getcwd() + '/' + filename):
            with open(os.getcwd() + '/' + filename, 'r') as f_schema:
                try:
                    json_schema = json.load(f_schema)
                    if not json_schema:
                        raise KeyError()
                except KeyError as ke:
                    logger.error(f"Error when accessing source `{filename}` file: {ke}")
                    exit()
                except json.JSONDecodeError as jde:
                    logger.error(f"Error when decoding `{filename}` file: {jde}")
                    exit()

        return json_schema


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
            for item_id in user["interactions"]:
                user_item_interactions.add((user["user_id"], item_id))

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


    def __extract_keywords(self, items_dump):

        # Load proper model
        model = it_core_news_lg.load()

        # For each item get extracted keywords
        item_keywords = []
        for item in items_dump:
            keywords = list(set((self.__get_keywords(model, item.get('title')))))
            item_keywords.append(keywords)

        items_df = pd.DataFrame(items_dump)
        items_df['extracted_keywords'] = item_keywords

        return items_df.to_dict('records')


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
