import json
import os
import random
from lightfm.data import Dataset as LightDataset
from constants import FilePath, DataSource, DatasetState
from util.logger import logger
from engine.load_store import store_fake_users_json, store_data, load_data


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
        self.users_list = []
        self.items_list = []

        # LightFM dataset 'LightDataset' object
        # https://making.lyst.com/lightfm/docs/lightfm.data.html
        self.dataset = LightDataset()

        # Interactions object for interactions between users and items
        # A row represents a user and a column represents an item
        self.interactions = None

        # Features for each user
        # A row represents a user, a column represents a feature listed
        # in in self.user_features List
        self.user_features_matrix = None

        # Features for each item
        # A row represents an item, a column represents a feature in self.item_features
        self.item_features_matrix = None

        # List of user features
        self.user_features = None

        # List of item features
        self.item_features = None

        # Test interactions to be used for evaluation purposes
        self.test_interactions = None

        # On Dataset Building delete old dataset and trained model pickle files if necessary
        # depending on data_source
        if data_source == DataSource.LOCAL_JSON:

            if os.path.isfile(FilePath.DATASET_PICKLE_PATH):
                os.remove(FilePath.DATASET_PICKLE_PATH)
                if os.path.isfile(FilePath.TRAINED_MODEL_PICKLE_PATH):
                    os.remove(FilePath.TRAINED_MODEL_PICKLE_PATH)

            self.__build_from_local_json(
                user_data_source=DataSource.LOCAL_USER_JSON)

        elif data_source == DataSource.ONLINE_DB:
            if os.path.isfile(FilePath.DATASET_PICKLE_PATH):
                os.remove(FilePath.DATASET_PICKLE_PATH)
                if os.path.isfile(FilePath.TRAINED_MODEL_PICKLE_PATH):
                    os.remove(FilePath.TRAINED_MODEL_PICKLE_PATH)

            self.__build_from_online_db()

        elif data_source == DataSource.PICKLE:
            if os.path.isfile(FilePath.TRAINED_MODEL_PICKLE_PATH):
                os.remove(FilePath.TRAINED_MODEL_PICKLE_PATH)

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

        # Load users from json file if necessary
        if user_data_source == DataSource.LOCAL_USER_JSON:
            users_dump = self.__load_users()

        # Build list of features names
        themes_names = list({item.get("theme") for item in items_dump})
        places_names = list({item.get("place") for item in items_dump})

        # User features are a mix of themes, places and national
        self.user_features = themes_names + places_names + ["national"]
        # Drop duplicates
        self.user_features = list(dict.fromkeys(self.user_features))
        self.item_features = self.user_features

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
            ([article["item_id"] for article in items_list]),
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
            self.__build_fake_test_interactions(self.users_list, items_list)
        )

        # Building the items and users feature matrixes
        # format: [(user1 , [feature1, feature2, ...]), ..]
        users_features_list = list()
        for user in self.users_list:
            users_features_list.append(
                (user["user_id"], user["fav_themes"] + user["fav_places"]))

        # format: [(item1 , [feature1, feature2, ...]), ..]
        items_features_list = list()
        for item in items_list:
            items_features_list.append(
                (item["item_id"], [item["theme"], item["place"]]))

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
        # Open json  file
        f = open(os.getcwd() + '/' + FilePath.ITEM_JSON_PATH)

        # Read json file as python dictionary
        item_data = json.load(f)

        # Close json file
        f.close()
        return item_data

    def __load_users(self):
        # Open json  file
        f = open(os.getcwd() + '/' + FilePath.USER_JSON_PATH)

        # Read json file as python dictionary
        user_data = json.load(f)

        # Close json file
        f.close()
        return user_data

    def __extract_items_from_local_dump(self, local_items_dump):
        '''
        Reads imported items and returns the ones having the correct strutcture as a list.
        '''
        items_list = []

        for item_dict in local_items_dump:
            if item_dict.get("entities") and item_dict.get("theme") and item_dict.get("place"):
                items_list.append(item_dict)

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
            DatasetState.USER_FEATURES_MATRIX: self.user_features_matrix,
            DatasetState.ITEM_FEATURES_MATRIX: self.item_features_matrix,
            DatasetState.USER_FEATURES: self.user_features,
            DatasetState.ITEM_FEATURES: self.item_features,
            DatasetState.USERS_LIST: self.users_list,
            DatasetState.ITEMS_LIST: self.items_list,
            DatasetState.DATASET: self.dataset
        }
        store_data(state, FilePath.DATASET_PICKLE_PATH)

    def __load_dataset(self):
        state = load_data(FilePath.DATASET_PICKLE_PATH)
        self.interactions = state[DatasetState.INTERACTIONS]
        self.test_interactions = state[DatasetState.TEST_INTERACTIONS]
        self.user_features_matrix = state[DatasetState.USER_FEATURES_MATRIX]
        self.item_features_matrix = state[DatasetState.ITEM_FEATURES_MATRIX]
        self.user_features = state[DatasetState.USER_FEATURES]
        self.item_features = state[DatasetState.ITEM_FEATURES]
        self.users_list = state[DatasetState.USERS_LIST]
        self.items_list = state[DatasetState.ITEMS_LIST]
        self.dataset = state[DatasetState.DATASET]
