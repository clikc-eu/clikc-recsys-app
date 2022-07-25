from fastapi import HTTPException, status
from ..constants import MappingType
from lightfm.data import Dataset as LightDataset

'''
This function maps an external id to an internal LightFM id
'''
def map_id_external_to_internal(dataset: LightDataset, external_id: int, id_type: MappingType):
    user_id_map, _, item_id_map, _ = dataset.mapping()
    if id_type == MappingType.USER_ID_TYPE:
        return user_id_map[external_id]
    else:
        return item_id_map[external_id]

'''
This function maps an internal LightFM id to an external id
'''
def map_id_internal_to_external(dataset: LightDataset, internal_id: int, id_type: MappingType):
    user_id_map, _, item_id_map, _ = dataset.mapping()

    if id_type == MappingType.USER_ID_TYPE:
        mapping = user_id_map
    else:
        mapping = item_id_map

    key_list = list(mapping.keys())
    val_list = list(mapping.values())
    position = val_list.index(internal_id)
    return key_list[position]

'''
This function return the mappings between external and internal ids for features
'''
def get_user_feature_mapping(dataset: LightDataset):
    _, user_feature_map, _, _ = dataset.mapping()

    return user_feature_map

'''
This function returns external ids from mappings, for users or items
'''
def get_external_ids(dataset: LightDataset, id_type: MappingType):
    user_id_map, _, item_id_map, _ = dataset.mapping()

    if id_type == MappingType.USER_ID_TYPE:
        mapping = user_id_map
    else:
        mapping = item_id_map

    return list(mapping.keys())

'''
This function maps cluster numbers from recommender format to DB (to_db=True) format
and viceversa (to_db=False).
'''
def cluster_mapper(id: int, cluster_number: str, skill: str, to_db: bool = False):

    db_to_rec = {
        "1" : "1",
        "2" : "2",
        "3" : "3",
        "4" : "1",
        "5" : "2",
        "6" : "3",
        "7" : "1",
        "8" : "2",
        "9" : "3",
        "10": "1",
        "11": "2",
        "12": "3"
    }

    if to_db == True:
        if skill == '1':
            return cluster_number
        elif skill == '2':
            m = {
                '1' : '4',
                '2' : '5',
                '3' : '6'
            }
            return m[cluster_number]
        elif skill == '3':
            m = {
                '1' : '7',
                '2' : '8',
                '3' : '9'
            }
            return m[cluster_number]
        elif skill == '4':
            m = {
                '1' : '10',
                '2' : '11',
                '3' : '12'
            }
            return m[cluster_number]

    try:
        val = db_to_rec[cluster_number]
    except:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f'Entity With ID={id} Not Valid.')

    return val