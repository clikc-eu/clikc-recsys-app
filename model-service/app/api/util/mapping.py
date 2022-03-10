from ..constants import MappingType
from lightfm.data import Dataset as LightDataset


def map_id_external_to_internal(dataset: LightDataset, external_id: int, id_type: MappingType):
    user_id_map, _, item_id_map, _ = dataset.mapping()
    if id_type == MappingType.USER_ID_TYPE:
        return user_id_map[external_id]
    else:
        return item_id_map[external_id]


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


def get_user_feature_mapping(dataset: LightDataset):
    _, user_feature_map, _, _ = dataset.mapping()

    return user_feature_map


def get_external_ids(dataset: LightDataset, id_type: MappingType):
    user_id_map, _, item_id_map, _ = dataset.mapping()

    if id_type == MappingType.USER_ID_TYPE:
        mapping = user_id_map
    else:
        mapping = item_id_map

    return list(mapping.keys())
