import json
import os
import random
from ..schemas import LearningUnit, DynamicField, Translation
from typing import List
from ..constants import FilePath

class LearningUnitGenerationException(Exception):
    pass

'''
This is an helper function that builds a learning units pickle
file starting from a json file.
'''
def generate_learning_units():

    learning_units: List[LearningUnit] = list()

    # Open json file
    if os.path.isfile(os.getcwd() + '/' + FilePath.BASE_LU_JSON_PATH):
        with open(os.getcwd() + '/' + FilePath.BASE_LU_JSON_PATH, 'r') as f:
            # Read json file as python dictionary
            item_data = json.load(f)

    # Genero elementi
    skill_1_cluster_1 = 30
    s1_c1 = list()
    skill_1_cluster_2 = 30
    s1_c2 = list()
    skill_1_cluster_3 = 30
    s1_c3 = list()
    skill_2_cluster_1 = 30
    s2_c1 = list()
    skill_2_cluster_2 = 30
    s2_c2 = list()
    skill_2_cluster_3 = 30
    s2_c3 = list()
    skill_3_cluster_1 = 30
    s3_c1 = list()
    skill_3_cluster_2 = 30
    s3_c2 = list()
    skill_3_cluster_3 = 30
    s3_c3 = list()
    skill_4_cluster_1 = 30
    s4_c1 = list()
    skill_4_cluster_2 = 30
    s4_c2 = list()
    skill_4_cluster_3 = 30
    s4_c3 = list()

    # Extract random 12 themes (required number of themes in order to generate data)
    themes = list(map(lambda i: i['theme'], item_data))
    themes = list(set(themes))
    if len(themes) < 12:
        raise LearningUnitGenerationException(f'Not enough themes. Required number is 12, found {len(themes)}')

    random.shuffle(themes)
    themes = themes[0:12]

    # For each theme we must have at least 30 elements
    for theme in themes:
        elems = list(filter(lambda i: i['theme'] == theme, item_data))
        if len(elems) < 30:
            raise LearningUnitGenerationException(f'Not enough elements for theme {theme}. Required number is 30, found {len(elems)}')

    for item in item_data:
        if item["theme"] == themes[0] and skill_1_cluster_1 > 0:
            s1_c1.append(item)
            skill_1_cluster_1 = skill_1_cluster_1 - 1

        if item["theme"] == themes[1] and skill_1_cluster_2 > 0:
            s1_c2.append(item)
            skill_1_cluster_2 = skill_1_cluster_2 - 1

        if item["theme"] == themes[2] and skill_1_cluster_3 > 0:
            s1_c3.append(item)
            skill_1_cluster_3 = skill_1_cluster_3 - 1

        if item["theme"] == themes[3] and skill_2_cluster_1 > 0:
            s2_c1.append(item)
            skill_2_cluster_1 = skill_2_cluster_1 - 1

        if item["theme"] == themes[4] and skill_2_cluster_2 > 0:
            s2_c2.append(item)
            skill_2_cluster_2 = skill_2_cluster_2 - 1

        if item["theme"] == themes[5] and skill_2_cluster_3 > 0:
            s2_c3.append(item)
            skill_2_cluster_3 = skill_2_cluster_3 - 1

        if item["theme"] == themes[6] and skill_3_cluster_1 > 0:
            s3_c1.append(item)
            skill_3_cluster_1 = skill_3_cluster_1 - 1

        if item["theme"] == themes[7] and skill_3_cluster_2 > 0:
            s3_c2.append(item)
            skill_3_cluster_2 = skill_3_cluster_2 - 1

        if item["theme"] == themes[8] and skill_3_cluster_3 > 0:
            s3_c3.append(item)
            skill_3_cluster_3 = skill_3_cluster_3 - 1

        if item["theme"] == themes[9] and skill_4_cluster_1 > 0:
            s4_c1.append(item)
            skill_4_cluster_1 = skill_4_cluster_1 - 1

        if item["theme"] == themes[10] and skill_4_cluster_2 > 0:
            s4_c2.append(item)
            skill_4_cluster_2 = skill_4_cluster_2 - 1

        if item["theme"] == themes[11] and skill_4_cluster_3 > 0:
            s4_c3.append(item)
            skill_4_cluster_3 = skill_4_cluster_3 - 1

    i = 0
    eqf = 2
    for item in s1_c1:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="1", skill="1", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s1_c2:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="2", skill="1", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s1_c3:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="3", skill="1", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s2_c1:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="1", skill="2", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s2_c2:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="2", skill="2", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s2_c3:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="3", skill="2", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s3_c1:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="1", skill="3", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s3_c2:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="2", skill="3", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s3_c3:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="3", skill="3", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s4_c1:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="1", skill="4", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s4_c2:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="2", skill="4", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1

    i = 0
    eqf = 2
    for item in s4_c3:
        if i == 10:
            i = 0
            eqf = eqf + 1

        df = DynamicField(type="paragraph", content=item.get("fulltext"))
        translation = Translation(language_name="en", title=item.get("title"), subtitle = item.get("subtitle"), keywords=item.get("keywords"), introduction=item.get("introduction"), text_area="", dynamic_fields=list())
        translation.dynamic_fields.append(df)
        lu = LearningUnit(identifier=item.get("item_id"), cluster_number="3", skill="4", eqf_level=str(eqf), translations=list())
        lu.translations.append(translation)
        learning_units.append(lu)
        i += 1


    return learning_units
