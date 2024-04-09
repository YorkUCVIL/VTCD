# BVH, Aug 2022.

# https://beta.openai.com/playground

# https://beta.openai.com/docs/quickstart

# python experimental/15_gpt_mass_est.py

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))

from __init__ import *

# Library imports.
import os
import openai
import pandas as pd
import time
import tempfile


# def asset_friendly(gso_source, ids_list):

#     base_prompt = (
#         'Weisshai_Great_White_Shark is a toy shark.\n'
#         'Utana_5_Porcelain_Ramekin_Large is a ramekin.\n'
#         'Top_Paw_Dog_Bow_Bone_Ceramic_13_fl_oz_total is a dog bowl.\n'
#         'Threshold_Porcelain_Spoon_Rest_White is a spoon.\n'
#         'Threshold_Dinner_Plate_Square_Rim_White_Porcelain is a plate.\n'
#         'Threshold_Hand_Towel_Blue_Medallion_16_x_27 is a hand towel.\n'
#         'Threshold_Basket_Natural_Finish_Fabric_Liner_Small is a basket.\n'
#         'Tag_Dishtowel_Green is a dish towel.\n'
#         'TWIST_SHAPE is a toy.\n'
#         'TABLEWARE_SET is a set of plates.\n'
#         'Sootheze_Cold_Therapy_Elephant is a toy elephant.\n'
#         'Shaxon_100_Molded_Category_6_RJ45RJ45_Shielded_Patch_Cord_White is a long cable.\n'
#         'Schleich_Lion_Action_Figure is a set of toy lion.\n'
#         'Schleich_African_Black_Rhino is a toy rhino.\n'
#         'STACKING_BEAR_V04KKgGBn2A is a toy.\n'
#         'SHAPE_SORTER is a toy.\n'
#         'SANDWICH_MEAL is a plate with food.\n'
#     )

#     all_objects = []
#     all_responses = []

#     for asset_id in ids_list:

#         # cur_obj = gso_source.create(asset_id=asset_id)
#         # all_objects.append(cur_obj)

#         prompt = f'{base_prompt}\n{asset_id}'
#         response = openai.Completion.create(
#             model='text-davinci-002',
#             prompt=prompt,
#             temperature=0.3,
#             max_tokens=64,
#             frequency_penalty=0,
#             presence_penalty=0,
#         )

#         all_responses.append(response)
#         response_text = response.choices[0].text

#         print(asset_id, response_text)

#     pass


# def asset_weight(gso_source, ids_list):

#     pass


# def friendly_weight(gso_source, ids_list):

#     prompt = (
#         'A car weighs between 700kg and 2000kg.\n'
#         'A person weighs between 40kg and 120kg.\n'
#     )

#     response = openai.Completion.create(
#         model='text-davinci-002',
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )

#     pass


def description_weight(gso_source, ids_list):

    all_data = []

    for asset_id in tqdm.tqdm(ids_list):

        cur_asset = gso_source._assets[asset_id]
        bounds = cur_asset["kwargs"]["bounds"]

        # for question in ['What is the approximate mass of this object?',
        #                  'What is the approximate weight of this object?']:

        # prompt = ('Category: ' + cur_asset['metadata']['category'] + '\n' +
        #           'Description: ' + cur_asset['metadata']['description'] + '\n' +
        #           'Question: ' + question + '\n' +
        #           'Answer:')

        #   f'Volume: {cur_asset["metadata"]["volume"] * 1e6:.1f} cm^3.\n' +

        for question in ['Mass:', 'Weight:']:

            try:

                prompt = (f'Category: {cur_asset["metadata"]["category"]}\n' +
                        f'Description: {cur_asset["metadata"]["description"]}\n' +
                        f'Height: {(bounds[1][2] - bounds[0][2]) * 1e2:.2f} cm.\n' +
                        question)

                response = openai.Completion.create(
                    model='text-davinci-002',
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=32,
                    n=2,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

                response_texts = [c.text for c in response.choices]
                all_data.append((cur_asset, question, prompt, response, response_texts))

                print()
                print('=>', asset_id)
                print(prompt, response_texts[0], response_texts[1])
                print()

            except Exception as e:

                print()
                print('=> EXCEPTION')
                print('=>', asset_id)
                print(e)
                print()

            time.sleep(0.9)  # Max 60 requests per minute in first 48 hours.

    return all_data


def generate_responses(resp_fp):

    os.makedirs(str(pathlib.Path(resp_fp).parent), exist_ok=True)

    import bpy
    import kubric as kb
    import kubric.simulator
    import kubric.renderer

    # openai.organization = 'org-5VAHV0A3YIAk4RtiD0EMgvcB'  # Personal
    openai.organization = 'org-TMBYkK7KHKqIHwiA3rmuw85H'  # Columbia
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # openai.Model.list()

    gso_source = kb.AssetSource.from_manifest('gs://kubric-public/assets/GSO/GSO.json')
    ids_list = sorted(gso_source.all_asset_ids())
    print()
    print('ids_list:', len(ids_list))
    print()

    # TEMP to decrease usage:
    # ids_list = random.sample(ids_list, 100)

    ids_list = sorted(ids_list)
    print('ids_list:', ids_list)
    print()

    # asset_friendly(gso_source, ids_list)
    # asset_weight(gso_source, ids_list)
    # friendly_weight(gso_source, ids_list)
    all_data = description_weight(gso_source, ids_list)

    with open(resp_fp, 'wb') as f:
        pickle.dump(all_data, f)

    return all_data


def generate_mass_dict(resp_fp, mass_fp):

    os.makedirs(str(pathlib.Path(mass_fp).parent), exist_ok=True)

    with open(resp_fp, 'rb') as f:
        all_data = pickle.load(f)

    asset_data = defaultdict(list)
    for data in all_data:
        asset_id = data[0]['id']
        asset_data[asset_id].append(data)

    to_save = []

    for asset_id in asset_data.keys():
        response_texts = []
        for data in asset_data[asset_id]:
            response_texts += data[4]
        print(asset_id, response_texts)
        
        values_kg = []
        for response_text in response_texts:
            if ' kg.' in response_text:
                cur_value = response_text.split(' kg.')[0].split(' ')[-1]
                multiplier = 1.0
            elif ' g.' in response_text:
                cur_value = response_text.split(' g.')[0].split(' ')[-1]
                multiplier = 0.001
            elif ' mg.' in response_text:
                cur_value = response_text.split(' mg.')[0].split(' ')[-1]
                multiplier = 0.000001
            elif ' pounds.' in response_text:
                cur_value = response_text.split(' pounds.')[0].split(' ')[-1]
                multiplier = 0.453592
            elif ' lbs.' in response_text:
                cur_value = response_text.split(' lbs.')[0].split(' ')[-1]
                multiplier = 0.453592
            elif ' ounces.' in response_text:
                cur_value = response_text.split(' ounces.')[0].split(' ')[-1]
                multiplier = 0.0283495
            elif ' oz.' in response_text:
                cur_value = response_text.split(' oz.')[0].split(' ')[-1]
                multiplier = 0.0283495
            
            cur_value = cur_value.replace(',', '.')
            values_kg.append(float(cur_value) * multiplier)

        values_kg = np.array(values_kg)
        print(values_kg)
        print()
        
        to_save.append((asset_id, values_kg))

    pd.DataFrame(to_save).to_csv(mass_fp, index=False, header=False)

    return to_save

def main():

    resp_fp = 'gpt_mass/v4.p'

    # if not os.path.exists(resp_fp):
    #     generate_responses(resp_fp)

    mass_fp = 'experimental/tmp_ignore.txt'

    if not os.path.exists(mass_fp):
        generate_mass_dict(resp_fp, mass_fp)

    print()
    print('Done!')
    print()

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    main()
