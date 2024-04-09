# BVH, Sep 2022.
# https://beta.openai.com/playground
# https://beta.openai.com/docs/quickstart

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


def main():

    resp_fp = 'gpt_mass/v4.p'

    with open(resp_fp, 'rb') as f:
        all_data = pickle.load(f)

    asset_id_data_dict = {x[0]['id'] + ' ' + x[1]: x for x in all_data}

    my_filter = {k: v for (k, v) in asset_id_data_dict.items()
                 if 'depot_canon' in k.lower()}

    my_filter = {k: v for (k, v) in asset_id_data_dict.items()
                 if 'Office_Depot_Canon_PGI5BK_Remanufactured_Ink_Cartridge_Black'.lower() in k.lower()}

    print(all_data)

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    main()
