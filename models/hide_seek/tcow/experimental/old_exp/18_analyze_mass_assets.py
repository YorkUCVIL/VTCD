'''
BVH, Aug 2022.
python experimental/18_analyze_mass_assets.py
'''

import os
import sys
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))

from __init__ import *

# Library imports.
import pandas as pd

# Internal imports.
import logvisgen


dst_dp = 'experimental/figs'
mass_est_fp = '/proj/vondrick3/basile/hide-seek/gpt_mass/v4_mass.txt'


def main():

    import bpy
    import kubric as kb
    import kubric.simulator
    import kubric.renderer

    logger = logvisgen.Logger(msg_prefix='exp18')
    os.makedirs(dst_dp, exist_ok=True)

    mass_est_list = pd.read_csv(mass_est_fp, header=None, names=['id', 'samples'])
    mass_samples_dict = {id: np.fromstring(samples[1:-1], dtype=np.float32, sep=' ')
                         for (id, samples) in mass_est_list.values}
    # mass_min_max_dict = {id: (samples.mean() * 0.5, samples.mean() * 1.5)
    #                             for (id, samples) in mass_samples_dict.items()}
    mass_mean_dict = {id: samples.mean() for (id, samples) in mass_samples_dict.items()}

    gso_source = kb.AssetSource.from_manifest('gs://kubric-public/assets/GSO/GSO.json')
    ids_list = sorted(gso_source.all_asset_ids())
    ids_list = np.array(ids_list)
    logger.info()
    logger.info(f'ids_list: {len(ids_list)}')
    logger.info(ids_list[:7])
    logger.info(ids_list[-7:])
    logger.info()

    all_data = []

    for asset_id in tqdm.tqdm(ids_list):

        # obj = gso_source.create(asset_id=asset_id)
        # old_mass = obj.mass
        # old_density = obj.mass / max(obj.metadata['volume'], 1e-6)

        volume = gso_source._assets[asset_id]['metadata']['volume']
        old_density = 1.0
        old_mass = old_density * volume

        new_mass = mass_mean_dict[asset_id]
        new_density = new_mass / max(volume, 1e-6)
        mass_factor = new_mass / max(old_mass, 1e-6)

        all_data.append([asset_id,
                         old_mass, old_density,
                         new_mass, new_density,
                         mass_factor])

    lightest_inds = np.argsort([x[3] for x in all_data])
    heaviest_inds = np.flip(lightest_inds)
    least_dense_inds = np.argsort([x[4] for x in all_data])
    most_dense_inds = np.flip(least_dense_inds)
    least_mass_increase_inds = np.argsort([x[5] for x in all_data])
    most_mass_increase_inds = np.flip(least_mass_increase_inds)

    logger.info('=> lightest:')
    for i in range(7):
        logger.info(ids_list[lightest_inds[i]] +
                    f': {all_data[lightest_inds[i]][3] * 1000.0:.1f}')
    logger.info()
    logger.info('=> heaviest:')
    for i in range(7):
        logger.info(ids_list[heaviest_inds[i]] +
                    f': {all_data[heaviest_inds[i]][3] * 1000.0:.1f}')
    logger.info()
    logger.info('=> least dense:')
    for i in range(7):
        logger.info(ids_list[least_dense_inds[i]] +
                    f': {all_data[least_dense_inds[i]][4]:.1f}')
    logger.info()
    logger.info('=> most dense:')
    for i in range(7):
        logger.info(ids_list[most_dense_inds[i]] +
                    f': {all_data[most_dense_inds[i]][4]:.1f}')
    logger.info()
    logger.info('=> least mass increase:')
    for i in range(7):
        logger.info(ids_list[least_mass_increase_inds[i]] +
                    f': {all_data[least_mass_increase_inds[i]][5]:.1f}')
    logger.info()
    logger.info('=> most mass increase:')
    for i in range(7):
        logger.info(ids_list[most_mass_increase_inds[i]] +
                    f': {all_data[most_mass_increase_inds[i]][5]:.1f}')
    logger.info()

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()

    pass
