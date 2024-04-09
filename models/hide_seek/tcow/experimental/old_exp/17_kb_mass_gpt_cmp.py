'''
BVH, Aug 2022.
NOTE: This is based on kubbench_v2, to visualize difference between auto_mass and gpt_mass.
python experimental/17_kb_mass_gpt_cmp.py
du -sh /tmp/
rm -rf /tmp/*
'''

import os
import sys
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'gen_bench/'))

from __init__ import *


# Internal imports.
import logvisgen
import scene_type_utils


# ==================================
# CUSTOMIZE DATASET PARAMETERS HERE:

# Columbia CV.
scene_types = [
    # 'gravity_bounce',
    # 'fall_onto_carrier',
    # 'fall_into_container',
    # 'slide_box_friction',
    # 'slide_box_collide',
    # 'box_push_carrier_slide',
    'box_push_container_slide',
    # 'fall_collide',
    # 'roll_collide',
    # 'contain_hide',
    # 'contain_roll',
    # 'slide_friction',
]
num_scenes_per_type = 10
num_total_scenes = len(scene_types) * num_scenes_per_type
global_start_idx = 0
global_end_idx = 999

num_workers = 10
num_perturbs = 1
num_views = 1
num_queries = 192

root_dn = 'exp17b'
root_dp = os.path.join('/proj/vondrick3/basile/hs_kubexp_out/', root_dn)
mass_est_fp = '/proj/vondrick3/basile/hide-seek/gpt_mass/v4_mass.txt'
ignore_if_exist = True

seed_offset = 5544
frame_width = 320
frame_height = 240
num_frames = 24
frame_rate = 12
render_samples_per_pixel = 16

split_backgrounds = False
split_objects = False


# NOTE: Because of /tmp size problems, we must stop and restart this script to empty the /tmp
# directory inbetween runs. This counter indicates when all threads should finish.
MAX_SCENE_COUNT = 100


def do_scene(scene_idx, logger, scene_dp, scene_dn, scene_type):

    # Mass provided => use GPT knowledge.
    # Mass file None => use density = 1.
    scenes_per_mode = num_total_scenes // 2
    used_scene_idx = scene_idx % scenes_per_mode
    used_mass_est_fp = (mass_est_fp if scene_idx < scenes_per_mode else None)
    logger.warning(f'used_scene_idx: {used_scene_idx}')
    logger.warning(f'used_mass_est_fp: {used_mass_est_fp}')

    scene_type_utils.generate_for_type(
        logger, seed_offset, used_scene_idx, scene_dp, scene_dn, scene_type,
        num_workers, num_perturbs, num_views, num_queries,
        frame_width, frame_height, num_frames, frame_rate, render_samples_per_pixel,
        split_backgrounds, split_objects, used_mass_est_fp)

    pass


def worker(worker_idx, num_workers, total_scn_cnt):

    logger = logvisgen.Logger(msg_prefix=f'{root_dn}_worker{worker_idx}')

    my_start_idx = worker_idx + global_start_idx
    my_end_idx = min(global_end_idx, num_total_scenes)

    for scene_idx in range(my_start_idx, my_end_idx, num_workers):

        scene_type = scene_types[scene_idx // num_scenes_per_type]
        scene_dn = f'{root_dn}_scn{scene_idx:03d}_{scene_type}'
        scene_dp = os.path.join(root_dp, scene_dn)

        logger.info()
        logger.info(f'scene_idx: {scene_idx} / scene_dn: {scene_dn}')
        logger.info()

        # Determine multiplicity of this scene based on index.
        used_num_perturbs = num_perturbs
        used_num_views = num_views

        # Check for the latest file that could have been written.
        dst_json_fp = os.path.join(
            scene_dp, f'{scene_dn}_p{used_num_perturbs - 1}_v{used_num_views - 1}.json')
        if ignore_if_exist and os.path.exists(dst_json_fp):
            logger.info(f'This scene already exists at {dst_json_fp}, skipping!')
            continue

        else:
            total_scn_cnt.value += 1
            logger.info(f'Total scene counter: {total_scn_cnt.value} / {MAX_SCENE_COUNT}')
            if total_scn_cnt.value >= MAX_SCENE_COUNT:
                logger.warning()
                logger.warning('Reached max allowed scene count, exiting!')
                logger.warning()
                break

            # We perform the actual generation in a separate thread to try to ensure that no memory
            # leaks survive.
            p = mp.Process(target=do_scene, args=(
                scene_idx, logger, scene_dp, scene_dn, scene_type))
            p.start()
            p.join()

        pass

    logger.info()
    logger.info(f'I am done!')
    logger.info()

    pass


def main():

    total_scn_cnt = mp.Value('i', 0)

    os.makedirs(root_dp, exist_ok=True)

    if num_workers <= 0:

        worker(0, 1, total_scn_cnt)

    else:

        processes = [mp.Process(target=worker,
                                args=(worker_idx, num_workers, total_scn_cnt))
                     for worker_idx in range(num_workers)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()

    pass
