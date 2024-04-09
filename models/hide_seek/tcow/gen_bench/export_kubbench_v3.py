'''
Special test set generation with my custom Kubric script.
Use experimental/ for debugging, and gen_bench/ for large exports.
Created by Basile Van Hoorick, Sep 2022.

On CV:
conda activate base < TODX
conda activate bcv11
python gen_bench/export_kubbench_v3.py
pkill -9 python
du -sh /tmp/
rm -rf /tmp/*

Aligned with: kubcon_v10.

Changes compared to v2:
- Many (see all changes from kubcon_v8 to kubcon_v10).
- In particular, xray segmentation.

Notes / fixes before v4:
- Changed positions and velocities slightly for box_push_container_slide to give more lead time.
'''

import os
import sys
import torch
sys.path.insert(0, os.path.join(os.getcwd(), 'gen_bench/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Library imports.
import pandas as pd


# Internal imports.
import data_vis
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
num_scenes_per_type = 30
num_total_scenes = len(scene_types) * num_scenes_per_type
global_start_idx = 0
global_end_idx = 999

num_workers = 10
num_perturbs = 1
num_views = 1

root_dn = 'kubbench_v3'
root_dp = '/proj/vondrick3/basile/' + root_dn
mass_est_fp = '/proj/vondrick3/basile/hide-seek/gpt_mass/v4_mass.txt'
ignore_if_exist = True

seed_offset = 3333333
frame_width = int(320 * 1.5)
frame_height = int(240 * 1.5)
num_frames = 36  # Kubric / MOVi = 24.
frame_rate = 12  # Kubric / MOVi = 12.
render_samples_per_pixel = 32  # Kubric / MOVi = 64.


# NOTE: Because of /tmp size problems, we must stop and restart this script to empty the /tmp
# directory inbetween runs. This counter indicates when all threads should finish.
MAX_SCENE_COUNT = 100


def save_mp4_gif(dst_fp, frames, fps):
    imageio.mimwrite(dst_fp + '.mp4', frames, format='ffmpeg', fps=fps, quality=10)
    imageio.mimwrite(dst_fp + '.gif', frames, format='gif', fps=fps)


def do_scene(scene_idx, logger, scene_dp, scene_dn, scene_type):
    # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
    import kubric as kb
    import kubric_sim
    import pybullet as pb

    render_cpu_threads = int(np.ceil(mp.cpu_count() / max(num_workers, 2)))
    logger.info(f'Using {render_cpu_threads} CPU threads for rendering.')

    # NOTE: This instance must only be created once per process!
    my_kubric = kubric_sim.MyKubricSimulatorRenderer(
        logger, frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
        frame_rate=frame_rate, render_samples_per_pixel=render_samples_per_pixel,
        split_backgrounds=False, split_objects=False,
        render_cpu_threads=render_cpu_threads, mass_est_fp=mass_est_fp)

    os.makedirs(scene_dp, exist_ok=True)

    start_time = time.time()

    t = my_kubric.prepare_next_scene('train', seed_offset + scene_idx)
    logger.info(f'prepare_next_scene took {t:.2f}s')

    # The main differentiation between different scene types happens here!
    scene_type_utils.apply_setup_for_type(logger, my_kubric, scene_type)

    all_data_stacks = []
    all_videos = []

    used_num_perturbs = num_perturbs
    used_num_views = num_views

    start_yaw = my_kubric.random_state.uniform(0.0, 360.0)

    # Loop over butterfly effect variations.
    for perturb_idx in range(used_num_perturbs):

        logger.info()
        logger.info(f'perturb_idx: {perturb_idx} / used_num_perturbs: {used_num_perturbs}')
        logger.info()

        # Ensure that the simulator resets its state for every perturbation.
        if perturb_idx == 0 and used_num_perturbs >= 2:
            logger.info(f'Saving PyBullet simulator state...')
            # https://github.com/bulletphysics/bullet3/issues/2982
            pb.setPhysicsEngineParameter(deterministicOverlappingPairs=0)
            pb_state = pb.saveState()

        elif perturb_idx >= 1:
            logger.info(f'Restoring PyBullet simulator state...')
            pb.restoreState(pb_state)

        # Always simulate a little bit just before the actual starting point to ensure Kubric
        # updates its internal state (in particular, object positions) properly.
        (_, _, t) = my_kubric.simulate_frames(-1, 0)
        logger.info(f'simulate_frames took {t:.2f}s')

        if used_num_perturbs >= 2:
            t = my_kubric.perturb_object_positions(max_offset_meters=0.005)
            logger.info(f'perturb_object_positions took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(0, num_frames)
        logger.info(f'simulate_frames took {t:.2f}s')

        # Loop over camera viewpoints.
        for view_idx in range(used_num_views):

            logger.info()
            logger.info(f'view_idx: {view_idx} / used_num_views: {used_num_views}')
            logger.info()

            camera_yaw = view_idx * 360.0 / used_num_views + start_yaw
            logger.info(f'Calling set_camera_yaw with {camera_yaw}...')
            t = my_kubric.set_camera_yaw(camera_yaw)
            logger.info(f'set_camera_yaw took {t:.2f}s')

            (data_stack, t) = my_kubric.render_frames(
                0, num_frames - 1, return_layers=['rgba', 'forward_flow', 'depth', 'normal',
                                                  'object_coordinates', 'segmentation'])
            logger.info(f'render_frames took {t:.2f}s')

            (metadata, t) = my_kubric.get_metadata(exclude_collisions=view_idx > 0)
            logger.info(f'get_metadata took {t:.2f}s')

            # Create videos of source data and annotations.
            save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_rgb'),
                         data_stack['rgba'][..., :3].copy(), frame_rate)
            save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_depth'),
                         data_vis.depth_to_rgb_vis(data_stack['depth']), frame_rate)
            save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_segm'),
                         data_vis.segm_ids_to_rgb(data_stack['segmentation']), frame_rate)

            # Now create videos for all isolated objects separately.
            (div_data, t) = my_kubric.render_frames_divided_objects(
                0, num_frames - 1, return_layers=['rgba', 'depth', 'segmentation'])
            logger.info(f'render_frames_divided_objects took {t:.2f}s')

            # Write all individual frames for normal and divided videos.
            t = my_kubric.write_all_data(os.path.join(
                scene_dp, f'frames_p{perturb_idx}_v{view_idx}'))
            logger.info(f'write_all_data took {t:.2f}s')

            # Write metadata last (this is a marker of completion).
            dst_json_fp = os.path.join(
                scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}.json')
            kb.write_json(metadata, dst_json_fp)

            logger.info(f'All together took {time.time() - start_time:.2f}s')

        pass

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
