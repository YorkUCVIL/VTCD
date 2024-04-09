'''
Dataset generation with my custom Kubric script.
Use experimental/ for debugging, and gen_dset/ for large exports.
Created by Basile Van Hoorick, Jul 2022.

conda activate kubric

python gen_dset/export_kubcon_v5.py

pkill -9 python

du -sh /tmp/

rm -rf /tmp/*

Changes compated to v4:
- Removed snitch stuff (including multi-stage simulation & rendering) altogether.
- Added 4D point tracking methods and annotations.
- Added butterfly effect perturbations / variations by simple resimulation (no explicit changes),
revealing that there is some implicit nondeterminism going on.
'''

import os
import sys
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'gen_dset/'))

from __init__ import *


# Internal imports.
import geometry
import logvisgen


# ==================================
# CUSTOMIZE DATASET PARAMETERS HERE:

num_workers = 6
num_scenes = 100
num_perturbs = 3
num_views = 3
num_queries = 128

seed_offset = 75000
root_dn = 'kubcon_v5'
root_dp = '/data/' + root_dn
global_start_idx = 0
global_end_idx = num_scenes
ignore_if_exist = True
scratch_dir = '/tmp/kubcon_scratch/'

frame_width = int(320 * 1.5)
frame_height = int(240 * 1.5)
num_frames = 60
frame_rate = 20


# NOTE: Because of /tmp size problems, we must stop and restart this script to empty the /tmp
# directory inbetween runs. This counter indicates when all threads should finish.
MAX_SCENE_COUNT = 100


def do_scene(scene_idx, logger, scene_dp, scene_dn):

    # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
    import kubric as kb
    import kubric_sim
    import pybullet as pb

    render_cpu_threads = int(np.ceil(mp.cpu_count() / num_workers))
    logger.info(f'Using {render_cpu_threads} CPU threads for rendering.')
    use_scratch_dir = os.path.join(
        scratch_dir, str(scene_idx) + '_' + str(np.random.randint(1000000, 9999999)))

    # NOTE: This instance must only be created once per process!
    my_kubric = kubric_sim.MyKubricSimulatorRenderer(
        logger, frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
        frame_rate=frame_rate, render_cpu_threads=render_cpu_threads, scratch_dir=use_scratch_dir)

    os.makedirs(scene_dp, exist_ok=True)

    start_time = time.time()

    t = my_kubric.prepare_next_scene('train', seed_offset + scene_idx)
    logger.info(f'prepare_next_scene took {t:.2f}s')

    t = my_kubric.insert_static_objects(min_count=8, max_count=24,
                                        force_containers=2, force_carriers=1)
    logger.info(f'insert_static_objects took {t:.2f}s')

    (_, _, t) = my_kubric.simulate_frames(-60, 0)
    logger.info(f'simulate_frames took {t:.2f}s')

    t = my_kubric.reset_objects_velocity_friction_restitution()
    logger.info(f'reset_objects_velocity_friction_restitution took {t:.2f}s')

    t = my_kubric.insert_dynamic_objects(min_count=4, max_count=12)
    logger.info(f'insert_dynamic_objects took {t:.2f}s')

    all_data_stacks = []

    # Loop over butterfly effect variations.
    for perturb_idx in range(num_perturbs):

        logger.info()
        logger.info(f'perturb_idx: {perturb_idx} / num_perturbs: {num_perturbs}')
        logger.info()

        # Ensure that the simulator resets its state for every perturbation.
        if perturb_idx == 0 and num_perturbs >= 2:
            logger.info(f'Saving PyBullet simulator state...')
            # https://github.com/bulletphysics/bullet3/issues/2982
            pb.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
            pb_state = pb.saveState()

        elif perturb_idx >= 1:
            logger.info(f'Restoring PyBullet simulator state...')
            pb.restoreState(pb_state)
            # TODX actually perturb objects?

        (_, _, t) = my_kubric.simulate_frames(0, num_frames)
        logger.info(f'simulate_frames took {t:.2f}s')

        for view_idx in range(num_views):

            logger.info()
            logger.info(f'view_idx: {view_idx} / num_views: {num_views}')
            logger.info()

            t = my_kubric.set_camera_yaw(view_idx * 360.0 / num_views)
            logger.info(f'set_camera_yaw took {t:.2f}s')

            (data_stack, t) = my_kubric.render_frames(0, num_frames - 1)
            logger.info(f'render_frames took {t:.2f}s')

            (metadata, t) = my_kubric.get_metadata(exclude_collisions=view_idx > 0)
            logger.info(f'get_metadata took {t:.2f}s')

            # NOTE: We select N uniformly random points at a random point in time!
            # NOTE: This is only an example, because given the exported data, we can easily generate new
            # tracks on the fly.
            query_time = np.ones(num_queries, dtype=np.int32) * (num_frames // 5)
            query_uv = np.random.rand(num_queries, 2)

            traject_retval = geometry.calculate_3d_point_trajectory_kubric(
                data_stack, metadata, query_time, query_uv)

            uvdxyz = traject_retval['uvdxyz']  # (N, T, 6).

            rgb_clean = data_stack['rgba'][..., :3].copy()
            rgb_annot = rgb_clean.copy()
            (T, H, W, _) = rgb_annot.shape

            for n in range(num_queries):
                point_color = matplotlib.colors.hsv_to_rgb([n / num_queries, 1.0, 1.0])
                point_color = (point_color * 255.0).astype(np.uint8)

                for t in range(T):
                    cur_u = np.clip(int(uvdxyz[n, t, 0] * W), 0, W - 1)
                    cur_v = np.clip(int(uvdxyz[n, t, 1] * H), 0, H - 1)
                    keep_pixel = rgb_annot[t, cur_v, cur_u, :].copy()
                    rgb_annot[t, cur_v - 1:cur_v + 2, cur_u - 1:cur_u + 2, :] = point_color
                    rgb_annot[t, cur_v, cur_u, :] = keep_pixel

            imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_clean.mp4'),
                                rgb_clean, format='ffmpeg', fps=frame_rate, quality=10)
            imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_clean.gif'),
                                rgb_clean, format='gif', fps=frame_rate)
            imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_annot.mp4'),
                                rgb_annot, format='ffmpeg', fps=frame_rate, quality=10)
            imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_annot.gif'),
                                rgb_annot, format='gif', fps=frame_rate)

            t = my_kubric.write_all_data(os.path.join(
                scene_dp, f'frames_p{perturb_idx}_v{view_idx}'))
            logger.info(f'write_all_data took {t:.2f}s')

            metadata['query_trajects'] = traject_retval

            kb.write_json(metadata, os.path.join(
                scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}.json'))

            all_data_stacks.append(data_stack)

            logger.info(f'All together took {time.time() - start_time:.2f}s')

        pass

    # Visualize pixel deltas across pairs of perturbations.
    if num_perturbs >= 2:
        for perturb_idx in range(num_perturbs - 1):
            for view_idx in range(num_views):
                rgba_p1 = all_data_stacks[view_idx + perturb_idx * num_views]['rgba']
                rgba_p2 = all_data_stacks[view_idx + (perturb_idx + 1) * num_views]['rgba']
                rgb_delta = rgba_p1[..., :3].astype(
                    np.int16) - rgba_p2[..., :3].astype(np.int16)
                rgb_delta = np.abs(rgb_delta).astype(np.uint8)

                imageio.mimwrite(os.path.join(
                    scene_dp, f'{scene_dn}_p{perturb_idx + 1}_v{view_idx}_delta.mp4'),
                    rgb_delta, format='ffmpeg', fps=frame_rate, quality=10)
                imageio.mimwrite(os.path.join(
                    scene_dp, f'{scene_dn}_p{perturb_idx + 1}_v{view_idx}_delta.gif'),
                    rgb_delta, format='gif', fps=frame_rate)

    pass


def worker(worker_idx, num_workers, total_scn_cnt):

    logger = logvisgen.Logger(msg_prefix=f'{root_dn}_worker{worker_idx}')

    my_start_idx = worker_idx + global_start_idx

    for scene_idx in range(my_start_idx, global_end_idx, num_workers):

        scene_dn = f'{root_dn}_scn{scene_idx:05d}'
        scene_dp = os.path.join(root_dp, scene_dn)

        logger.info()
        logger.info(f'scene_idx: {scene_idx} / scene_dn: {scene_dn}')
        logger.info()

        # Check for the latest file that could have been written.
        dst_json_fp = os.path.join(scene_dp, f'{scene_dn}_p{num_perturbs - 1}_v{num_views - 1}.json')
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
            p = mp.Process(target=do_scene, args=(scene_idx, logger, scene_dp, scene_dn))
            p.start()
            p.join()

        pass

    logger.info()
    logger.info(f'I am done!')
    logger.info()

    pass


def main():

    os.makedirs(root_dp, exist_ok=True)

    if num_workers <= 0:

        worker(0, 1)

    else:

        total_scn_cnt = mp.Value('i', 0)

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
