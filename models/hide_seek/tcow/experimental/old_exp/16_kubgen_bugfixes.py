'''
BVH, Aug 2022.
NOTE: This is a testbed for export_kubcon_v9.
python experimental/16_kubgen_bugfixes.py
du -sh /tmp/
rm -rf /tmp/*
'''

import os
import sys
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'gen_dset/'))

from __init__ import *

# Internal imports.
import geometry
import logvisgen


# ==================================
# CUSTOMIZE DATASET PARAMETERS HERE:

# Columbia CV.
num_scenes = 20
global_start_idx = 0
global_end_idx = num_scenes  # Val + train + test set.

num_workers = 10
num_perturbs = 1
num_views = 1
num_queries = 128
perturbs_first_scenes = 0  # Only test.
views_first_scenes = 0  # Only test.
test_first_scenes = 0  # For handling background & object asset splits (optional).

root_dn = 'exp16i'
root_dp = os.path.join('/proj/vondrick3/basile/hs_kubexp_out/', root_dn)
mass_est_fp = '/proj/vondrick3/basile/hide-seek/gpt_mass/v4_mass.txt'
ignore_if_exist = True

seed_offset = 5544
frame_width = 320
frame_height = 240
num_frames = 24
frame_rate = 12
render_samples_per_pixel = 16

min_static = 4
max_static = 24
min_dynamic = 2
max_dynamic = 12
split_backgrounds = False
split_objects = False


# NOTE: Because of /tmp size problems, we must stop and restart this script to empty the /tmp
# directory inbetween runs. This counter indicates when all threads should finish.
MAX_SCENE_COUNT = 100


def save_mp4_gif(dst_fp, frames, fps):
    imageio.mimwrite(dst_fp + '.mp4', frames, format='ffmpeg', fps=fps, quality=10)
    imageio.mimwrite(dst_fp + '.gif', frames, format='gif', fps=fps)


def do_scene(scene_idx, logger, scene_dp, scene_dn):
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
        split_backgrounds=split_backgrounds, split_objects=split_objects,
        render_cpu_threads=render_cpu_threads, mass_est_fp=mass_est_fp)

    os.makedirs(scene_dp, exist_ok=True)

    start_time = time.time()

    phase = 'test' if scene_idx < test_first_scenes else 'train'
    t = my_kubric.prepare_next_scene(phase, seed_offset + scene_idx)
    logger.info(f'prepare_next_scene took {t:.2f}s')

    t = my_kubric.insert_static_objects(min_count=min_static, max_count=max_static,
                                        force_containers=2, force_carriers=1)
    logger.info(f'insert_static_objects took {t:.2f}s')

    (_, _, t) = my_kubric.simulate_frames(-60, -1)
    logger.info(f'simulate_frames took {t:.2f}s')

    t = my_kubric.reset_objects_velocity_friction_restitution()
    logger.info(f'reset_objects_velocity_friction_restitution took {t:.2f}s')

    t = my_kubric.insert_dynamic_objects(min_count=min_dynamic, max_count=max_dynamic)
    logger.info(f'insert_dynamic_objects took {t:.2f}s')

    all_data_stacks = []
    all_videos = []

    # Determine multiplicity of this scene based on index.
    used_num_perturbs = num_perturbs if scene_idx < perturbs_first_scenes else 1
    used_num_views = num_views if scene_idx < views_first_scenes else 1

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

            (data_stack, t) = my_kubric.render_frames(0, num_frames - 1)
            logger.info(f'render_frames took {t:.2f}s')

            (metadata, t) = my_kubric.get_metadata(exclude_collisions=view_idx > 0)
            logger.info(f'get_metadata took {t:.2f}s')

            # NOTE: We select N uniformly random points at a random point in time!
            # NOTE: This is only an example, because given the exported data, we can easily
            # generate new tracks on the fly.
            query_time = np.ones(num_queries, dtype=np.int32) * (num_frames // 5)
            query_uv = my_kubric.random_state.rand(num_queries, 2) * 0.8 + 0.1

            traject_retval = geometry.calculate_3d_point_trajectory_kubric(
                data_stack, metadata, query_time, query_uv)
            uvdxyz = traject_retval['uvdxyz']  # (N, T, 6).

            # Create videos of source data and annotations.
            # NOTE: This drawing procedure is somewhat outdated, but is kept for illustration.
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

            save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_clean'),
                         rgb_clean, frame_rate)
            save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_annot'),
                         rgb_annot, frame_rate)

            t = my_kubric.write_all_data(os.path.join(
                scene_dp, f'frames_p{perturb_idx}_v{view_idx}'))
            logger.info(f'write_all_data took {t:.2f}s')

            metadata['query_trajects'] = traject_retval
            dst_json_fp = os.path.join(
                scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}.json')
            kb.write_json(metadata, dst_json_fp)

            all_data_stacks.append(data_stack)
            all_videos.append(rgb_annot)

            logger.info(f'All together took {time.time() - start_time:.2f}s')

        pass

    # Visualize pixel deltas across pairs of perturbations.
    if used_num_perturbs >= 2:
        for perturb_idx in range(used_num_perturbs):
            for view_idx in range(used_num_views):
                p1_idx = perturb_idx
                p2_idx = (perturb_idx + 1) % used_num_perturbs
                rgba_p1 = all_data_stacks[view_idx + p1_idx * used_num_views]['rgba']
                rgba_p2 = all_data_stacks[view_idx + p2_idx * used_num_views]['rgba']
                rgb_delta = rgba_p1[..., :3].astype(
                    np.int16) - rgba_p2[..., :3].astype(np.int16)
                rgb_delta = np.clip(np.abs(rgb_delta * 2), 0, 255).astype(np.uint8)

                save_mp4_gif(os.path.join(
                    scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_delta'),
                    rgb_delta, frame_rate)

                all_videos.append(rgb_delta)

    # Finally, bundle all variations and viewpoints together into one big video.
    big_video = None
    all_videos = np.stack(all_videos, axis=0)
    if len(all_videos) == used_num_perturbs * used_num_views:
        big_video = rearrange(all_videos, '(P V) T H W C -> T (V H) (P W) C',
                              P=used_num_perturbs, V=used_num_views)
    elif len(all_videos) == 2 * used_num_perturbs * used_num_views:
        big_video = rearrange(all_videos, '(D P V) T H W C -> T (V H) (P D W) C',
                              D=2, P=used_num_perturbs, V=used_num_views)
        # Ignore last ("wrap around") delta video for big bundle.
        big_video = big_video[:, :, :-W]
    else:
        logger.warning()
        logger.warning(f'Expected {used_num_perturbs * used_num_views} or '
                       f'{used_num_perturbs * used_num_views * 2} '
                       f'videos, but got {len(all_videos)}?')
        logger.warning()

    if big_video is not None:
        save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_bundle'), big_video, frame_rate)

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

        # Determine multiplicity of this scene based on index.
        used_num_perturbs = num_perturbs if scene_idx < perturbs_first_scenes else 1
        used_num_views = num_views if scene_idx < views_first_scenes else 1

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
            p = mp.Process(target=do_scene, args=(scene_idx, logger, scene_dp, scene_dn))
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
