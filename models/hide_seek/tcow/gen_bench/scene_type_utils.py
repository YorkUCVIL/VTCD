'''
Kubric scene creation methods for benchmarking isolated physical concepts and/or emergent phenomena.
Created by Basile Van Hoorick, Jul 2022.
'''


import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'gen_bench/'))

from __init__ import *


# Internal imports.
import geometry
import logvisgen


def spawn_background_static(logger, my_kubric, along_x=True):

    my_kubric.insert_static_objects(min_count=4, max_count=4,
                                    any_diameter_range=(1.0, 2.0))

    if along_x:
        # All four objects at -X and +X.
        my_kubric.scene.foreground_assets[-4].position = my_kubric.random_state.uniform(
            (-6.0, -2.0, 1.5), (-4.0, -1.0, 1.5))
        my_kubric.scene.foreground_assets[-3].position = my_kubric.random_state.uniform(
            (-6.0, 1.0, 1.5), (-4.0, 2.0, 1.5))
        my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
            (4.0, -2.0, 1.5), (6.0, -1.0, 1.5))
        my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
            (4.0, 1.0, 1.5), (6.0, 2.0, 1.5))

    else:
        # All four objects at -X, -Y, +X, +Y (one per side).
        my_kubric.scene.foreground_assets[-4].position = my_kubric.random_state.uniform(
            (-6.0, -2.0, 1.5), (-4.0, -2.0, 1.5))
        my_kubric.scene.foreground_assets[-3].position = my_kubric.random_state.uniform(
            (-2.0, -6.0, 1.5), (2.0, -4.0, 1.5))
        my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
            (4.0, -2.0, 1.5), (6.0, -2.0, 1.5))
        my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
            (-2.0, 4.0, 1.5), (2.0, 6.0, 1.5))

    pass


def setup_gravity_bounce(logger, my_kubric):

    # my_kubric.insert_static_objects(min_count=4, max_count=4)
    # my_kubric.scene.foreground_assets[0].position = [-5, -2, 0]
    # my_kubric.scene.foreground_assets[1].position = [-5, 1, 0]
    # my_kubric.scene.foreground_assets[2].position = [5, -1, 0]
    # my_kubric.scene.foreground_assets[3].position = [5, 2, 0]

    # my_kubric.simulate_frames(-60, -1)
    # my_kubric.reset_objects_velocity_friction_restitution()

    # my_kubric.insert_dynamic_objects(min_count=2, max_count=2)
    # my_kubric.scene.foreground_assets[4].position = [0, -2, 5]
    # my_kubric.scene.foreground_assets[4].velocity = [0, 0, -2]
    # my_kubric.scene.foreground_assets[5].position = [0, 2, 5]
    # my_kubric.scene.foreground_assets[5].velocity = [0, 0, -2]

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=2, max_count=2,
                                     any_diameter_range=(1.0, 2.0))
    my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
        (-1.0, -2.5, 4.0), (1.0, -1.5, 6.0))
    my_kubric.scene.foreground_assets[-2].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -3.0), (0.5, 0.5, -1.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-1.0, 1.5, 4.0), (1.0, 2.5, 6.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -3.0), (0.5, 0.5, -1.0))

    pass


def setup_fall_onto_carrier(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=False)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_carriers=1,
                                    container_carrier_diameter_range=(2.0, 3.0))
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 1.5]

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 4.0), (0.5, -0.5, 6.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    pass


def setup_fall_into_container(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=False)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_containers=1,
                                    container_carrier_diameter_range=(2.0, 3.0),
                                    simple_containers_only=True)
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 1.5]

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 4.0), (0.5, -0.5, 6.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    pass


def setup_slide_box_friction(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=2, max_count=2, force_boxes=2,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
        (1.5, -2.0, 1.0), (2.0, -1.5, 1.5))
    my_kubric.scene.foreground_assets[-2].velocity = my_kubric.random_state.uniform(
        (0.0, 4.5, 0.0), (0.0, 5.0, 0.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-2.0, 1.5, 1.0), (-1.5, 2.0, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -5.0, 0.0), (0.0, -4.5, 0.0))

    pass


def setup_slide_box_collide(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()

    my_kubric.insert_dynamic_objects(min_count=2, max_count=2, force_boxes=2,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-2].position = my_kubric.random_state.uniform(
        (0.0, -4.0, 1.0), (0.0, -3.5, 1.5))
    my_kubric.scene.foreground_assets[-2].velocity = my_kubric.random_state.uniform(
        (0.0, 4.5, 0.0), (0.0, 5.0, 0.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (0.0, 3.5, 1.0), (0.0, 4.0, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -5.0, 0.0), (0.0, -4.5, 0.0))

    pass


def setup_box_push_carrier_slide(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_carriers=1,
                                    container_carrier_diameter_range=(2.0, 3.0))
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 1.5]

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()
    
    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 2.0), (0.5, -0.5, 4.0))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1, force_boxes=1,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (0.0, 4.5, 1.0), (0.0, 5.0, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -6.0, 0.0), (0.0, -5.5, 0.0))
    my_kubric.scene.foreground_assets[-1].mass *= 2.0

    pass


def setup_box_push_container_slide(logger, my_kubric):

    spawn_background_static(logger, my_kubric, along_x=True)

    my_kubric.insert_static_objects(min_count=1, max_count=1, force_containers=1,
                                    container_carrier_diameter_range=(2.0, 2.5),
                                    simple_containers_only=True)
    my_kubric.scene.foreground_assets[-1].position = [0.0, 0.0, 2.0]
    
    # DEBUG / TEMP:
    # if 0:
    #     my_kubric.scene.foreground_assets[-1].mass *= 4.0  # Make container heavier for realism.

    my_kubric.simulate_frames(-100, -1)
    my_kubric.reset_objects_velocity_friction_restitution()
    
    my_kubric.insert_dynamic_objects(min_count=1, max_count=1,
                                     any_diameter_range=(0.5, 1.5))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (-0.5, -0.5, 2.5), (0.5, -0.5, 4.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (-0.5, -0.5, -1.0), (0.5, 0.5, 0.0))

    my_kubric.insert_dynamic_objects(min_count=1, max_count=1, force_boxes=1,
                                     box_diameter_range=(1.5, 2.0))
    my_kubric.scene.foreground_assets[-1].position = my_kubric.random_state.uniform(
        (0.0, 5.0, 1.0), (0.0, 5.5, 1.5))
    my_kubric.scene.foreground_assets[-1].velocity = my_kubric.random_state.uniform(
        (0.0, -6.0, 0.0), (0.0, -5.5, 0.0))
    my_kubric.scene.foreground_assets[-1].mass *= 3.0

    # DEBUG / TEMP:
    # if 0:
    #     my_kubric.scene.foreground_assets[-1].mass /= 2.0  # Undo artificial weight.
    #     my_kubric.scene.foreground_assets[-1].mass /= 4.0  # Make box lighter for realism.

    # DEBUG / TEMP:
    # if 0:
    #     print()
    #     print('=> setup_box_push_container_slide')
    #     for instance in my_kubric.scene.foreground_assets:
    #         print(f'asset_id: {instance.asset_id}  '
    #               f'scale_factor: {instance.metadata["scale_factor"]:.2f}  '
    #               f'volume: {instance.metadata["volume"] * 1000.0:.2f}  '
    #               f'mass_pre: {instance.metadata["mass_pre"] * 1000.0:.2f}  '
    #               f'mass_post: {instance.metadata["mass_post"] * 1000.0:.2f}')
    #     print()

    pass


# def setup_fall_collide(logger, my_kubric):

#     my_kubric.insert_static_objects(min_count=4, max_count=4, any_diameter_range=(1.0, 2.0))
#     my_kubric.scene.foreground_assets[0].position = my_kubric.random_state.uniform(
#         (-6.0, -2.0, 1.5), (-4.0, -1.0, 1.5))
#     my_kubric.scene.foreground_assets[1].position = my_kubric.random_state.uniform(
#         (-6.0, 1.0, 1.5), (-4.0, 2.0, 1.5))
#     my_kubric.scene.foreground_assets[2].position = my_kubric.random_state.uniform(
#         (4.0, -2.0, 1.5), (6.0, -1.0, 1.5))
#     my_kubric.scene.foreground_assets[3].position = my_kubric.random_state.uniform(
#         (4.0, 1.0, 1.5), (6.0, 2.0, 1.5))

#     my_kubric.simulate_frames(-100, -1)
#     my_kubric.reset_objects_velocity_friction_restitution()

#     my_kubric.insert_dynamic_objects(min_count=2, max_count=2, diameter_range=(1.0, 2.0))
#     my_kubric.scene.foreground_assets[4].position = my_kubric.random_state.uniform(
#         (0.0, -3.0, 4.5), (0.0, -2.5, 5.5))
#     my_kubric.scene.foreground_assets[4].velocity = my_kubric.random_state.uniform(
#         (0.0, 2.5, -1.5), (0.0, 3.0, -1.0))
#     my_kubric.scene.foreground_assets[5].position = my_kubric.random_state.uniform(
#         (0.0, 2.5, 4.5), (0.0, 3.0, 5.5))
#     my_kubric.scene.foreground_assets[5].velocity = my_kubric.random_state.uniform(
#         (0.0, -3.0, -1.5), (0.0, -2.5, -1.0))

#     pass


def apply_setup_for_type(logger, my_kubric, scene_type):
    if scene_type == 'gravity_bounce':
        setup_gravity_bounce(logger, my_kubric)

    elif scene_type == 'fall_onto_carrier':
        setup_fall_onto_carrier(logger, my_kubric)

    elif scene_type == 'fall_into_container':
        setup_fall_into_container(logger, my_kubric)

    elif scene_type == 'slide_box_friction':
        setup_slide_box_friction(logger, my_kubric)

    elif scene_type == 'slide_box_collide':
        setup_slide_box_collide(logger, my_kubric)

    elif scene_type == 'box_push_carrier_slide':
        setup_box_push_carrier_slide(logger, my_kubric)

    elif scene_type == 'box_push_container_slide':
        setup_box_push_container_slide(logger, my_kubric)

    # elif scene_type == 'fall_collide':
    #     setup_fall_collide(logger, my_kubric)

    else:
        raise ValueError(f'Unknown scene type: {scene_type}')


# def generate_for_type_old(
#         logger, seed_offset, scene_idx, scene_dp, scene_dn, scene_type,
#         num_workers, num_perturbs, num_views, num_queries,
#         frame_width, frame_height, num_frames, frame_rate, render_samples_per_pixel,
#         split_backgrounds, split_objects, mass_est_fp):

#     # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
#     import kubric as kb
#     import kubric_sim
#     import pybullet as pb

#     render_cpu_threads = int(np.ceil(mp.cpu_count() / max(num_workers, 2)))
#     logger.info(f'Using {render_cpu_threads} CPU threads for rendering.')

#     # NOTE: This instance must only be created once per process!
#     my_kubric = kubric_sim.MyKubricSimulatorRenderer(
#         logger, frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
#         frame_rate=frame_rate, render_samples_per_pixel=render_samples_per_pixel,
#         split_backgrounds=split_backgrounds, split_objects=split_objects,
#         render_cpu_threads=render_cpu_threads, mass_est_fp=mass_est_fp)

#     os.makedirs(scene_dp, exist_ok=True)

#     start_time = time.time()

#     t = my_kubric.prepare_next_scene('train', seed_offset + scene_idx)
#     logger.info(f'prepare_next_scene took {t:.2f}s')

#     # The main differentiation between different scene types happens here!
#     apply_setup_for_type(logger, my_kubric, scene_type)

#     all_data_stacks = []
#     all_videos = []

#     start_yaw = my_kubric.random_state.uniform(0.0, 360.0)

#     # Loop over butterfly effect variations.
#     for perturb_idx in range(num_perturbs):

#         logger.info()
#         logger.info(f'perturb_idx: {perturb_idx} / num_perturbs: {num_perturbs}')
#         logger.info()

#         # Ensure that the simulator resets its state for every perturbation.
#         if perturb_idx == 0 and num_perturbs >= 2:
#             logger.info(f'Saving PyBullet simulator state...')
#             # https://github.com/bulletphysics/bullet3/issues/2982
#             pb.setPhysicsEngineParameter(deterministicOverlappingPairs=0)
#             pb_state = pb.saveState()

#         elif perturb_idx >= 1:
#             logger.info(f'Restoring PyBullet simulator state...')
#             pb.restoreState(pb_state)

#         # Always simulate a little bit just before the actual starting point to ensure Kubric
#         # updates its internal state (in particular, object positions) properly.
#         (_, _, t) = my_kubric.simulate_frames(-1, 0)
#         logger.info(f'simulate_frames took {t:.2f}s')

#         if num_perturbs >= 2:
#             t = my_kubric.perturb_object_positions(max_offset_meters=0.005)
#             logger.info(f'perturb_object_positions took {t:.2f}s')

#         (_, _, t) = my_kubric.simulate_frames(0, num_frames)
#         logger.info(f'simulate_frames took {t:.2f}s')

#         # Loop over camera viewpoints.
#         for view_idx in range(num_views):

#             logger.info()
#             logger.info(f'view_idx: {view_idx} / num_views: {num_views}')
#             logger.info()

#             camera_yaw = view_idx * 360.0 / num_views + start_yaw
#             logger.info(f'Calling set_camera_yaw with {camera_yaw}...')
#             t = my_kubric.set_camera_yaw(camera_yaw)
#             logger.info(f'set_camera_yaw took {t:.2f}s')

#             (data_stack, t) = my_kubric.render_frames(0, num_frames - 1)
#             logger.info(f'render_frames took {t:.2f}s')

#             (metadata, t) = my_kubric.get_metadata(exclude_collisions=view_idx > 0)
#             logger.info(f'get_metadata took {t:.2f}s')

#             # NOTE: We select N uniformly random points at a random point in time!
#             # NOTE: This is only an example, because given the exported data, we can easily
#             # generate new tracks on the fly.
#             query_time = np.ones(num_queries, dtype=np.int32) * (num_frames // 5)
#             query_uv = my_kubric.random_state.rand(num_queries, 2)

#             traject_retval = geometry.calculate_3d_point_trajectory_kubric(
#                 data_stack, metadata, query_time, query_uv)
#             uvdxyz = traject_retval['uvdxyz']  # (N, T, 6).

#             # Create videos of source data and annotations.
#             # NOTE: This drawing procedure is somewhat outdated, but is kept for illustration.
#             rgb_clean = data_stack['rgba'][..., :3].copy()
#             rgb_annot = rgb_clean.copy()
#             (T, H, W, _) = rgb_annot.shape

#             for n in range(num_queries):
#                 point_color = matplotlib.colors.hsv_to_rgb([n / num_queries, 1.0, 1.0])
#                 point_color = (point_color * 255.0).astype(np.uint8)

#                 for t in range(T):
#                     cur_u = np.clip(int(uvdxyz[n, t, 0] * W), 0, W - 1)
#                     cur_v = np.clip(int(uvdxyz[n, t, 1] * H), 0, H - 1)
#                     keep_pixel = rgb_annot[t, cur_v, cur_u, :].copy()
#                     rgb_annot[t, cur_v - 1:cur_v + 2, cur_u - 1:cur_u + 2, :] = point_color
#                     rgb_annot[t, cur_v, cur_u, :] = keep_pixel

#             save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_clean'),
#                          rgb_clean, frame_rate)
#             save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_annot'),
#                          rgb_annot, frame_rate)

#             t = my_kubric.write_all_data(os.path.join(
#                 scene_dp, f'frames_p{perturb_idx}_v{view_idx}'))
#             logger.info(f'write_all_data took {t:.2f}s')

#             metadata['query_trajects'] = traject_retval
#             dst_json_fp = os.path.join(
#                 scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}.json')
#             kb.write_json(metadata, dst_json_fp)

#             all_data_stacks.append(data_stack)
#             all_videos.append(rgb_annot)

#             logger.info(f'All together took {time.time() - start_time:.2f}s')

#         pass

#     # Visualize pixel deltas across pairs of perturbations.
#     if num_perturbs >= 2:
#         for perturb_idx in range(num_perturbs):
#             for view_idx in range(num_views):
#                 p1_idx = perturb_idx
#                 p2_idx = (perturb_idx + 1) % num_perturbs
#                 rgba_p1 = all_data_stacks[view_idx + p1_idx * num_views]['rgba']
#                 rgba_p2 = all_data_stacks[view_idx + p2_idx * num_views]['rgba']
#                 rgb_delta = rgba_p1[..., :3].astype(
#                     np.int16) - rgba_p2[..., :3].astype(np.int16)
#                 rgb_delta = np.clip(np.abs(rgb_delta * 2), 0, 255).astype(np.uint8)

#                 save_mp4_gif(os.path.join(
#                     scene_dp, f'{scene_dn}_p{perturb_idx}_v{view_idx}_delta'),
#                     rgb_delta, frame_rate)

#                 # if perturb_idx < num_perturbs - 1:
#                 all_videos.append(rgb_delta)

#     # Finally, bundle all variations and viewpoints together into one big video.
#     big_video = None
#     all_videos = np.stack(all_videos, axis=0)

#     if len(all_videos) == num_perturbs * num_views:
#         big_video = rearrange(all_videos, '(P V) T H W C -> T (V H) (P W) C',
#                               P=num_perturbs, V=num_views)

#     elif len(all_videos) == num_perturbs * 2 * num_views:
#         big_video = rearrange(all_videos, '(D P V) T H W C -> T (V H) (P D W) C',
#                               D=2, P=num_perturbs, V=num_views)
#         # Ignore last ("wrap around") delta video for big bundle.
#         big_video = big_video[:, :, :-W]

#     else:
#         logger.warning()
#         logger.warning(f'Expected {num_perturbs * num_views} or '
#                        f'{num_perturbs * 2 * num_views} '
#                        f'videos, but got {len(all_videos)}?')
#         logger.warning()

#     if big_video is not None:
#         save_mp4_gif(os.path.join(scene_dp, f'{scene_dn}_bundle'), big_video, frame_rate)

#     pass
