'''
Bootstrap dataset generation with my custom Kubric script.
Use experimental/ for debugging.
Created by Basile Van Hoorick, Jun 2022.

python gen_dset/export_kubcon_v2.py

Changes compated to v1:
- Closed the simulation frame range gap to hopefully mitigate jittery motion in first few frames.
- Containers are strictly upright, with uniformly random yaw.
- Weight adjusted proportionally to change in geometric scale (only in one dimension).
- Increased snitch_size_meters from 0.4 to 0.45.

Conclusions for v2 / to change before v3:
- Looks pretty good, now just need container spawn range to be consistent with snitch insertion
    range (action space).
- Frame rate wasn't actually registered; now it is, so everything should happen faster in v3.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'eval/'))

from __init__ import *

# Library imports.
import kubric as kb

# Internal imports.
import kubric_sim
import logvisgen


def main():

    num_scenes = 50
    seed_offset = 62000
    root_dn = 'kubcon_v2'
    root_dp = '/data/' + root_dn
    start_idx = 1
    end_idx = num_scenes
    os.makedirs(root_dp, exist_ok=True)

    # frame_width = 288
    # frame_height = 224
    frame_width = 288 * 2
    frame_height = 224 * 2
    num_frames = 50
    frame_rate = 20
    snitch_size_meters = 0.45

    logger = logvisgen.Logger(msg_prefix='gends_kubcon_v2')

    # NOTE: This instance must only be created once per process!
    my_kubric = kubric_sim.MyKubricSimulatorRenderer(
        logger, frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
        frame_rate=frame_rate)

    for scene_idx in range(start_idx, end_idx):

        scene_dn = f'{root_dn}_scn{scene_idx:05d}'
        scene_dp = os.path.join(root_dp, scene_dn)
        os.makedirs(scene_dp, exist_ok=True)

        logger.info()
        logger.info(scene_idx)
        logger.info(scene_dn)
        logger.info()

        start_time = time.time()

        t = my_kubric.prepare_next_scene('train', seed_offset + scene_idx)
        logger.debug(f'prepare_next_scene took {t:.2f}s')

        t = my_kubric.insert_static_objects(min_count=8, max_count=24,
                                            force_containers=2, force_carriers=1)
        logger.debug(f'insert_static_objects took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(-50, 0)
        logger.debug(f'simulate_frames took {t:.2f}s')

        t = my_kubric.reset_objects_velocity_friction_restitution()
        logger.debug(f'reset_objects_velocity_friction_restitution took {t:.2f}s')

        t = my_kubric.insert_dynamic_objects(min_count=4, max_count=12)
        logger.debug(f'insert_dynamic_objects took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(0, 2)
        logger.debug(f'simulate_frames took {t:.2f}s')

        (data_stack1, t) = my_kubric.render_frames(0, 1)
        logger.debug(f'render_frames took {t:.2f}s')

        snitch_action = my_kubric.random_state.randint(16)
        snitch_x = snitch_action % 4 - 1.5
        snitch_y = (snitch_action // 4) % 4 - 1.5
        logger.info(f'snitch_action: {snitch_action}  snitch_x: {snitch_x}  snitch_y: {snitch_y}')

        t = my_kubric.insert_snitch(at_x=snitch_x, at_y=snitch_y, size_meters=snitch_size_meters)
        logger.debug(f'insert_snitch took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(2, num_frames)
        logger.debug(f'simulate_frames took {t:.2f}s')

        (data_stack2, t) = my_kubric.render_frames(2, num_frames - 1)
        logger.debug(f'render_frames took {t:.2f}s')

        # NOTE: data_stack2 contains both first and second render call results!

        (metadata, t) = my_kubric.get_metadata()
        logger.debug(f'get_metadata took {t:.2f}s')

        rgb_annot = data_stack2['rgba'][..., :3].copy()
        (T, H, W, _) = rgb_annot.shape
        snitch_metadata = [x for x in metadata['instances']
                           if x['asset_id'] == 'Vtech_Roll_Learn_Turtle'
                           and x['axis_diameter'] == snitch_size_meters][0]

        snitch_bboxes = list(zip(snitch_metadata['bbox_frames'], snitch_metadata['bboxes']))
        for (t, bbox) in snitch_bboxes:
            (x1, x2) = int(bbox[1] * W), int(bbox[3] * W)
            (y1, y2) = int(bbox[0] * H), int(bbox[2] * H)
            rgb_annot[t, y1 - 2:y1, x1 - 2:x2 + 2, :] = [255, 255, 0]
            rgb_annot[t, y2:y2 + 2, x1 - 2:x2 + 2, :] = [255, 255, 0]
            rgb_annot[t, y1 - 2:y2 + 2, x1 - 2:x1, :] = [255, 255, 0]
            rgb_annot[t, y1 - 2:y2 + 2, x2:x2 + 2, :] = [255, 255, 0]

        snitch_imgpos = snitch_metadata['image_positions']
        for (t, imgpos) in enumerate(snitch_imgpos):
            (x, y) = int(imgpos[0] * W), int(imgpos[1] * H)
            rgb_annot[t, y - 2:y + 2, x - 2:x + 2, :] = [0, 255, 255]

        imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}.gif'),
                         rgb_annot, format='gif', fps=frame_rate)
        imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}.mp4'),
                         rgb_annot, format='ffmpeg', fps=frame_rate, quality=10)

        t = my_kubric.write_all_data(os.path.join(scene_dp, 'frames'))
        logger.debug(f'write_all_data took {t:.2f}s')

        kb.write_json(metadata, os.path.join(scene_dp, f'{scene_dn}.json'))

        logger.info(f'All together took {time.time() - start_time:.2f}s')

    logger.info()

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)

    main()

    pass
