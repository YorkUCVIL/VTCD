'''
Bootstrap dataset generation with my custom Kubric script.
Use experimental/ for debugging.
Created by Basile Van Hoorick, Jun 2022.

python gen_dset/export_kubcon_v1.py

Conclusion / to fix before v2:
- Motion jittery in first few frames, probably due to gap in simulation frame range (1 to 2) or
    progressive/alternating calls.
- Containers have random rotation, should be mostly upright instead.
- Some tiny objects seem too heavy.
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

    num_scenes = 400
    seed_offset = 61000
    root_dp = '/data/kubcon_v1'
    os.makedirs(root_dp, exist_ok=True)

    # frame_width = 288
    # frame_height = 224
    frame_width = 288 * 2
    frame_height = 224 * 2
    num_frames = 36
    frame_rate = 12
    snitch_size_meters = 0.4

    logger = logvisgen.Logger(msg_prefix='gends_kubcon')

    # NOTE: This instance must only be created once per process!
    my_kubric = kubric_sim.MyKubricSimulatorRenderer(
        logger, frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
        frame_rate=frame_rate)

    for scene_idx in range(2, num_scenes):

        scene_dn = f'kubcon_v1_scn{scene_idx:05d}'
        scene_dp = os.path.join(root_dp, scene_dn)
        os.makedirs(scene_dp, exist_ok=True)

        logger.info()
        logger.info(scene_idx)
        logger.info(scene_dn)
        logger.info()

        start_time = time.time()

        t = my_kubric.prepare_next_scene('train', seed_offset + scene_idx)
        logger.info(f'prepare_next_scene took {t:.2f}s')

        t = my_kubric.insert_static_objects(min_count=8, max_count=24,
                                            force_containers=2, force_carriers=1)
        logger.info(f'insert_static_objects took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(-60, -1)
        logger.info(f'simulate_frames took {t:.2f}s')

        t = my_kubric.reset_objects_velocity_friction_restitution()
        logger.info(f'reset_objects_velocity_friction_restitution took {t:.2f}s')

        t = my_kubric.insert_dynamic_objects(min_count=4, max_count=12)
        logger.info(f'insert_dynamic_objects took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(0, 1)
        logger.info(f'simulate_frames took {t:.2f}s')

        (data_stack1, t) = my_kubric.render_frames(0, 1)
        logger.info(f'render_frames took {t:.2f}s')

        t = my_kubric.insert_snitch(size_meters=snitch_size_meters)
        logger.info(f'insert_snitch took {t:.2f}s')

        (_, _, t) = my_kubric.simulate_frames(2, num_frames - 1)
        logger.info(f'simulate_frames took {t:.2f}s')

        (data_stack2, t) = my_kubric.render_frames(2, num_frames - 1)
        logger.info(f'render_frames took {t:.2f}s')

        # NOTE: data_stack2 contains both first and second render call results!

        (metadata, t) = my_kubric.get_metadata()
        logger.info(f'get_metadata took {t:.2f}s')

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

        t = my_kubric.write_all_data(os.path.join(scene_dp, scene_dn))
        logger.info(f'write_all_data took {t:.2f}s')

        kb.write_json(metadata, os.path.join(scene_dp, f'{scene_dn}.json'))

        logger.info(f'All together took {time.time() - start_time:.2f}s')

    logger.info()

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)

    main()

    pass
