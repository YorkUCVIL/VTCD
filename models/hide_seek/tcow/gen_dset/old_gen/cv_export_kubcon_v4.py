'''
Bootstrap dataset generation with my custom Kubric script.
Use experimental/ for debugging.
Created by Basile Van Hoorick, Jun 2022.

NEW:
bash gen_dset/loop_cv_kubcon_v4.sh

OLD:
conda activate bcv11
python gen_dset/cv_export_kubcon_v4.py
rm -rf /tmp/*
pkill -9 python

ALL NOTES:
See export_kubcon_v4.py!
'''

import os
import sys
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'gen_dset/'))

from __init__ import *

# Internal imports.
import logvisgen


# ==================================
# CUSTOMIZE DATASET PARAMETERS HERE:

num_workers = 16
num_scenes = 8000
seed_offset = 64000
root_dn = 'kubcon_v4'
root_dp = '/proj/vondrick3/basile/cv_' + root_dn
global_start_idx = 4000
global_end_idx = num_scenes
ignore_if_exist = True
scratch_dir = '/local/vondrick/basile/kubcon_scratch/'

frame_width = int(320 * 1.5)
frame_height = int(240 * 1.5)
num_frames = 40
frame_rate = 16
snitch_grid_size = 4
snitch_max_extent = 1.5
snitch_grid_step = 1.5
snitch_size_meters = 0.45


# NOTE: Because of /tmp size problems, we must stop and restart this script to empty the /tmp
# directory inbetween runs. This counter indicates when all threads should finish.
MAX_SCENE_COUNT = 400


def do_scene(scene_idx, logger, scene_dp, scene_dn):

    # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
    import kubric as kb
    import kubric_sim

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

    t = my_kubric.prepare_next_scene('train', seed_offset + scene_idx, fix_cam_start_yaw=True)
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
    snitch_x = (snitch_action % snitch_grid_size - snitch_max_extent)
    snitch_y = ((snitch_action // snitch_grid_size) % snitch_grid_size - snitch_max_extent)
    snitch_x *= snitch_grid_step
    snitch_y *= snitch_grid_step
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

    rgb_clean = data_stack2['rgba'][..., :3].copy()
    rgb_annot = rgb_clean.copy()
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

    imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_clean.mp4'),
                     rgb_clean, format='ffmpeg', fps=frame_rate, quality=10)
    imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_annot.mp4'),
                     rgb_annot, format='ffmpeg', fps=frame_rate, quality=10)
    imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_annot.gif'),
                     rgb_annot, format='gif', fps=frame_rate)

    t = my_kubric.write_all_data(os.path.join(scene_dp, 'frames'))
    logger.debug(f'write_all_data took {t:.2f}s')

    dst_json_fp = os.path.join(scene_dp, f'{scene_dn}.json')
    kb.write_json(metadata, dst_json_fp)

    logger.info(f'All together took {time.time() - start_time:.2f}s')


def worker(worker_idx, num_workers, total_scn_cnt):

    logger = logvisgen.Logger(msg_prefix=f'{root_dn}_worker{worker_idx}')

    my_start_idx = worker_idx + global_start_idx

    for scene_idx in range(my_start_idx, global_end_idx, num_workers):

        scene_dn = f'{root_dn}_scn{scene_idx:05d}'
        scene_dp = os.path.join(root_dp, scene_dn)

        logger.info()
        logger.info(scene_idx)
        logger.info(scene_dn)
        logger.info()

        dst_json_fp = os.path.join(scene_dp, f'{scene_dn}.json')
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

            # We perform the actual generation in a separate thread to ensure that no memory leaks
            # survive.
            p = mp.Process(target=do_scene, args=(scene_idx, logger, scene_dp, scene_dn))
            p.start()
            p.join()

        pass

    logger.info()
    logger.info(f'I am done!')
    logger.info()

    pass


class StubModule(torch.nn.Module):

    def __init__(self, num_threads):
        super().__init__()
        self.num_threads = num_threads

    def forward(self, x):
        print(f'forward called with arg {x}')

        thread_idx = x[0].item()
        sleep_s = thread_idx * 5.0
        print(f'sleeping {sleep_s} seconds before proceeding...')
        time.sleep(sleep_s)

        worker(thread_idx, self.num_threads)


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

        # model = StubModule(num_workers)
        # model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[0] * num_workers)

        # thread_inds = torch.arange(num_workers)
        # thread_inds = thread_inds.cuda()

        # print('Start stub model forward call...')

        # model(thread_inds)

        # print('Finished stub model forward call!')

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()

    pass
