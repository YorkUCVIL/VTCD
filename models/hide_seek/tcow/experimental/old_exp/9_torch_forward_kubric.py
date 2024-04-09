# BVH, Jun 2022.

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))

from __init__ import *


# Internal imports.
import logvisgen


# NOTE: This is mostly a copy of gen_dset/export_kubcon_v4.


# ==================================
# CUSTOMIZE DATASET PARAMETERS HERE:

num_workers = 2
num_scenes = 2
seed_offset = 54321
root_dn = 'exp9'
root_dp = os.path.join('unit_test', root_dn)
global_start_idx = 0
global_end_idx = num_scenes
ignore_if_exist = True

frame_width = 320
frame_height = 240
num_frames = 24
frame_rate = 12
snitch_size_meters = 0.45


def worker(worker_idx, num_workers):

    logger = logvisgen.Logger(msg_prefix=f'{root_dn}_worker{worker_idx}')

    # WARNING / NOTE: We CANNOT import bpy outside of the actual thread / process using it!
    import kubric as kb
    import kubric_sim

    render_cpu_threads = int(np.ceil(mp.cpu_count() / num_workers))

    # NOTE: This instance must only be created once per process!
    my_kubric = kubric_sim.MyKubricSimulatorRenderer(
        logger, frame_width=frame_width, frame_height=frame_height, num_frames=num_frames,
        frame_rate=frame_rate, render_cpu_threads=render_cpu_threads)

    my_start_idx = worker_idx + global_start_idx

    for scene_idx in range(my_start_idx, global_end_idx, num_workers):

        scene_dn = f'{root_dn}_scn{scene_idx:05d}'
        scene_dp = os.path.join(root_dp, scene_dn)

        logger.info()
        logger.info(scene_idx)
        logger.info(scene_dn)
        logger.info()

        dst_video_fp = os.path.join(scene_dp, f'{scene_dn}_clean.mp4')
        if ignore_if_exist and os.path.exists(dst_video_fp):
            logger.info(f'This video already exists at {dst_video_fp}, skipping!')
            continue

        else:
            os.makedirs(scene_dp, exist_ok=True)

            start_time = time.time()

            t = my_kubric.prepare_next_scene('train', seed_offset + scene_idx, fix_cam_start_yaw=True)
            logger.info(f'prepare_next_scene took {t:.2f}s')

            t = my_kubric.insert_static_objects(min_count=8, max_count=24,
                                                force_containers=2, force_carriers=1)
            logger.info(f'insert_static_objects took {t:.2f}s')

            (_, _, t) = my_kubric.simulate_frames(-50, 0)
            logger.info(f'simulate_frames took {t:.2f}s')

            t = my_kubric.reset_objects_velocity_friction_restitution()
            logger.info(f'reset_objects_velocity_friction_restitution took {t:.2f}s')

            t = my_kubric.insert_dynamic_objects(min_count=4, max_count=12)
            logger.info(f'insert_dynamic_objects took {t:.2f}s')

            (_, _, t) = my_kubric.simulate_frames(0, 2)
            logger.info(f'simulate_frames took {t:.2f}s')

            (data_stack1, t) = my_kubric.render_frames(0, 1)
            logger.info(f'render_frames took {t:.2f}s')

            snitch_action = my_kubric.random_state.randint(16)
            snitch_x = (snitch_action % 4 - 1.5) * 1.5
            snitch_y = ((snitch_action // 4) % 4 - 1.5) * 1.5
            logger.info(f'snitch_action: {snitch_action}  snitch_x: {snitch_x}  snitch_y: {snitch_y}')

            t = my_kubric.insert_snitch(at_x=snitch_x, at_y=snitch_y, size_meters=snitch_size_meters)
            logger.info(f'insert_snitch took {t:.2f}s')

            (_, _, t) = my_kubric.simulate_frames(2, num_frames)
            logger.info(f'simulate_frames took {t:.2f}s')

            (data_stack2, t) = my_kubric.render_frames(2, num_frames - 1)
            logger.info(f'render_frames took {t:.2f}s')

            # NOTE: data_stack2 contains both first and second render call results!

            (metadata, t) = my_kubric.get_metadata()
            logger.info(f'get_metadata took {t:.2f}s')

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

            dst_video_fp = os.path.join(scene_dp, f'{scene_dn}_clean.mp4')
            imageio.mimwrite(dst_video_fp,
                            rgb_clean, format='ffmpeg', fps=frame_rate, quality=10)
            imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_annot.mp4'),
                            rgb_annot, format='ffmpeg', fps=frame_rate, quality=10)
            imageio.mimwrite(os.path.join(scene_dp, f'{scene_dn}_annot.gif'),
                            rgb_annot, format='gif', fps=frame_rate)

            t = my_kubric.write_all_data(os.path.join(scene_dp, 'frames'))
            logger.info(f'write_all_data took {t:.2f}s')

            kb.write_json(metadata, os.path.join(scene_dp, f'{scene_dn}.json'))

            logger.info(f'All together took {time.time() - start_time:.2f}s')

        pass

    logger.info()
    logger.info(f'I am done!')
    logger.info()

    pass


class MyPipeline(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inds):
        within_batch_idx = inds[0].item()
        time.sleep(within_batch_idx)
        worker(within_batch_idx, num_workers)


def main():

    if 0:

        # NOTE: These are NOT actually separate processes -- I have to augment DataParallel somehow!

        assert num_workers == 2

        device = torch.device('cuda')
        net = MyPipeline()
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=[0, 1])

        inds = torch.arange(0, num_workers, dtype=torch.int32).to(device)
        y = net(inds)

    else:

        processes = [mp.Process(target=worker, args=(worker_idx, num_workers))
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
