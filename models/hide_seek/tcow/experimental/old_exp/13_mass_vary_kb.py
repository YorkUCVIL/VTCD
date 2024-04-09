# BVH, Aug 2022.

# NOTE: This is mostly based on export_kubbench_v2.py.

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'gen_bench/'))

from __init__ import *

# Library imports.
import json

# Internal imports.
import logvisgen
import scene_type_utils


num_workers = 0
root_dn = 'exp13b'
# root_dp = os.path.join(r'C:\Development\CVR Data\Hide & Seek\hs_kubexp_out', root_dn)
root_dp = os.path.join('/proj/vondrick3/basile/hs_kubexp_out/', root_dn)
# invest_dp = r'C:\Development\CVR Data\Hide & Seek\kubbench_v2_cv'
invest_dp1 = '/proj/vondrick3/basile/kubbench_v2_cv/'
invest_dp2 = '/proj/vondrick3/basile/kubcon_v8_tri_test/'
dst_dp = 'experimental/figs'

scene_types = [
    # 'gravity_bounce',
    # 'fall_onto_carrier',
    # 'fall_into_container',
    # 'slide_box_friction',
    # 'slide_box_collide',
    # 'box_push_carrier_slide',
    'box_push_container_slide',
]
num_scenes_per_type = 10
num_total_scenes = len(scene_types) * num_scenes_per_type
global_start_idx = 0
global_end_idx = 999

# num_perturbs = 3
# num_views = 2
num_perturbs = 1
num_views = 1
num_queries = 192

# NOTE: Point to exact scene I want to reproduce by setting index here.
seed_offset = 635222 + 6 * num_scenes_per_type + 9
ignore_if_exist = False

frame_width = int(320 * 1.5)
frame_height = int(240 * 1.5)
num_frames = 44
frame_rate = 16
render_samples_per_pixel = 32
# frame_width = 320
# frame_height = 240
# num_frames = 24
# frame_rate = 12
# render_samples_per_pixel = 16


def plot_dset_stats(invest_dp, fn_prefix):
    scene_dns = sorted(os.listdir(invest_dp))
    scene_dps = [os.path.join(invest_dp, dn) for dn in scene_dns]
    scene_dps = [dp for dp in scene_dps if os.path.isdir(dp)]
    my_data = []

    for scene_dp in tqdm.tqdm(scene_dps):
        scene_dn = os.path.basename(scene_dp)
        metadata_fp = os.path.join(scene_dp, f'{scene_dn}_p0_v0.json')
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
        
        for instance in metadata['instances']:
            scale_factor = instance["scale_factor"]
            volume_pre = instance["volume"] * 1000.0
            volume_post = instance["volume"] * 1000.0 * (scale_factor ** 3.0)
            mass_pre = instance["mass"] * 1000.0 / scale_factor
            mass_post = instance["mass"] * 1000.0
            
            my_data.append([scale_factor, volume_pre, mass_pre, volume_post, mass_post])

    my_data = np.array(my_data)
    os.makedirs(dst_dp, exist_ok=True)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure()
    plt.scatter(my_data[:, 0], my_data[:, 1], label='Instances')
    plt.xlabel('scale_factor')
    plt.ylabel('volume_pre')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dst_dp, fn_prefix + '_scale_factor_volume_pre.png'), dpi=384)

    plt.figure()
    plt.scatter(my_data[:, 0], my_data[:, 2], label='Instances')
    plt.xlabel('scale_factor')
    plt.ylabel('mass_pre')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dst_dp, fn_prefix + '_scale_factor_mass_pre.png'), dpi=384)

    plt.figure()
    plt.scatter(my_data[:, 1], my_data[:, 2], label='Instances', color=cycle[0])
    plt.xlabel('volume_pre')
    plt.ylabel('mass_pre')
    plt.plot([my_data[:, 1:3].min(), my_data[:, 1:3].max()],
             [my_data[:, 1:3].min(), my_data[:, 1:3].max()],
             label='Density = 1', color=cycle[1])
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dst_dp, fn_prefix + '_volume_pre_mass_pre.png'), dpi=384)
    
    plt.figure()
    plt.scatter(my_data[:, 3], my_data[:, 4], label='Instances', color=cycle[0])
    plt.xlabel('volume_post')
    plt.ylabel('mass_post')
    plt.plot([my_data[:, 3:5].min(), my_data[:, 3:5].max()],
             [my_data[:, 3:5].min(), my_data[:, 3:5].max()],
             label='Density = 1', color=cycle[1])
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dst_dp, fn_prefix + '_volume_post_mass_post.png'), dpi=384)
    
    plt.figure()
    plt.hist(my_data[:, 0], bins=32, label='Instances')
    plt.xlabel('scale_factor')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dst_dp, fn_prefix + '_scale_factor_hist.png'), dpi=384)

    print()
    pass


def investigate_scale_mass_stats():

    # Print instance info for kubbench_v2 scene index 69.
    if 1:
        scene_dns = sorted(os.listdir(invest_dp1))
        scene_dps = [os.path.join(invest_dp1, dn) for dn in scene_dns]
        scene_dps = [dp for dp in scene_dps if os.path.isdir(dp)]

        for scene_dp in scene_dps:
            scene_dn = os.path.basename(scene_dp)

            if not '69' in scene_dn:
                continue

            print()
            print('=> ' + scene_dn)
            metadata_fp = os.path.join(scene_dp, f'{scene_dn}_p0_v0.json')
            with open(metadata_fp, 'r') as f:
                metadata = json.load(f)

            for instance in metadata['instances']:
                print(f'asset_id: {instance["asset_id"]}  '
                      f'scale_factor: {instance["scale_factor"]:.2f}  '
                      f'volume: {instance["volume"] * 1000.0:.2f}  '
                      f'mass: {instance["mass"] * 1000.0:.2f}')

            print()

    # Plot dataset stats for kubbench_v2.
    if 0:
        plot_dset_stats(invest_dp1, 'kb2')

    # Plot dataset stats for kubcon_v8_test.
    if 0:
        plot_dset_stats(invest_dp2, 'kc8test')

    print()
    pass


def do_scene_here(scene_idx, logger, scene_dp, scene_dn, scene_type):

    scene_type_utils.generate_for_type(
        logger, seed_offset, scene_idx, scene_dp, scene_dn, scene_type,
        num_workers, num_perturbs, num_views, num_queries,
        frame_width, frame_height, num_frames, frame_rate, render_samples_per_pixel)

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
            # We perform the actual generation in a separate thread to try to ensure that no memory
            # leaks survive.
            # Also, weird errors & crashes seem to occur otherwise..
            p = mp.Process(target=do_scene_here, args=(
                scene_idx, logger, scene_dp, scene_dn, scene_type))
            p.start()
            p.join()

        pass

    logger.info()
    logger.info(f'I am done!')
    logger.info()

    pass


def main():

    investigate_scale_mass_stats()

    if 1:

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
