

import os
import sys
import time
import random
import pickle

import argparse
import json

import numpy as np
import torch
import torchvision
from vcd import VideoConceptDiscovery as VCD
from utilities.utils import load_model, save_vcd, prepare_directories

def main(args):
    print('Experiment name: {}'.format(args.exp_name))
    # tmp_path = '/data/kubcon_v10/val/v1_Max50.pkl'
    # with open(tmp_path, 'rb') as f:
    #     dataset = pickle.load(f)

    # prepare directories
    args = prepare_directories(args)
    vcd_path = os.path.join(args.vcd_dir, 'vcd.pkl')
    if args.preload_vcd and os.path.exists(vcd_path):
            print('Loading VCD from {}'.format(vcd_path))
            vcd = pickle.load(open(vcd_path, 'rb'))
            args.model = vcd.args.model
            args.cluster_layer = vcd.args.cluster_layer
            args.cluster_subject = vcd.args.cluster_subject
            args.concept_clustering = vcd.args.concept_clustering
            args.cluster_memory = vcd.args.cluster_memory
            print('Trying to load cached file from {}'.format(vcd.cached_file_path))

            if os.path.exists(vcd.cached_file_path) and not args.force_reload_videos:
                print('Loading cached file from {}'.format(vcd.cached_file_path))
                if args.exp_name == 'Occ_Keys_OG_dsbs':
                    vcd.cached_file_path = vcd.cached_file_path.replace('v1', 'Rosetta')
                with open(vcd.cached_file_path, 'rb') as f:
                    vcd.dataset = pickle.load(f)
            # if args.force_reload_videos:
            #     if 'davis16' in vcd.args.dataset:
            #         vcd.dataset = vcd.load_davis16_videos()

            # else:
            #     print('Cached file does not exist, reloading videos')

            #
            #
            # vcd.cached_file_path = vcd.cached_file_path.replace('__', '_')
            # if os.path.exists(vcd.cached_file_path) and not args.force_reload_videos:
            #     with open(vcd.cached_file_path, 'rb') as f:
            #         vcd.dataset2 = pickle.load(f)

            size = (240,320) if 'timesformer' in args.model else (224,224)
            vcd.post_resize_nearest = torchvision.transforms.Resize(
                size,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    else:
        # save args
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # 0a) load model
        model = load_model(args)

        # 0b) initialize vcd
        print('Initializing VCD...')
        start_time = time.time()
        vcd = VCD(args, model)
        end_time = time.time()
        print('Initializing VCD took {:.2f} minutes'.format((end_time-start_time)/60))

        # 1) intra video clustering
        print('Intra video clustering...')
        start_time = time.time()
        vcd.intra_video_clustering()
        end_time = time.time()
        print('Intra video clustering took {:.2f} minutes'.format((end_time-start_time)/60))

        print('Inter video clustering...')
        # 2) inter video clustering
        start_time = time.time()
        vcd.inter_video_clustering()
        end_time = time.time()
        print('Inter video clustering took {:.2f} minutes'.format((end_time-start_time)/60))

    # 3) Compute DAVIS16 metrics
    if args.compute_concept_metrics:
        print('Computing concept metrics...')
        start_time = time.time()
        if 'davis16' in vcd.args.dataset:
            from evaluation.vtcd_vos_eval import compute_davis16_vos_score
            compute_davis16_vos_score(vcd,
                              first_frame_query=args.first_frame_query,
                              train_head_search=args.train_head_search,
                              post_sam=args.post_sam,
                              num_points=args.num_sam_points,
                              mode=args.sam_sampling_mode,
                              sample_factor=args.sample_factor,
                              use_last_centroid=args.use_last_centroid,
                              use_box=args.use_box,)
        else:
            raise NotImplementedError

        end_time = time.time()
        print('Computing concept metrics took {:.2f} minutes'.format((end_time-start_time)/60))

    if args.save_concepts or args.save_single_concept_videos:
        start_time = time.time()
        print('Saving concepts...')
        vcd.save_concepts(args)
        end_time = time.time()
        print('Saving concepts took {:.2f} minutes'.format((end_time-start_time)/60))

    if args.save_vcd:
        print('Saving vcd...')
        save_vcd(vcd)
def vcd_args():
    # todo: rename bad flags
    parser = argparse.ArgumentParser()

    # General experiment settings
    parser.add_argument('--exp_name', default='test', type=str,help='experiment name (used for saving)')
    parser.add_argument('--preload_vcd', action='store_true', help='Try to directly load saved vcd with experiment name.')

    # data
    # parser.add_argument('--dataset', default='kubric', type=str,help='dataset to use')
    parser.add_argument('--dataset', default='ssv2', type=str,help='dataset to use')
    # parser.add_argument('--dataset', default='davis16', type=str,help='dataset to use')
    # parser.add_argument('--dataset', default='davis16_val', type=str,help='dataset to use')
    parser.add_argument('--kubric_path', default='/data/kubcon_v10', type=str,help='kubric path')
    parser.add_argument('--ssv2_path', default='/data/ssv2', type=str,help='kubric path')
    parser.add_argument('--davis16_path', default='/data/DAVIS16', type=str,help='kubric path')
    parser.add_argument('--target_class', default='Rolling something on a flat surface', type=str,help='target class name for classification dataset')
    # parser.add_argument('--target_class', default='Hitting something with something', type=str,help='target class name for classification dataset')
    # parser.add_argument('--target_class', default='Dropping something behind something', type=str,help='target class name for classification dataset')
    parser.add_argument('--target_class_idxs', nargs='+', default=[], type=int,help='target class idx for multiple target class setting')
    parser.add_argument('--max_num_videos', default=31, type=int,help='Number of videos to use during clustering.')
    parser.add_argument('--force_reload_videos', action='store_true',help='Maximum number of videos to use during clustering.')
    parser.add_argument('--cache_name', default='v1', type=str,help='experiment name (used for saving)')
    parser.add_argument('--use_train', action='store_true', help='experiment name (used for saving)')

    parser.add_argument('--checkpoint_path', default='', type=str,help='Override checkpoint path for any model.')

    # model and clustering settings
    # parser.add_argument('--model', default='timesformer_occ', type=str,help='Model to run.')
    parser.add_argument('--model', default='vidmae_ssv2_pre', type=str,help='Model to run.')
    # parser.add_argument('--model', default='vidmae_ssv2_ft', type=str,help='Model to run.')
    # parser.add_argument('--model', default='intern', type=str,help='Model to run.')
    parser.add_argument('--cluster_layer', nargs='+', default=[10], type=int,
                        help='Layers to perform clustering at.')
    parser.add_argument('--attn_head', nargs='+', default=[0], type=int,
                        help='Which heads to cluster.')
    parser.add_argument('--cluster_subject', default='keys', type=str,
                        help='Subject to cluster within attention layers)', choices=['keys','queries','values'])
    parser.add_argument('--use_temporal_attn', action='store_true', help='Flag to use temporal feature maps for timesformer.')


    # clustering
    parser.add_argument('--intra_cluster_method', default='slic', type=str, help='Method to use for intra clustering (max | yellowbrick | dino | dino_og | slic | multikmeans | crop).')
    parser.add_argument('--inter_cluster_method', default='cnmf_elbow', type=str, help='K determination method to use for inter clustering (max | yellowbrick | dino | dino_og | cnmf | cnmf_elbow).')
    parser.add_argument('--pool_feature', action='store_false', help='Flag to perform spatial pooling before inter clustering.')
    parser.add_argument('--pool_non_zero_features', action='store_false', help='Flag to ignore zeros during pooling.')
    parser.add_argument('--concept_clustering', action='store_false', help='Flag to perform concept clustering.')
    parser.add_argument('--intra_elbow_threshold', nargs='+', default=[0.95], help='Threshold for intra-video elbow method.')
    parser.add_argument('--inter_elbow_threshold', default=0.95, type=float, help='Threshold for inter-video elbow method.')
    parser.add_argument('--intra_max_cluster', default=10, type=int, help='Maximum number of clusters to use for intra-video clustering.')
    parser.add_argument('--inter_max_cluster', default=15, type=int, help='Maximum number of clusters to use for inter-video clustering.')
    parser.add_argument('--sample_interval', default=10, type=int, help='Sample interval for intra-video clustering.')
    parser.add_argument('--intra_inter_cluster', action='store_false', help='Flag to perform clustering across all videos at intra.')
    parser.add_argument('--inter_norm', action='store_true', help='Flag to normalize features during inter-clustering.')
    parser.add_argument('--n_segments', nargs='+', default=[12], help='Threshold for intra-video elbow method.')
    parser.add_argument('--slic_compactness', nargs='+', default=[0.1], help='Compactnesses to use for SLIC clustering.')
    parser.add_argument('--slic_spacing', nargs='+', default=[1,1,1], type=str, help='spacing to use for SLIC clustering.')
    parser.add_argument('--spatial_resize_factor', default=0.5, type=float, help='Fraction of video size to perform clustering at.')
    parser.add_argument('--temporal_resize_factor', default=1, type=float, help='Fraction of video size to perform clustering at.')
    # parser.add_argument('--exit_after_intra', action='store_true', help='Flag to perform clustering across all videos at intra.')

    # visualization intra_cluster_all_videos
    parser.add_argument('--save_intra_segments', action='store_true',help='Flag to save intra-video segments.')
    parser.add_argument('--save_prediction', action='store_true',help='Flag prediction of model if applicable.')

    parser.add_argument('--save_intra_indv_segments', action='store_true',help='Flag to save intra-video segments as individual videos.')
    parser.add_argument('--save_concepts', action='store_true',help='Flag to save concept videos.')
    parser.add_argument('--save_concept_frames', action='store_true',help='Flag to save concepts as frames.')
    parser.add_argument('--save_intr_concept_videos_all_k', action='store_true',help='Flag to save intra-video segments for all k, debugging purposes.')
    parser.add_argument('--save_vcd', action='store_false',help='Flag to save vcd')
    parser.add_argument('--save_num_vids_per_concept', default=30, type=int, help='Sample interval for intra-video clustering.')
    parser.add_argument('--single_save_layer_head', nargs='+', default=[], type=int, help='Save a single layer and head and concepts')
    parser.add_argument('--save_single_concept_videos', action='store_true', help='Save a single layer and head')

    # concept importance
    # parser.add_argument('--cat_method', default='occlusion_soft', help='Method for concept importance calculation (occlusion_soft | occlusion_hard | ig [integrated gradients]).')
    parser.add_argument('--cat_method', default='integrated_gradients', help='Method for concept importance calculation (occlusion_soft | occlusion_hard | ig [integrated gradients]).')
    parser.add_argument('--ig_steps', default=4, type=int, help='Number of ig steps.')
    parser.add_argument('--random_importance', action='store_true', help='Use random concept importance.')
    parser.add_argument('--baseline_compare', action='store_true', help='Compare with random and inverse baselines.')
    parser.add_argument('--importance_loss', default='track', type=str,help='Loss to use for importance [track | occl_pct].')
    # attribution settings
    parser.add_argument('--zero_features', action='store_true', help='Zero out all other features during attribution.')
    parser.add_argument('--process_full_video', action='store_true', help='Run VTCD on the full video length.')
    parser.add_argument('--removal_type', default='perlay', help='type of attribution removal to do. [perlay | alllay | alllayhead]')
    parser.add_argument('--attribution_evaluation_metric', nargs='+', default=['mean_snitch_iou'], type=str, help='Metrics to use during attribution calculation.')

    # metrics
    parser.add_argument('--compute_concept_metrics', action='store_true',help='Flag to compute concept metrics.')
    parser.add_argument('--first_frame_query', action='store_true',help='Flag to compute concept metrics.')
    parser.add_argument('--post_sam', action='store_true',help='Flag to compute concept metrics.')
    parser.add_argument('--num_sam_points', default=1, type=int,help='')
    parser.add_argument('--sample_factor', default=8, type=int,help='')
    parser.add_argument('--use_last_centroid', action='store_true', help='')
    parser.add_argument('--use_box', action='store_true', help='')
    parser.add_argument('--sam_sampling_mode', default='random',help='')
    parser.add_argument('--train_head_search', action='store_true',help='Flag to compute concept metrics.')
    parser.add_argument('--compute_concept_importance', action='store_true',help='Flag to compute concept metrics.')
    # parser.add_argument('--cont_threshold_path', default='evaluation/thresholds/thresholds_01.json', type=str,help='path to concept thresholds')
    parser.add_argument('--cont_threshold_path', default='', type=str,help='path to concept thresholds')

    # computation
    parser.add_argument('--max_num_workers', default=16, type=int,help='Maximum number of workers for clustering')

    # reproducibility and debugging
    parser.add_argument('--seed', default=0, type=int,help='seed')
    parser.add_argument('--debug', action='store_true', help='Debug using only 2 videos for all functions.')

    args = parser.parse_args(sys.argv[1:])

    # random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return args

if __name__ == '__main__':
    start_time = time.time()
    vcd_args = vcd_args()
    main(vcd_args)
    print('Total time in minutes: {:.2f}'.format((time.time()-start_time)/60))

