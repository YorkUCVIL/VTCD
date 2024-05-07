

import os
import sys
import time
import random
import pickle

import argparse
import json

import numpy as np
import torch
from vcd import VideoConceptDiscovery as VTCD
from utilities.utils import load_model, save_vcd, prepare_directories

def main(args):
    # prepare directories
    print('Experiment name: {}'.format(args.exp_name))
    args = prepare_directories(args)
    vcd_path = os.path.join(args.vcd_dir, 'vcd.pkl')
    if args.preload_vcd and os.path.exists(vcd_path):
            print('Loading VTCD from {}'.format(vcd_path))
            vcd = pickle.load(open(vcd_path, 'rb'))
            # setting arguments
            args.model = vcd.args.model
            args.cluster_layer = vcd.args.cluster_layer
            args.cluster_subject = vcd.args.cluster_subject
            args.concept_clustering = vcd.args.concept_clustering

            print('Trying to load cached file from {}'.format(vcd.cached_file_path))
            if os.path.exists(vcd.cached_file_path) and not args.force_reload_videos:
                print('Loading cached file from {}'.format(vcd.cached_file_path))
                with open(vcd.cached_file_path, 'rb') as f:
                    vcd.dataset = pickle.load(f)
    else:
        # save args
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # load model
        model = load_model(args)

        # initialize vcd
        print('Initializing VTCD...')
        start_time = time.time()
        vcd = VTCD(args, model)
        end_time = time.time()
        print('Initializing VTCD took {:.2f} minutes'.format((end_time-start_time)/60))

        print('Tubelet clustering...')
        start_time = time.time()
        vcd.intra_video_clustering()
        end_time = time.time()
        print('Tubelet clustering took {:.2f} minutes'.format((end_time-start_time)/60))

        print('Concept clustering...')
        start_time = time.time()
        vcd.inter_video_clustering()
        end_time = time.time()
        print('Concept clustering took {:.2f} minutes'.format((end_time-start_time)/60))

    # 3) Compute DAVIS16 metrics
    if args.Eval_D16_VOS:
        print('Computing Davis16 performance...')
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
            raise NotImplementedError('Only DAVIS16 metrics are supported.')
        end_time = time.time()
        print('Running Davis16 val took {:.2f} minutes'.format((end_time-start_time)/60))
        exit()

    if args.save_concepts or args.save_single_concept_videos:
        start_time = time.time()
        print('Saving concepts...')
        vcd.save_concepts(args)
        end_time = time.time()
        print('Saving concepts took {:.2f} minutes'.format((end_time-start_time)/60))

    print('Saving vcd...')
    save_vcd(vcd)

def vcd_args():
    # todo: rename bad flags
    parser = argparse.ArgumentParser()

    # General experiment settings
    parser.add_argument('--exp_name', default='test', type=str,help='experiment name (used for saving)')
    parser.add_argument('--preload_vcd', action='store_true', help='Try to directly load saved vcd with experiment name.')

    # data
    parser.add_argument('--dataset', default='ssv2', type=str,help='Dataset to use. [kubric | ssv2 | davis16 | davis16_val]]')
    parser.add_argument('--target_class', default='Rolling something on a flat surface', type=str,help='Target class name for classification dataset')
    parser.add_argument('--max_num_videos', default=5, type=int,help='Number of videos to use during clustering.')
    parser.add_argument('--target_class_idxs', nargs='+', default=[], type=int,help='target class idx for multiple target class setting')
    parser.add_argument('--force_reload_videos', action='store_true',help='Flag to force reload videos and not use cache.')
    parser.add_argument('--dataset_cache', action='store_true',help='Option to cache dataset in memory.')
    parser.add_argument('--use_train', action='store_true', help='Option to use training set for ssv2')
    parser.add_argument('--kubric_path', default='/home/m2kowal/data/kubcon_v10', type=str,help='kubric path')
    parser.add_argument('--ssv2_path', default='/home/m2kowal/data/ssv2', type=str,help='kubric path')
    parser.add_argument('--davis16_path', default='/home/m2kowal/data/DAVIS', type=str,help='kubric path')

    # model and clustering settings
    parser.add_argument('--model', default='timesformer_occ', type=str,help='Model to run. [timesformer_occ | vidmae_ssv2_pre | vidmae_ssv2_ft | intern]')
    parser.add_argument('--process_full_video', action='store_true', help='Run VTCD on the full video length.')
    parser.add_argument('--cluster_layer', nargs='+', default=[10], type=int,
                        help='Layers to perform clustering at. [0-11]')
    parser.add_argument('--attn_head', nargs='+', default=[0], type=int,
                        help='Which heads to cluster. [0-11]')
    parser.add_argument('--cluster_subject', default='keys', type=str,
                        help='Subject to cluster within attention layers)', choices=['keys','queries','values'])
    parser.add_argument('--checkpoint_path', default='', type=str,help='Override checkpoint path for any model.')
    parser.add_argument('--use_temporal_attn', action='store_true', help='Flag to use temporal feature maps for timesformer.')


    # clustering
    parser.add_argument('--intra_cluster_method', default='slic', type=str, help='Method to use for intra clustering (dino | dino_og | slic | multikmeans | crop).')
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
    parser.add_argument('--n_segments', nargs='+', default=[12], help='Threshold for intra-video elbow method.')
    parser.add_argument('--slic_compactness', nargs='+', default=[0.1], help='Compactnesses to use for SLIC clustering.')
    parser.add_argument('--slic_spacing', nargs='+', default=[1,1,1], type=str, help='Spacing to use for SLIC clustering.')
    parser.add_argument('--spatial_resize_factor', default=0.5, type=float, help='Fraction of spatial video size to perform clustering at.')
    parser.add_argument('--temporal_resize_factor', default=1, type=float, help='Fraction of temporal video size to perform clustering at.')

    # Debugging - visualization for tubelet clustering
    parser.add_argument('--save_intra_segments', action='store_true',help='Flag to save intra-video segments.')
    parser.add_argument('--save_prediction', action='store_true',help='Flag prediction of model if applicable.')
    parser.add_argument('--save_intra_indv_segments', action='store_true',help='Flag to save intra-video segments as individual videos.')
    parser.add_argument('--save_concepts', action='store_true',help='Flag to save concept videos.')
    parser.add_argument('--save_concept_frames', action='store_true',help='Flag to save concepts as frames.')
    parser.add_argument('--save_intr_concept_videos_all_k', action='store_true',help='Flag to save intra-video segments for all k, debugging purposes.')
    parser.add_argument('--save_num_vids_per_concept', default=30, type=int, help='Sample interval for intra-video clustering.')
    parser.add_argument('--single_save_layer_head', nargs='+', default=[], type=int, help='Save a single layer and head and concepts')
    parser.add_argument('--save_single_concept_videos', action='store_true', help='Save a single layer and head')

    # metrics
    parser.add_argument('--Eval_D16_VOS', action='store_true',help='Flag to compute concept metrics.')
    parser.add_argument('--first_frame_query', action='store_true',help='Flag to compute concept metrics.')
    parser.add_argument('--post_sam', action='store_true',help='Flag to compute concept metrics.')
    parser.add_argument('--num_sam_points', default=1, type=int,help='')
    parser.add_argument('--sample_factor', default=8, type=int,help='')
    parser.add_argument('--use_last_centroid', action='store_true', help='')
    parser.add_argument('--use_box', action='store_true', help='')
    parser.add_argument('--sam_sampling_mode', default='random',help='')
    parser.add_argument('--train_head_search', action='store_true',help='Flag to compute concept metrics.')

    # computation and reproducibility
    parser.add_argument('--max_num_workers', default=16, type=int,help='Maximum number of threads for clustering')
    parser.add_argument('--seed', default=0, type=int,help='Random seed')

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

