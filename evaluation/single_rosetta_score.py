import os
import sys
sys.path.append('/home/ubuntu/video_concept_discovery/')

import pickle
import json
import argparse
import numpy as np
import random
import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
import vcd
import itertools
import math
import psutil
import gc
import matplotlib.pyplot as plt




def main(args):
    # for each vcd
    # open up vcd
    # get masks for the target concept
    # add to list

    # compute miou for mask combination

    # dsbs - objects before and after dropping
    # ssl 9 1 6
    # intern 8 3 2
    # tcow 10 3 4
    # vidmae 6 2 6


    concepts = []
    masks = []


    # load masks for all experiments
    for exp_idx, exp_name in enumerate(args.exp_name):
        vcd_path = os.path.join('results', exp_name, 'vcd', 'vcd.pkl')
        print('Loading VCD from {}...'.format(vcd_path))
        with open(vcd_path, 'rb') as f:
            vcd = pickle.load(f)
        max_num_videos = vcd.args.max_num_videos


        masks[exp_name] = all_concept_masks
        # add exp_name to the beginning of each concept
        all_concepts = [exp_name + ' ' + concept for concept in all_concepts]
        concepts[exp_name] = all_concepts
    del vcd

    # construct list of every single combination of concepts
    print('Number of combinations: {}'.format(len(all_concepts)))
    for concept_set in tqdm(all_concepts):
        # get all concepts and names
        exp_names = []
        concept_names = []
        for concept in concept_set:
            exp_name, concept_name = concept.split(' ', 1)
            exp_names.append(exp_name)
            concept_names.append(concept_name)

        # calculate iou for each video
        iou_list = []
        for video_idx in range(max_num_videos):
            # new
            iou = compute_mask_and_compute_iou(args, exp_names, concept_names, video_idx, masks)
            if psutil.virtual_memory().percent > 90:
                print('Memory usage too high, clearing memory...')
                gc.collect()
                torch.cuda.empty_cache()
                exit()
            # calculate iou for all masks
            iou_list.append(iou)
        try:
            miou = sum(iou_list) / len(iou_list)
        except:
            miou = 0.0

        # create string from concept_set alphabetically
        concept_set_name = ' '.join(sorted(concept_set))
        # store in rosetta_concepts
        rosetta_concepts[concept_set_name] = miou

    # save rosetta_concepts as json
    print('Saving rosetta_concepts to {}...'.format(results_path))
    with open(results_path, 'w') as f:
        json.dump(rosetta_concepts, f, indent=4)

    print('done yo')


def compute_mask_and_compute_iou(args, exp_names, concept_names, video_idx, masks):
    # get masks for each concept
    all_video_masks = []
    for exp_name, concept_name in zip(exp_names, concept_names):
        if args.preload_masks:
            all_video_masks.append(masks[exp_name][video_idx][concept_name])
        else:
            # get mask
            mask_paths = masks[exp_name][video_idx][concept_name]
            if len(mask_paths) > 0:
                video_masks = np.stack([np.load(mask_path) for mask_path in mask_paths])
                video_masks = (np.sum(video_masks, axis=0) > 0).astype(int)
                video_masks = torch.tensor(video_masks)

                try:
                    video_masks = F.interpolate(video_masks.unsqueeze(0).unsqueeze(0).type(torch.uint8), size=(30, 15, 20),
                                                mode='nearest').squeeze(0).squeeze(0).float()
                except:
                    video_masks = F.interpolate(video_masks.unsqueeze(0).unsqueeze(0).type(torch.uint8), size=(8, 14, 14),
                                                mode='nearest').squeeze(0).squeeze(0).float()

                # if video_idx not in all_concept_masks.keys():
                #     all_concept_masks[video_idx] = {}
                video_masks = video_masks.bool()
            else:
                # if no masks, construct a dummy mask of all 0's
                # if video_idx not in all_concept_masks.keys():
                #     all_concept_masks[video_idx] = {}
                video_masks = torch.zeros((30, 15, 20)).bool()
            all_video_masks.append(video_masks)

    # calculate iou for all masks
    iou = compute_iou_list(all_video_masks)
    return iou



def compute_iou(v1, v2):
    '''
    v1: (B, T, H, W)
    v2: (B, T, H, W) or mask
    '''

    # compute intersection and union
    intersection = torch.logical_and(v1, v2).sum()
    union = torch.logical_or(v1, v2).sum()

    # compute iou
    iou = intersection / union

    # if nan, set to 0
    if torch.isnan(iou):
        iou = 0
    return float(iou)

def compute_iou_list(data):
    # Compute the combined intersection and union
    intersection = torch.logical_and(data[0], data[1])
    union = torch.logical_or(data[0], data[1])
    for tensor in data[2:]:
        intersection = torch.logical_and(intersection, tensor)
        union = torch.logical_or(union, tensor)

    # Calculate IoU
    iou = (torch.sum(intersection.float()) / torch.sum(union.float())).item()
    # if nan, set to 0
    if math.isnan(iou):
        iou = 0

    return iou


def get_all_concept_masks(all_concepts, vcd, args, return_masks=False):
    # get all concepts in a huge list
    # each mask is a tensor of shape (num_frames, h, w) where 1's are the pixels that are part of the concept
    all_concept_masks = {}
    # for layer in tqdm(vcd.dic.keys()):
    #     for head in vcd.dic[layer].keys():
    #         for concept in vcd.dic[layer][head]['concepts']:
    #             all_concepts.append('Layer{}-Head{}-{}'.format(layer, head,concept))
    for concept in tqdm(all_concepts):
        layer, head, concept = concept.split('-')
        layer = int(layer.replace('Layer', ''))
        head = int(head.replace('Head', ''))

        for video_idx in range(vcd.args.max_num_videos):
            if args.debug:
                if video_idx == 3:
                    break
            mask_paths = [mask for mask in vcd.dic[layer][head][concept]['video_mask'] if'video_{}'.format(video_idx) in mask]
            if return_masks:
                if len(mask_paths) > 0:
                    masks = []
                    for mask_path in mask_paths:
                        with open(mask_path, 'rb') as f:
                            video_masks = np.load(f)
                            masks.append(video_masks)
                    masks = np.stack(masks)
                    # masks = np.stack([np.load(mask_path) for mask_path in mask_paths])
                    masks = (np.sum(masks, axis=0) > 0).astype(int)
                    masks = torch.tensor(masks)

                    try:
                        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8),size=(30, 15, 20), mode='nearest').squeeze(0).squeeze(0).float()
                    except:
                        masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8), size=(8, 14, 14), mode='nearest').squeeze(0).squeeze(0).float()

                    if video_idx not in all_concept_masks.keys():
                        all_concept_masks[video_idx] = {}
                    all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head,concept)] = masks.bool()
                else:
                    # if no masks, construct a dummy mask of all 0's
                    if video_idx not in all_concept_masks.keys():
                        all_concept_masks[video_idx] = {}
                    all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head,concept)] = torch.zeros((30, 15, 20)).bool()
            else:
                # return the mask paths instead
                if video_idx not in all_concept_masks.keys():
                    all_concept_masks[video_idx] = {}
                all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head,concept)] = mask_paths

                if psutil.virtual_memory().percent > 90:
                    print('Memory usage too high, clearing memory...')
                    gc.collect()
                    torch.cuda.empty_cache()
                    exit()
    return all_concept_masks

def get_all_concepts(vcd, args):
    # get all concepts in a huge list
    all_concepts = []
    # each mask is a tensor of shape (num_frames, h, w) where 1's are the pixels that are part of the concept

    # todo: only open the masks for the concepts we need

    for layer in tqdm(vcd.dic.keys()):
        for head in vcd.dic[layer].keys():
            for concept in vcd.dic[layer][head]['concepts']:
                all_concepts.append('Layer{}-Head{}-{}'.format(layer, head,concept))
                for video_idx in range(vcd.args.max_num_videos):
                    if args.debug:
                        if video_idx == 3:
                            break

                    if psutil.virtual_memory().percent > 90:
                        print('Memory usage too high, clearing memory...')
                        gc.collect()
                        torch.cuda.empty_cache()
                        exit()
    return all_concepts

# def get_all_concept_masks(vcd, args, return_masks=False):
#     # get all concepts in a huge list
#     all_concepts = []
#     # each mask is a tensor of shape (num_frames, h, w) where 1's are the pixels that are part of the concept
#     all_concept_masks = {}
#     for layer in tqdm(vcd.dic.keys()):
#         for head in vcd.dic[layer].keys():
#             for concept in vcd.dic[layer][head]['concepts']:
#                 all_concepts.append('Layer{}-Head{}-{}'.format(layer, head,concept))
#                 for video_idx in range(vcd.args.max_num_videos):
#                     if args.debug:
#                         if video_idx == 3:
#                             break
#                     mask_paths = [mask for mask in vcd.dic[layer][head][concept]['video_mask'] if'video_{}'.format(video_idx) in mask]
#                     if return_masks:
#                         if len(mask_paths) > 0:
#                             masks = np.stack([np.load(mask_path) for mask_path in mask_paths])
#                             masks = (np.sum(masks, axis=0) > 0).astype(int)
#                             masks = torch.tensor(masks)
#
#                             try:
#                                 masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8),size=(30, 15, 20), mode='nearest').squeeze(0).squeeze(0).float()
#                             except:
#                                 masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8), size=(8, 14, 14), mode='nearest').squeeze(0).squeeze(0).float()
#
#                             if video_idx not in all_concept_masks.keys():
#                                 all_concept_masks[video_idx] = {}
#                             all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head,concept)] = masks.bool()
#                         else:
#                             # if no masks, construct a dummy mask of all 0's
#                             if video_idx not in all_concept_masks.keys():
#                                 all_concept_masks[video_idx] = {}
#                             all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head,concept)] = torch.zeros((30, 15, 20)).bool()
#                     else:
#                         # return the mask paths instead
#                         if video_idx not in all_concept_masks.keys():
#                             all_concept_masks[video_idx] = {}
#                         all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head,concept)] = mask_paths
#
#                         if psutil.virtual_memory().percent > 90:
#                             print('Memory usage too high, clearing memory...')
#                             gc.collect()
#                             torch.cuda.empty_cache()
#                             exit()
#     return all_concepts, all_concept_masks

def vcd_args():

    parser = argparse.ArgumentParser()

    # general
    # parser.add_argument('--exp_name', nargs='+', default=['videomae_ssv2_keys_rsoafs_OneMask'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--exp_concepts', nargs='+', default=['videomae_ssv2_pre_keys_rsoafs_v2', 'videomae_ssv2_keys_rsoafs_v2', 'Occ_Keys_OG_rsoafs', 'Intern_Keys_SSv2_rsoafs'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--filter_concepts_by_importance', action='store_false', help='Filter Concepts by importance')
    # parser.add_argument('--importance_files', nargs='+', default=['ConceptHeadImporance_HW_4000MasksRatio095.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--importance_files', nargs='+', default=['ConceptHeadImporance_HW_4000MasksRatio095.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl', 'ConceptHeadImporance_HW_8000Masks.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--frac_important_concepts_keep', default=0.03, type=float, help='Fraction of top concepts to keep.')

    parser.add_argument('--fig_save_name', default='attribution_plot',help='figure name (used for saving)')
    parser.add_argument('--compare_multiclass', action='store_true', help='if true, average over multiple classes and then compare ')
    parser.add_argument('--overwrite_importance', action='store_true', help='Overwrite importance results')
    parser.add_argument('--overwrite_attribution', action='store_true', help='Overwrite addtribution results')
    # parser.add_argument('--results_name', default='test4', type=str,help='figure name (used for saving)')
    parser.add_argument('--results_name', default='hsws_TopK0075_FourModels', type=str,help='figure name (used for saving)')
    # parser.add_argument('--results_name', default='ModelPair_ssl_vidmae_rsoafs', type=str,help='figure name (used for saving)')

    # data
    parser.add_argument('--dataset', default='kubric', type=str,help='dataset to use')
    parser.add_argument('--kubric_path', default='/data/kubcon_v10', type=str,help='kubric path')
    parser.add_argument('--checkpoint_path', default='', type=str,help='Override checkpoint path.')

    # concept importance
    parser.add_argument('--cat_method', default='occlusion_soft', help='Method for concept importance calculation (occlusion_soft | occlusion_hard | ig [integrated gradients]).')
    # parser.add_argument('--cat_method', default='integrated_gradients', help='Method for concept importance calculation (occlusion_soft | occlusion_hard | ig [integrated gradients]).')
    parser.add_argument('--ig_steps', default=4, type=int, help='Number of ig steps.')
    parser.add_argument('--preload_masks', action='store_true', help='Preload masks')
    parser.add_argument('--random_importance', action='store_true', help='Use random concept importance.')
    parser.add_argument('--baseline_compare', action='store_true', help='Compare with random and inverse baselines.')
    parser.add_argument('--importance_loss', default='track', type=str,help='Loss to use for importance [track | occl_pct].')

    # attribution settings
    parser.add_argument('--attribution_evaluation_metric', nargs='+', default=['mean_snitch_iou'], type=str, help='Metrics to use during attribution calculation.')
    parser.add_argument('--zero_features', action='store_true', help='Zero out all other features during attribution.')
    parser.add_argument('--removal_type', default='perlay', help='type of attribution removal to do. [perlay | alllay | alllayhead]')

    # computation
    parser.add_argument('--max_num_workers', default=16, type=int,help='Maximum number of workers for clustering')

    # reproducibility
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
