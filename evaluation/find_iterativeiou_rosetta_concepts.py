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


def n_tuple_combinations(elements, n):
    """
    This function generates all possible n-tuple combinations from a given list of elements.

    :param elements: List of elements from which the combinations are to be made.
    :param n: The size of each combination tuple. It should be less than or equal to the length of elements list.
    :return: A list of all possible n-tuple combinations.
    """
    # Checking if n is less than or equal to the length of the elements list
    if n > len(elements):
        raise ValueError("The value of n should be less than or equal to the length of the elements list.")

    # Using itertools.combinations to generate all possible n-tuple combinations
    return list(itertools.combinations(elements, n))

def measure_iou_named_concepts(experiments, concepts, args):
    all_exp_masks = []

    for i in range(len(concepts)):

        exp_name = experiments[i]
        # open vcd
        vcd_path = os.path.join('results', exp_name, 'vcd', 'vcd.pkl')
        print('Loading VCD from {}...'.format(vcd_path))
        with open(vcd_path, 'rb') as f:
            vcd = pickle.load(f)
        layer, head, concept = concepts[i].split('-')
        layer = int(layer.replace('Layer', ''))
        head = int(head.replace('Head', ''))
        per_video_masks = []
        for video_idx in range(30):
            mask_paths = [mask for mask in vcd.dic[layer][head][concept]['video_mask'] if 'video_{}'.format(video_idx) in mask]
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
                    masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8), size=(30, 15, 20),
                                          mode='nearest').squeeze(0).squeeze(0).float()
                except:
                    masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8), size=(8, 14, 14),
                                          mode='nearest').squeeze(0).squeeze(0).float()
                masks = masks.bool()

                # if video_idx not in all_concept_masks.keys():
                #     all_concept_masks[video_idx] = {}
                # all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head, concept)] = masks.bool()
            else:
            #     if no masks, construct a dummy mask of all 0's
            #     if video_idx not in all_concept_masks.keys():
            #         all_concept_masks[video_idx] = {}
            #     all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head, concept)] = torch.zeros((30, 15, 20)).bool()
                masks = torch.zeros((30, 15, 20)).bool()

            per_video_masks.append(masks)
        all_video_masks = torch.stack(per_video_masks) # n t h w
        all_exp_masks.append(all_video_masks)

    iou = compute_iou_list(all_exp_masks)
    return iou

def create_all_combinations(lists):
    # Base case: if only one list left, return its elements as individual sets
    if len(lists) == 1:
        return [set([item]) for item in lists[0]]

    # Recursive case: get all combinations for the remaining lists
    remaining_combinations = create_all_combinations(lists[1:])

    # For each item in the current list, pair with each combination from the remaining lists
    result = []
    for item in lists[0]:
        for combination in remaining_combinations:
            result.append({item}.union(combination))
            if psutil.virtual_memory().percent > 90:
                print('Memory usage too high, clearing memory...')
                gc.collect()
                torch.cuda.empty_cache()
                exit()

    return result

def main(args):
    # load vcd's from list of models that use the same dataset
    rosetta_dir = os.path.join('results', 'Rosetta')
    if not os.path.exists(rosetta_dir):
        os.makedirs(rosetta_dir, exist_ok=True)
    results_path = os.path.join(rosetta_dir, args.results_name + '_rosetta_concepts.json')
    print('Save path: {}'.format(results_path))

    model_names = ['ssl', 'vidmae', 'tcow', 'Intern']
    model_names = sorted(model_names)
    model_name_1 = model_names[0]
    model_name_2 = model_names[1]
    model_name_3 = model_names[2]
    model_name_4 = model_names[3]
    curr_results_path = 'results/Rosetta/ConceptIoU_Filter{}_{}_{}_{}_{}_{}.json'.format(
        str(args.rosetta_iou_thresholds[1]).replace('.', ''), model_name_1, model_name_2, model_name_3, model_name_4,
        results_path.split('/')[-1].split('_')[0])

    # open results
    results_path_no_repeat = os.path.join(rosetta_dir, args.results_name + '_no_repeats.json')
    if os.path.exists(results_path_no_repeat):
        print('Loading results from {}...'.format(results_path_no_repeat))
        with open(results_path_no_repeat, 'r') as f:
            results_four_no_repeat = json.load(f)
        sorted_rosetta_concepts = sorted(results_four_no_repeat.items(), key=lambda x: x[1][1], reverse=True)
        # filter by threshold
        sorted_rosetta_concepts_filter = [x[1] for x in sorted_rosetta_concepts if x[1][1] > 0]
        # calculate average iou
        avg_iou = sum([x[1] for x in sorted_rosetta_concepts_filter])/len(sorted_rosetta_concepts_filter)
        print()



    if os.path.exists(curr_results_path):
        print('Results already exist for four tuple at location: {}'.format(curr_results_path))
        # load results
        if os.path.exists(curr_results_path):
            with open(curr_results_path, 'r') as f:
                results_four = json.load(f)
        # results = {key: value for key, value in results_four.items() if value > args.rosetta_iou_thresholds[2]}
        sorted_rosetta_concepts = sorted(results_four.items(), key=lambda x: x[1], reverse=True)

        # recreate list ranking but without repeating concepts
        global_concept_list1 = []
        global_concept_list2 = []
        global_concept_list3 = []
        global_concept_list4 = []
        sorted_rosetta_concepts_no_repeats = {}
        # use tqdm to show progress bar
        for idx, concept in enumerate(tqdm(sorted_rosetta_concepts)):
            concept_name = concept[0]
            concept_score = concept[1]
            model_concept_names = concept_name.split(' ')
            concept_1 = model_concept_names[0] + ' ' + model_concept_names[1]
            concept_2 = model_concept_names[2] + ' ' + model_concept_names[3]
            concept_3 = model_concept_names[4] + ' ' + model_concept_names[5]
            concept_4 = model_concept_names[6] + ' ' + model_concept_names[7]

            if idx == 0:
                sorted_rosetta_concepts_no_repeats[idx] = concept
            else:
                if concept_1 in global_concept_list1 or concept_2 in global_concept_list2 or concept_3 in global_concept_list3 or concept_4 in global_concept_list4:
                    continue
                else:
                    sorted_rosetta_concepts_no_repeats[idx] = concept
            if concept_1 not in global_concept_list1:
                global_concept_list1.append(concept_1)
            if concept_2 not in global_concept_list2:
                global_concept_list2.append(concept_2)
            if concept_3 not in global_concept_list3:
                global_concept_list3.append(concept_3)
            if concept_4 not in global_concept_list4:
                global_concept_list4.append(concept_4)

        # save results
        results_path_no_repeat = os.path.join(rosetta_dir, args.results_name + '_no_repeats.json')
        print('Saving results to {}...'.format(results_path_no_repeat))
        with open(results_path_no_repeat, 'w') as f:
            json.dump(sorted_rosetta_concepts_no_repeats, f)


        print()
        exit()

    print('Loading all concepts...')
    # load all concepts for each experiment
    all_concepts = {}
    for exp_name in args.exp_name:
        vcd_path = os.path.join('results', exp_name, 'vcd', 'vcd.pkl')
        print('Loading VCD from {}...'.format(vcd_path))
        with open(vcd_path, 'rb') as f:
            vcd = pickle.load(f)
        max_num_videos = vcd.args.max_num_videos
        all_concepts[exp_name] = get_all_concepts(vcd, args)

    # Iterative IoU filtering
    print('Filtering two tuples...')
    experiment_combinations = n_tuple_combinations(args.exp_name, 2)
    for experiment_combination in experiment_combinations:
        # experiment_combination = experiment_combinations[5]
        # check if results exist for this combination
        model_names = []
        if 'videomae_ssv2_pre_keys' in experiment_combination[0] or 'videomae_ssv2_pre_keys' in experiment_combination[1]:
            model_names.append('ssl')
        if 'videomae_ssv2_keys' in experiment_combination[0] or 'videomae_ssv2_keys' in experiment_combination[1]:
            model_names.append('vidmae')
        if 'Occ' in experiment_combination[0] or 'Occ' in experiment_combination[1]:
            model_names.append('tcow')
        if 'Intern' in experiment_combination[0] or 'Intern' in experiment_combination[1]:
            model_names.append('Intern')

        model_name_1 = model_names[0]
        model_name_2 = model_names[1]
        curr_results_path1 = 'results/Rosetta/PairConceptIoU_{}_{}_{}.json'.format(model_name_1, model_name_2, results_path.split('/')[-1].split('_')[0])
        curr_results_path2 = 'results/Rosetta/PairConceptIoU_{}_{}_{}.json'.format(model_name_2, model_name_1, results_path.split('/')[-1].split('_')[0])
        # curr_results_path1 = 'results/Rosetta/ModelPair_{}_{}_{}_rosetta_concepts.json'.format(model_name_1, model_name_2, results_path.split('/')[-1].split('_')[0])
        # curr_results_path2 = 'results/Rosetta/ModelPair_{}_{}_{}_rosetta_concepts.json'.format(model_name_2, model_name_1, results_path.split('/')[-1].split('_')[0])
        if os.path.exists(curr_results_path1) or os.path.exists(curr_results_path2):
            print('Results already exist for this combination: {}'.format(experiment_combination))
            # load results
            if os.path.exists(curr_results_path1):
                with open(curr_results_path1, 'r') as f:
                    results_two = json.load(f)
            else:
                with open(curr_results_path2, 'r') as f:
                    results_two = json.load(f)
        else:
            # compute and save results
            print('Computing results for this combination: {}'.format(experiment_combination))
            print('Saving results to: {}'.format(curr_results_path1))
            results_two = compute_rosetta_concepts(list(experiment_combination), args, save_path = curr_results_path1)
        # python evaluation/find_iterativeiou_rosetta_concepts.py --preload_masks
        print('Filtering results via rosetta IoU threshold {}...'.format(args.rosetta_iou_thresholds[0]))
        thresh_concept_results = {key: value for key, value in results_two.items() if value > args.rosetta_iou_thresholds[0]}

        # remove all concepts in either model that have no overlaps above the threshold
        for model_name in experiment_combination:

            # get all concepts for both model
            model_concepts_1 = []
            model_concepts_2 = []
            for concept_pair_name in thresh_concept_results.keys():
                model_1, concept_1, model_2, concept_2 = concept_pair_name.split(' ')
                model_concept_1 = '{} {}'.format(model_1, concept_1)
                model_concept_2 = '{} {}'.format(model_2, concept_2)
                model_concepts_1.append(model_concept_1)
                model_concepts_2.append(model_concept_2)

            # remove all concepts in either model that have no overlaps above the threshold
            concepts = all_concepts[model_name]
            for concept in concepts:
                model_concept = '{} {}'.format(model_name, concept)
                if model_concept not in model_concepts_1 and model_concept not in model_concepts_2:
                    all_concepts[model_name].remove(concept)
    
    print('Filtering three tuples...')
    # Iterative IoU filtering
    experiment_combinations = n_tuple_combinations(args.exp_name, 3)
    for experiment_combination in experiment_combinations:
        # experiment_combination = experiment_combinations[3]
        # check if results exist for this combination
        model_names = []
        if 'videomae_ssv2_pre_keys' in experiment_combination[0] or 'videomae_ssv2_pre_keys' in experiment_combination[1] or 'videomae_ssv2_pre_keys' in experiment_combination[2]:
            model_names.append('ssl')
        if 'videomae_ssv2_keys' in experiment_combination[0] or 'videomae_ssv2_keys' in experiment_combination[1] or 'videomae_ssv2_keys' in experiment_combination[2]:
            model_names.append('vidmae')
        if 'Occ' in experiment_combination[0] or 'Occ' in experiment_combination[1] or 'Occ' in experiment_combination[2]:
            model_names.append('tcow')
        if 'Intern' in experiment_combination[0] or 'Intern' in experiment_combination[1] or 'Intern' in experiment_combination[2]:
            model_names.append('Intern')

        model_names = sorted(model_names)
        model_name_1 = model_names[0]
        model_name_2 = model_names[1]
        model_name_3 = model_names[2]
        curr_results_path = 'results/Rosetta/ConceptIoU_Filter{}_{}_{}_{}_{}.json'.format(str(args.rosetta_iou_thresholds[0]).replace('.',''),model_name_1, model_name_2, model_name_3, results_path.split('/')[-1].split('_')[0])
        if os.path.exists(curr_results_path):
            print('Results already exist for this combination: {} at location: {}'.format(experiment_combination, curr_results_path))
            # load results
            if os.path.exists(curr_results_path):
                with open(curr_results_path, 'r') as f:
                    results_three = json.load(f)
        else:
            # compute and save results
            print('Computing results for this combination: {}'.format(experiment_combination))
            print('Saving results to: {}'.format(curr_results_path))
            results_three = compute_rosetta_concepts(list(experiment_combination), args, save_path = curr_results_path, all_to_use_concepts = all_concepts)
        print('Filtering results via rosetta IoU threshold {}...'.format(args.rosetta_iou_thresholds[1]))
        thresh_concept_results_three = {key: value for key, value in results_three.items() if value > args.rosetta_iou_thresholds[1]}

        # remove all concepts in either model that have no overlaps above the threshold
        for model_name in experiment_combination:

            # get all concepts for both model
            model_concepts_1 = []
            model_concepts_2 = []
            model_concepts_3 = []
            for concept_pair_name in thresh_concept_results_three.keys():
                model_1, concept_1, model_2, concept_2, model_3, concept_3 = concept_pair_name.split(' ')
                model_concept_1 = '{} {}'.format(model_1, concept_1)
                model_concept_2 = '{} {}'.format(model_2, concept_2)
                model_concept_3 = '{} {}'.format(model_3, concept_3)
                model_concepts_1.append(model_concept_1)
                model_concepts_2.append(model_concept_2)
                model_concepts_3.append(model_concept_3)

            # remove all concepts in either model that have no overlaps above the threshold
            concepts = all_concepts[model_name]
            for concept in concepts:
                model_concept = '{} {}'.format(model_name, concept)
                if model_concept not in model_concepts_1 and model_concept not in model_concepts_2 and model_concept not in model_concepts_3:
                    all_concepts[model_name].remove(concept)

    if len(args.exp_name) > 3:
        print('Filtering four tuples...')
        # Iterative IoU filtering
        experiment_combinations = n_tuple_combinations(args.exp_name, 4)
        for experiment_combination in experiment_combinations:
            # experiment_combination = experiment_combinations[3]
            # check if results exist for this combination
            model_names = []
            if 'videomae_ssv2_pre_keys' in experiment_combination[0] or 'videomae_ssv2_pre_keys' in experiment_combination[1] or 'videomae_ssv2_pre_keys' in experiment_combination[2] or 'videomae_ssv2_pre_keys' in experiment_combination[3]:
                model_names.append('ssl')
            if 'videomae_ssv2_keys' in experiment_combination[0] or 'videomae_ssv2_keys' in experiment_combination[1] or 'videomae_ssv2_keys' in experiment_combination[2] or 'videomae_ssv2_keys' in experiment_combination[3]:
                model_names.append('vidmae')
            if 'Occ' in experiment_combination[0] or 'Occ' in experiment_combination[1] or 'Occ' in experiment_combination[2] or 'Occ' in experiment_combination[3]:
                model_names.append('tcow')
            if 'Intern' in experiment_combination[0] or 'Intern' in experiment_combination[1] or 'Intern' in experiment_combination[2] or 'Intern' in experiment_combination[3]:
                model_names.append('Intern')

            model_names = sorted(model_names)
            model_name_1 = model_names[0]
            model_name_2 = model_names[1]
            model_name_3 = model_names[2]
            model_name_4 = model_names[3]
            curr_results_path = 'results/Rosetta/ConceptIoU_Filter{}_{}_{}_{}_{}_{}.json'.format(str(args.rosetta_iou_thresholds[1]).replace('.',''),model_name_1, model_name_2, model_name_3, model_name_4, results_path.split('/')[-1].split('_')[0])
            if os.path.exists(curr_results_path):
                print('Results already exist for this combination: {}'.format(experiment_combination))
                # load results
                if os.path.exists(curr_results_path):
                    with open(curr_results_path, 'r') as f:
                        results_four = json.load(f)
            else:
                # compute and save results
                print('Computing results for this combination: {}'.format(experiment_combination))
                print('Saving results to: {}'.format(curr_results_path))
                results_four = compute_rosetta_concepts(list(experiment_combination), args, save_path=curr_results_path, all_to_use_concepts=all_concepts)
            print('Filtering results via rosetta IoU threshold {}...'.format(args.rosetta_iou_thresholds[2]))
            thresh_concept_results_three = {key: value for key, value in results_four.items() if value > args.rosetta_iou_thresholds[2]}
            # remove all concepts in either model that have no overlaps above the threshold
            for model_name in experiment_combination:
                # get all concepts for both model
                model_concepts_1 = []
                model_concepts_2 = []
                model_concepts_3 = []
                model_concepts_4 = []
                for concept_pair_name in thresh_concept_results_three.keys():
                    model_1, concept_1, model_2, concept_2, model_3, concept_3, model_4, concept_4 = concept_pair_name.split(' ')
                    model_concept_1 = '{} {}'.format(model_1, concept_1)
                    model_concept_2 = '{} {}'.format(model_2, concept_2)
                    model_concept_3 = '{} {}'.format(model_3, concept_3)
                    model_concept_4 = '{} {}'.format(model_4, concept_4)
                    model_concepts_1.append(model_concept_1)
                    model_concepts_2.append(model_concept_2)
                    model_concepts_3.append(model_concept_3)
                    model_concepts_4.append(model_concept_4)

                # remove all concepts in either model that have no overlaps above the threshold
                concepts = all_concepts[model_name]
                for concept in concepts:
                    model_concept = '{} {}'.format(model_name, concept)
                    if model_concept not in model_concepts_1 and model_concept not in model_concepts_2 and model_concept not in model_concepts_3 and model_concept not in model_concepts_4:
                        all_concepts[model_name].remove(concept)

def compute_rosetta_concepts(exp_names, args, save_path, all_to_use_concepts=None):
    # compute miou between concept mask and other concepts in other models
    rosetta_concepts = {}

    masks = {}
    concepts = {}
    # load masks for all experiments
    for exp_idx, exp_name in enumerate(exp_names):
        vcd_path = os.path.join('results', exp_name, 'vcd', 'vcd.pkl')
        print('Loading VCD from {}...'.format(vcd_path))
        with open(vcd_path, 'rb') as f:
            vcd = pickle.load(f)
        max_num_videos = vcd.args.max_num_videos
        if all_to_use_concepts is None:
            all_concepts = get_all_concepts(vcd, args)
        else:
            all_concepts = all_to_use_concepts[exp_name]
        if args.filter_concepts_by_importance:  # todo: don't open up masks if not necessary
            path_to_importance = os.path.join('results', exp_name, args.importance_files[exp_idx])
            print('Loading importance scores from {}...'.format(path_to_importance))
            with open(path_to_importance, 'rb') as f:
                importance = pickle.load(f)
            num_concepts_keep = int(args.frac_important_concepts_keep * len(importance['concept_importance_most_to_least']))
            # keep only the most important concepts
            all_concepts = importance['concept_importance_most_to_least'][:num_concepts_keep]
            print('Keeping {} most important concepts...'.format(num_concepts_keep))
        all_concept_masks = get_all_concept_masks(all_concepts, vcd, args, return_masks=args.preload_masks)
        # filter concepts by importance


        masks[exp_name] = all_concept_masks
        # add exp_name to the beginning of each concept
        all_concepts = [exp_name + ' ' + concept for concept in all_concepts]
        concepts[exp_name] = all_concepts
    del vcd

    # construct list of every single combination of concepts
    all_concepts = create_all_combinations([concepts[exp_name] for exp_name in exp_names])
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
    print('Saving rosetta_concepts to {}...'.format(save_path))
    with open(save_path, 'w') as f:
        json.dump(rosetta_concepts, f, indent=4)

    return rosetta_concepts

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

def vcd_args():

    parser = argparse.ArgumentParser()

    # general
    # parser.add_argument('--exp_name', nargs='+', default=['videomae_ssv2_keys_rsoafs_OneMask'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--exp_name', nargs='+', default=['videomae_ssv2_pre_keys_rsoafs_v2', 'videomae_ssv2_keys_rsoafs_v2', 'Occ_Keys_OG_rsoafs', 'Intern_Keys_SSv2_rsoafs'], type=str,help='experiment name (used for saving)')
    # parser.add_argument('--exp_name', nargs='+', default=['videomae_ssv2_pre_keys_dsbs', 'videomae_ssv2_keys_dsbs', 'Occ_Keys_OG_dsbs', 'Intern_Keys_SSv2_dsbs'], type=str,help='experiment name (used for saving)')
    # parser.add_argument('--exp_name', nargs='+', default=['videomae_ssv2_pre_keys_hsws', 'videomae_ssv2_keys_hsws', 'Occ_Keys_OG_hsws', 'Intern_Keys_SSv2_hsws'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--filter_concepts_by_importance', action='store_true', help='Filter Concepts by importance')
    # parser.add_argument('--importance_files', nargs='+', default=['ConceptHeadImporance_HW_4000MasksRatio095.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--importance_files', nargs='+', default=['ConceptHeadImporance_HW_4000Masks.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl', 'ConceptHeadImporance_HW_8000Masks.pkl', 'ConceptHeadImporance_HW_4000Masks.pkl'], type=str,help='experiment name (used for saving)')
    parser.add_argument('--frac_important_concepts_keep', default=0.075, type=float, help='Fraction of top concepts to keep.')
    parser.add_argument('--rosetta_iou_thresholds', nargs='+', default=[0.1, 0.1, 0.1], type=float, help='')

    parser.add_argument('--fig_save_name', default='attribution_plot',help='figure name (used for saving)')
    parser.add_argument('--compare_multiclass', action='store_true', help='if true, average over multiple classes and then compare ')
    parser.add_argument('--overwrite_importance', action='store_true', help='Overwrite importance results')
    parser.add_argument('--overwrite_attribution', action='store_true', help='Overwrite addtribution results')
    # parser.add_argument('--results_name', default='test4', type=str,help='figure name (used for saving)')
    # parser.add_argument('--results_name', default='rsoafs_TopK0075_FourModels', type=str,help='figure name (used for saving)')
    # parser.add_argument('--results_name', default='rsoafs_TopK005_FourModels', type=str,help='figure name (used for saving)')
    parser.add_argument('--results_name', default='rsoafs_IterIoU_FourModels', type=str,help='figure name (used for saving)')
    # parser.add_argument('--results_name', default='dsbs_BB5_FourModels', type=str,help='figure name (used for saving)')
    # parser.add_argument('--results_name', default='dsbs_TopK0075_FourModels', type=str,help='figure name (used for saving)')
    # parser.add_argument('--results_name', default='testy', type=str,help='figure name (used for saving)')
    # parser.add_argument('--results_name', default='ModelPair_05_ssl_vidmae_rsoafs', type=str,help='figure name (used for saving)')

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
    parser.add_argument('--inference', action='store_true', help='Debug using only 2 videos for all functions.')

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
