import os
import sys
import pickle
import json
import argparse
import numpy as np
import random
import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
import itertools
import math
import psutil
import gc
import torchvision

def main(args):
    # load vcd's from the list of models (that use the same dataset)
    rosetta_dir = os.path.join('results', 'Rosetta')
    if not os.path.exists(rosetta_dir):
        os.makedirs(rosetta_dir, exist_ok=True)
    results_path = os.path.join(rosetta_dir, args.results_name + '_rosetta_concepts.json')

    print('Loading all concepts...')
    # load all concepts for each experiment
    all_concepts = {}
    for exp_name in args.exp_name:
        vcd_path = os.path.join('results', exp_name, 'vcd', 'vcd.pkl')
        print('Loading VCD from {}...'.format(vcd_path))
        with open(vcd_path, 'rb') as f:
            vcd = pickle.load(f)
        all_concepts[exp_name] = get_all_concepts(vcd)

        # load concept importance and filter by most important concepts
        path_to_importance = os.path.join('results', exp_name, args.importance_file_name)
        if not os.path.exists(path_to_importance):
            print('Importance scores do not exist for experiment: {}'.format(exp_name))
            print('Computing Rosetta concepts without importance scores. WARNING: This will take up a lot of memory...')
            continue
        else:
            print('Loading importance scores from {}...'.format(path_to_importance))
            with open(path_to_importance, 'rb') as f:
                importance = pickle.load(f)
            num_concepts_keep = int(args.frac_important_concepts_keep * len(importance['concept_importance_most_to_least']))
            # keep only the most important concepts
            all_concepts[exp_name]  = importance['concept_importance_most_to_least'][:num_concepts_keep]
            print('Keeping {}/{} most important concepts...'.format(num_concepts_keep, len(importance['concept_importance_most_to_least'])))

    if args.save_rosetta_videos:
        # define nearest and smooth resize
        single_post_resize_nearest = torchvision.transforms.Resize(
            (224, 224),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        single_post_resize_smooth = torchvision.transforms.Resize(
            (224, 224),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)


    # Iterative IoU filtering
    print('###############################################')
    print('Filtering Rosetta two tuples...')
    print('###############################################')
    experiment_combinations = n_tuple_combinations(args.exp_name, 2)
    for experiment_combination in experiment_combinations:
        # check if results exist for this combination
        model_names = []
        if 'Pretrained' in experiment_combination[0] or 'Pretrained' in experiment_combination[1]:
            model_names.append('VideoMAE_Pretrained')
        if 'VideoMAE_FT' in experiment_combination[0] or 'VideoMAE_FT' in experiment_combination[1]:
            model_names.append('VideoMAE_FT')
        if 'Occ' in experiment_combination[0] or 'Occ' in experiment_combination[1]:
            model_names.append('TCOW')
        if 'Intern' in experiment_combination[0] or 'Intern' in experiment_combination[1]:
            model_names.append('Intern')
        model_names = sorted(model_names)
        model_name_1 = model_names[0]
        model_name_2 = model_names[1]
        curr_results_path = 'results/Rosetta/ConceptIoUImportance-{}_{}_{}_{}.json'.format(str(args.frac_important_concepts_keep).replace('.', ''), model_name_1, model_name_2, results_path.split('/')[-1].split('_')[0])
        if os.path.exists(curr_results_path):
            print('Results already exist for this combination: {}'.format(experiment_combination))
            # load results
            with open(curr_results_path, 'r') as f:
                results_two = json.load(f)
        else:
            # If not, compute and save results
            print('Computing results for this combination: {}'.format(experiment_combination))
            print('Saving results to: {}'.format(curr_results_path))
            results_two = compute_rosetta_concepts(list(experiment_combination), args, save_path = curr_results_path, all_to_use_concepts = all_concepts)
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

        # no repeats for two tuples
        print('Loading and removing repeats...')
        sorted_rosetta_concepts = sorted(results_two.items(), key=lambda x: x[1], reverse=True)

        # recreate list ranking but without repeating concepts
        global_concept_list1 = []
        global_concept_list2 = []
        sorted_rosetta_concepts_no_repeats = {}
        # use tqdm to show progress bar
        for idx, concept in enumerate(tqdm(sorted_rosetta_concepts)):
            concept_name = concept[0]
            model_concept_names = concept_name.split(' ')
            concept_1 = model_concept_names[0] + ' ' + model_concept_names[1]
            concept_2 = model_concept_names[2] + ' ' + model_concept_names[3]

            if idx == 0:
                sorted_rosetta_concepts_no_repeats[idx] = concept
            else:
                if concept_1 in global_concept_list1 or concept_2 in global_concept_list2:
                    continue
                else:
                    sorted_rosetta_concepts_no_repeats[idx] = concept
            if concept_1 not in global_concept_list1:
                global_concept_list1.append(concept_1)
            if concept_2 not in global_concept_list2:
                global_concept_list2.append(concept_2)

        # save results
        results_path_no_repeat = curr_results_path.replace('.json', '_no_repeats.json')
        print('Saving results to {}...'.format(results_path_no_repeat))
        with open(results_path_no_repeat, 'w') as f:
            json.dump(sorted_rosetta_concepts_no_repeats, f)
        sorted_rosetta_concepts = sorted(sorted_rosetta_concepts_no_repeats.items(), key=lambda x: x[1][1], reverse=True)
        # only save concepts with iou > 0.1
        sorted_rosetta_concepts_filter = [x[1] for x in sorted_rosetta_concepts if x[1][1] > 0.1]
        # save videos to visualize Rosetta concepts
        if args.save_rosetta_videos:
            print('Saving 2-tuple videos...')
            save_rosetta_concepts_videos(curr_results_path, sorted_rosetta_concepts_filter,
                                         single_post_resize_nearest, single_post_resize_smooth, args.num_vids_per_model)


    if len(args.exp_name) > 2:
        print('###############################################')
        print('Filtering Rosetta three tuples...')
        print('###############################################')
        experiment_combinations = n_tuple_combinations(args.exp_name, 3)
        for experiment_combination in reversed(experiment_combinations):
            model_names = []
            if 'Pretrained' in experiment_combination[0] or 'Pretrained' in experiment_combination[1] or 'Pretrained' in experiment_combination[2]:
                model_names.append('VideoMAE_Pretrained')
            if 'VideoMAE_FT' in experiment_combination[0] or 'VideoMAE_FT' in experiment_combination[1] or 'VideoMAE_FT' in experiment_combination[2]:
                model_names.append('VideoMAE_FT')
            if 'Occ' in experiment_combination[0] or 'Occ' in experiment_combination[1] or 'Occ' in experiment_combination[2]:
                model_names.append('TCOW')
            if 'Intern' in experiment_combination[0] or 'Intern' in experiment_combination[1] or 'Intern' in experiment_combination[2]:
                model_names.append('Intern')

            model_names = sorted(model_names)
            model_name_1 = model_names[0]
            model_name_2 = model_names[1]
            model_name_3 = model_names[2]
            curr_results_path = 'results/Rosetta/ConceptIoUImportance_Filter{}-{}_{}_{}_{}_{}.json'.format(str(args.frac_important_concepts_keep).replace('.', ''),str(args.rosetta_iou_thresholds[0]).replace('.',''),model_name_1, model_name_2, model_name_3, results_path.split('/')[-1].split('_')[0])
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

            # no repeats for three tuples
            print('Loading and removing repeats...')
            sorted_rosetta_concepts = sorted(results_three.items(), key=lambda x: x[1], reverse=True)

            # recreate list ranking but without repeating concepts
            global_concept_list1 = []
            global_concept_list2 = []
            global_concept_list3 = []
            sorted_rosetta_concepts_no_repeats = {}
            # use tqdm to show progress bar
            for idx, concept in enumerate(tqdm(sorted_rosetta_concepts)):
                concept_name = concept[0]
                model_concept_names = concept_name.split(' ')
                concept_1 = model_concept_names[0] + ' ' + model_concept_names[1]
                concept_2 = model_concept_names[2] + ' ' + model_concept_names[3]
                concept_3 = model_concept_names[4] + ' ' + model_concept_names[5]

                if idx == 0:
                    sorted_rosetta_concepts_no_repeats[idx] = concept
                else:
                    if concept_1 in global_concept_list1 or concept_2 in global_concept_list2 or concept_3 in global_concept_list3:
                        continue
                    else:
                        sorted_rosetta_concepts_no_repeats[idx] = concept
                if concept_1 not in global_concept_list1:
                    global_concept_list1.append(concept_1)
                if concept_2 not in global_concept_list2:
                    global_concept_list2.append(concept_2)
                if concept_3 not in global_concept_list3:
                    global_concept_list3.append(concept_3)

            # save results
            results_path_no_repeat = curr_results_path.replace('.json', '_no_repeats.json')
            print('Saving results to {}...'.format(results_path_no_repeat))
            with open(results_path_no_repeat, 'w') as f:
                json.dump(sorted_rosetta_concepts_no_repeats, f)

            sorted_rosetta_concepts = sorted(sorted_rosetta_concepts_no_repeats.items(), key=lambda x: x[1][1], reverse=True)
            # only save concepts with iou > 0.1
            sorted_rosetta_concepts_filter = [x[1] for x in sorted_rosetta_concepts if x[1][1] > 0.1]

            # save videos to visualize Rosetta concepts
            if args.save_rosetta_videos:
                print('Saving 3-tuple videos...')
                save_rosetta_concepts_videos(curr_results_path, sorted_rosetta_concepts_filter,
                                             single_post_resize_nearest, single_post_resize_smooth,
                                             args.num_vids_per_model)


    if len(args.exp_name) > 3:
        print('###############################################')
        print('Filtering Rosetta four tuples...')
        print('###############################################')        # Iterative IoU filtering
        experiment_combinations = n_tuple_combinations(args.exp_name, 4)
        for experiment_combination in experiment_combinations:
            # experiment_combination = experiment_combinations[3]
            # check if results exist for this combination
            model_names = []
            if 'Pretrained' in experiment_combination[0] or 'Pretrained' in experiment_combination[1] or 'Pretrained' in experiment_combination[2] or 'Pretrained' in experiment_combination[3]:
                model_names.append('VideoMAE_Pretrained')
            if 'VideoMAE_FT' in experiment_combination[0] or 'VideoMAE_FT' in experiment_combination[1] or 'VideoMAE_FT' in experiment_combination[2] or 'VideoMAE_FT' in experiment_combination[3]:
                model_names.append('VideoMAE_FT')
            if 'Occ' in experiment_combination[0] or 'Occ' in experiment_combination[1] or 'Occ' in experiment_combination[2] or 'Occ' in experiment_combination[3]:
                model_names.append('TCOW')
            if 'Intern' in experiment_combination[0] or 'Intern' in experiment_combination[1] or 'Intern' in experiment_combination[2] or 'Intern' in experiment_combination[3]:
                model_names.append('Intern')

            model_names = sorted(model_names)
            model_name_1 = model_names[0]
            model_name_2 = model_names[1]
            model_name_3 = model_names[2]
            model_name_4 = model_names[3]
            curr_results_path = 'results/Rosetta/ConceptIoUImportance_Filter{}-{}-{}_{}_{}_{}_{}_{}.json'.format(str(args.frac_important_concepts_keep).replace('.', ''),str(args.rosetta_iou_thresholds[0]).replace('.',''),str(args.rosetta_iou_thresholds[1]).replace('.',''),model_name_1, model_name_2, model_name_3, model_name_4, results_path.split('/')[-1].split('_')[0])
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

        # open results
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

        sorted_rosetta_concepts = sorted(sorted_rosetta_concepts_no_repeats.items(), key=lambda x: x[1][1], reverse=True)
        # Only save videos above certain threshold
        sorted_rosetta_concepts_filter = [x[1] for x in sorted_rosetta_concepts if x[1][1] > 0.1]
        if args.save_rosetta_videos:
            print('Saving 4-tuple videos...')
            save_rosetta_concepts_videos(curr_results_path, sorted_rosetta_concepts_filter,
                                         single_post_resize_nearest, single_post_resize_smooth, args.num_vids_per_model)

def compute_rosetta_concepts(exp_names, args, save_path, all_to_use_concepts=None):
    # compute miou between concept mask and other concepts in other models
    rosetta_concepts = {}

    masks = {}
    concepts = {}
    # load masks for all experiments
    for exp_idx, exp_name in enumerate(exp_names):
        vcd_path = os.path.join('results', exp_name, 'vcd', 'vcd.pkl')
        with open(vcd_path, 'rb') as f:
            vcd = pickle.load(f)
        max_num_videos = vcd.args.max_num_videos
        if all_to_use_concepts is None:
            all_concepts = get_all_concepts(vcd)
        else:
            all_concepts = all_to_use_concepts[exp_name]
        all_concept_masks = get_all_concept_masks(all_concepts, vcd, args, return_masks=args.preload_masks)

        masks[exp_name] = all_concept_masks
        # add exp_name to the beginning of each concept
        all_concepts = [exp_name + ' ' + concept for concept in all_concepts]
        concepts[exp_name] = all_concepts
    del vcd

    # construct list of every single combination of concepts
    print('Creating all {}-tuple combinations of concepts. May take a while...'.format(len(exp_names)))
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
                video_masks = video_masks.bool()
            else:
                # if no masks, construct a dummy mask of all 0's
                video_masks = torch.zeros((30, 15, 20)).bool()
            all_video_masks.append(video_masks)

    # calculate iou for all masks
    iou = compute_iou_list(all_video_masks)
    return iou

def save_rosetta_concepts_videos(save_path, sorted_rosetta_concepts_filter,single_post_resize_nearest,single_post_resize_smooth, num_vids_per_side=3):
    '''

    :param sorted_rosetta_concepts_filter: list of lists of rosetta concepts sorted by iou, each list is a different model
    :return: saves 4x4 concept videos with white-alpha blending in a row, where each column is a different model/concept
    '''
    extensions = ['.mp4']

    # create directory to save videos
    save_dir = os.path.join('results', 'Rosetta', 'concept_videos')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, save_path.split('/')[-1].replace('.json', ''))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    # iterate through all rosetta concepts
    for rosetta_concept_idx, concept in enumerate(tqdm(sorted_rosetta_concepts_filter)):
        models = concept[0].split(' ')[::2]
        concepts = concept[0].split(' ')[1::2]
        r_score = concept[1]
        # create canvas that is num_vids_per_side videos tall and num_vids_per_side*len(models) videos wide, with 10 pixels between each video grid
        global_canvas = np.zeros((16, 224 * num_vids_per_side, 224 * num_vids_per_side * len(models) + 10 * (len(models) - 1), 3))

        lay_head_concept = []
        # iterate through all models
        for model_idx, exp_name in enumerate(models):

            layer, head, concept = concepts[model_idx].split('-')
            layer = int(layer.replace('Layer', ''))
            head = int(head.replace('Head', ''))
            concept = concept.replace(' ', '_')
            lay_head_concept.append('{}-{}-{}'.format(layer, head, concept.split('_')[-1]))

            # This is slow but keeping all VCDs in memory can be too much...
            vcd_path = os.path.join('results', exp_name, 'vcd', 'vcd.pkl')
            with open(vcd_path, 'rb') as f:
                vcd = pickle.load(f)

            if exp_name == 'Occ_Keys_OG_dsbs':
                vcd.cached_file_path = vcd.cached_file_path.replace('v1', 'Rosetta')
            with open(vcd.cached_file_path, 'rb') as f:
                vcd.dataset = pickle.load(f)

            # get all concept masks for this concept
            concept_rgb_videos = vcd.dataset[vcd.dic[layer][head][concept]['video_numbers']]
            concept_rgb_videos = torch.stack([single_post_resize_smooth(video) for video in concept_rgb_videos])
            vid_nums = range(len(vcd.dic[layer][head][concept]['video_numbers']))
            concept_masks = vcd.dic[layer][head][concept]['video_mask']

            # define single model canvas of 4x4 videos
            model_canvas = np.zeros((16, 224 * num_vids_per_side, 224 * num_vids_per_side, 3))

            # save each video in concept
            for num_segment, i in enumerate(vid_nums):
                if num_segment == num_vids_per_side ** 2:
                    break
                # get rgb video
                rgb_video = concept_rgb_videos[i].permute(1,2,3,0)
                mask_path = concept_masks[i]
                if isinstance(mask_path, list):
                    # combine all masks into one
                    mask = torch.tensor(np.stack([np.load(path) for path in mask_path], axis=0).sum(axis=0))
                else:
                    mask = torch.tensor(np.load(mask_path))
                try:
                    concept_mask = single_post_resize_nearest(mask)
                except:
                    concept_mask = single_post_resize_nearest(mask.unsqueeze(0))
                mask = np.array(np.repeat(concept_mask.unsqueeze(0), 3, axis=0).permute(1, 2, 3, 0))  # repeat along channels
                if mask.shape[0] != rgb_video.shape[0]:
                    mask = np.concatenate([mask[::2], mask[-1:]], axis=0)
                    concept_mask = torch.cat([concept_mask[::2], concept_mask[-1:]], dim=0)
                vis_concept_assign = vcd.create_concept_mask_video(rgb_video, mask, alpha=0.5, blend_white=True)
                concept_mask = np.array(concept_mask.float())
                mask_border = vcd.draw_segm_borders(concept_mask[..., None], fill_white=False)
                vis_concept_assign = vcd.create_model_input_video(vis_concept_assign, concept_mask,
                                                                   mask_border, extra_frames=0,
                                                                   target=True, color='orange')

                # grab every other frame and concat last frame if too long
                if vis_concept_assign.shape[0] > 16:
                    vis_concept_assign = np.concatenate([vis_concept_assign[::2], vis_concept_assign[-1:]], axis=0)
                model_canvas[:,num_segment // num_vids_per_side * 224: (num_segment // num_vids_per_side + 1) * 224,
                (num_segment % num_vids_per_side) * 224: (num_segment % num_vids_per_side + 1) * 224] = vis_concept_assign

            # add model canvas to global canvas
            global_canvas[:, :, model_idx * 224 * num_vids_per_side + model_idx * 10: (model_idx + 1) * 224 * num_vids_per_side + model_idx * 10] = model_canvas

        print()
        # save global canvas
        if len(models) == 2:
            file_name = os.path.join(save_dir, 'RIdx{}_Score{}_{}_{}'.format(rosetta_concept_idx, int(r_score*100), lay_head_concept[0], lay_head_concept[1]))
        elif len(models) == 3:
            file_name = os.path.join(save_dir, 'RIdx{}_Score{}_{}_{}_{}'.format(rosetta_concept_idx, int(r_score*100), lay_head_concept[0], lay_head_concept[1], lay_head_concept[2]))
        elif len(models) == 4:
            file_name = os.path.join(save_dir, 'RIdx{}_Score{}_{}_{}_{}_{}'.format(rosetta_concept_idx, int(r_score*100), lay_head_concept[0], lay_head_concept[1], lay_head_concept[2], lay_head_concept[3]))
        else:
            raise ValueError('Only supports 2, 3, or 4 models')
        vcd.save_video(frames=global_canvas,
                        file_name=file_name,
                        extensions=extensions,
                        fps=6,
                        upscale_factor=1)

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
            if psutil.virtual_memory().percent > 95:
                print('Memory usage too high, clearing memory...')
                gc.collect()
                torch.cuda.empty_cache()
                exit()

    return result


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
    for concept in tqdm(all_concepts):
        layer, head, concept = concept.split('-')
        layer = int(layer.replace('Layer', ''))
        head = int(head.replace('Head', ''))

        for video_idx in range(vcd.args.max_num_videos):
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

def get_all_concepts(vcd, memory_limit=90):
    # get all concepts in a huge list
    all_concepts = []
    # each mask is a tensor of shape (num_frames, h, w) where 1's are the pixels that are part of the concept
    for layer in tqdm(vcd.dic.keys()):
        for head in vcd.dic[layer].keys():
            for concept in vcd.dic[layer][head]['concepts']:
                all_concepts.append('Layer{}-Head{}-{}'.format(layer, head,concept))
                if psutil.virtual_memory().percent > memory_limit:
                    print('Memory usage too high, clearing memory and exiting...')
                    gc.collect()
                    torch.cuda.empty_cache()
                    exit()
    return all_concepts

def vcd_args():

    parser = argparse.ArgumentParser()

    # Experiment names and file names
    parser.add_argument('--exp_name', nargs='+',
                        default=['Intern_SSv2_Rolling', 'Occ_SSv2_Rolling', 'VideoMAE_FT_SSv2_Rolling', 'VideoMAE_Pretrained_SSv2_Rolling'],
                        type=str,help='Experiment names from VTCD to use for Rosetta concepts')
    parser.add_argument('--importance_file_name', default='ConceptImportance_4000Masks.pkl', type=str,
                        help='Name of importance file used during CRIS.')

    # Hyperparameters for filtering Rosetta concepts
    parser.add_argument('--frac_important_concepts_keep', default=0.075, type=float,
                        help='Fraction of top important concepts to keep.')
    parser.add_argument('--rosetta_iou_thresholds', nargs='+', default=[0.2, 0.2, 0.2], type=float,
                        help='The minimum IoU threshold for filtering Rosetta concepts at each n-tuple stage.')
    parser.add_argument('--results_name', default='Rolling_Rosetta_4Models', type=str,help='Results file name to save output.')

    # Data loading and saving
    parser.add_argument('--preload_masks', action='store_false',
                        help='If true, preloads all masks into memory before IoU computation.')
    parser.add_argument('--save_rosetta_videos', action='store_true', help='Save visualization of Rosetta concepts as mp4.')
    parser.add_argument('--num_vids_per_model', default=3, type=int, help='Number of videos to save per model in Rosetta visualization.')

    # reproducibility
    parser.add_argument('--seed', default=0, type=int,help='seed')

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
