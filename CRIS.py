
import os
import sys
import time

import pickle
import csv
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
import seaborn as sns
from einops import rearrange
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torchvision

from models.hide_seek.tcow.eval.metrics import calculate_metrics_mask_track
import models.hide_seek.tcow as tcow
from utilities.utils import load_model

def main(args):
    results_path = 'results/{}/ConceptImportance_{}Masks{}.pkl'.format(args.exp_name, args.num_masks, args.results_name)
    print('Results will be saved to {}'.format(results_path))
    if args.use_saved_results and not args.recompute_performance_curves:
        if os.path.exists(results_path):
            print('Loading results from {}'.format(results_path))
            with open(results_path, 'rb') as f:
                all_results = pickle.load(f)

            # quantify what layers the most important concepts are in
            layer_scores = {layer: [] for layer in range(12)}
            for idx, concept in enumerate(all_results['concept_importance_most_to_least']):
                layer = int(concept.split('-')[0].split('Layer')[-1])
                score = (len(all_results['concept_importance_most_to_least']) - idx) / len(all_results['concept_importance_most_to_least'])
                layer_scores[layer].append(score)

            # average scores
            for layer in layer_scores.keys():
                layer_scores[layer] = np.mean(layer_scores[layer])

            # sort layers by score
            sorted_layer_scores = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
            print('Most important layers:')
            for layer, score in sorted_layer_scores:
                print('Layer {}: {:.3f}'.format(layer, score))

            # plot results
            results = {
                'most_to_least': all_results['most_to_least'],
                'least_to_most': all_results['least_to_most'],
                'random': all_results['random'],
            }
            plot_save_results(args, results)
            exit()
        else:
            print('No results found at {}'.format(results_path))


    # load vcd with pickle
    vcd_path = os.path.join('results', args.exp_name, 'vcd', 'vcd.pkl')
    print('Loading VCD from {}...'.format(vcd_path))
    vcd = pickle.load(open(vcd_path, 'rb'))

    # load args from vcd
    args.model = vcd.args.model
    args.dataset = vcd.args.dataset
    args.cluster_layer = vcd.args.cluster_layer
    args.cluster_subject = vcd.args.cluster_subject
    args.attn_head = vcd.args.attn_head
    args.concept_clustering = vcd.args.concept_clustering
    args.use_temporal_attn = vcd.args.use_temporal_attn

    if 'pre' in vcd.args.model:
        import torch.nn as nn
        mse_loss_func = nn.MSELoss()
        mask = torch.zeros((1, 1568)).cuda().type(torch.bool)

    # load dataset
    path_to_dataset = vcd.cached_file_path.replace(' ', '_')
    # open pkl file
    print('Loading dataset from {}...'.format(path_to_dataset))
    try:
        with open(path_to_dataset, 'rb') as f:
            dataset = pickle.load(f)
    except:
        print('Failed to load dataset from {}'.format(path_to_dataset))
        if 'timesformer' in vcd.args.model:
            path_to_dataset2 = vcd.cached_file_path.replace(' ', '_').replace('v1', 'Rosetta')
            print('Trying to load dataset from {}...'.format(path_to_dataset2))
            with open(path_to_dataset2, 'rb') as f:
                dataset = pickle.load(f)

    if 'ssv2' in path_to_dataset:
        ssv2_label_path = os.path.join(vcd.args.ssv2_path, 'something-something-v2-labels.json')
        with open(ssv2_label_path, 'r') as f:
            ssv2_labels = json.load(f)
        target_class_name = path_to_dataset.split('_Max')[0].replace('_', ' ').split('/')[-1]
        target_class_idx = int(ssv2_labels[target_class_name])
    else:
        target_class_idx = None

    num_videos = dataset['pv_rgb_tf'].shape[0] if 'kub' in path_to_dataset else dataset.shape[0]

    # get all concepts in a list
    all_concepts = []
    all_concept_masks = {}
    print('Loading masks...')
    for layer in tqdm(vcd.dic.keys()):
        for head in vcd.dic[layer].keys():
            for concept in vcd.dic[layer][head]['concepts']:
                all_concepts.append('Layer{}-Head{}-{}'.format(layer, head,concept))
                for video_idx in range(num_videos):
                    mask_paths = [mask for mask in vcd.dic[layer][head][concept]['video_mask'] if'video_{}'.format(video_idx) in mask]
                    if len(mask_paths) > 0:
                        masks = np.stack([np.load(mask_path) for mask_path in mask_paths])
                        masks = 1 - (np.sum(masks, axis=0) > 0)
                        masks = torch.tensor(masks)

                        if 'timesformer' in args.model:
                            masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8),size=(30, 15, 20), mode='nearest').squeeze(0).squeeze(0).float()
                        else:
                            masks = F.interpolate(masks.unsqueeze(0).unsqueeze(0).type(torch.uint8), size=(8, 14, 14), mode='nearest').squeeze(0).squeeze(0).float()

                        if video_idx not in all_concept_masks.keys():
                            all_concept_masks[video_idx] = {}
                        all_concept_masks[video_idx]['Layer{}-Head{}-{}'.format(layer, head,concept)] = masks
    all_concepts = list(enumerate(all_concepts))
    number_total_concepts = len(all_concepts)
    num_concepts_to_mask = int(number_total_concepts * args.masking_ratio)

    if not args.recompute_performance_curves:
        with torch.no_grad():
            concept_importances = []
            baseline_performances = []
            baseline_masks = []
            print('Computing importance of heads with {} masks'.format(args.num_masks))
            for video_idx in tqdm(range(num_videos)):
                # load model
                model = load_model(args)
                if 'timesformer' in args.model:
                    if 'kub' in path_to_dataset:
                        output_mask, output_flags, target_mask, features, model_retval = tcow_timesformer_forward(dataset, model, video_idx, keep_all=True)
                        model_retval = {
                            'output_mask': output_mask.unsqueeze(0),
                            'target_mask': target_mask.unsqueeze(0)}
                        metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,source_name='kubric')
                        # put results from cuda to cpu
                        metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                        baseline_performance = metrics_retval['mean_snitch_iou'].item()
                        baseline_performances.append(baseline_performance)
                    elif 'ssv2' in path_to_dataset:
                        # if ssv2, we don't have labels, so just assume the baseline is 1.0 (i.e., the prediction is always correct)
                        video = vcd.post_resize_smooth(dataset[video_idx]).unsqueeze(0).cuda()
                        seeker_query_mask = vcd.post_resize_nearest(
                            vcd.seeker_query_labels[video_idx].squeeze(0)).unsqueeze(0).cuda()
                        (output_mask, output_flags, features) = model(video, seeker_query_mask)
                        baseline_mask = (output_mask>0.5).cpu().detach()
                        baseline_masks.append(baseline_mask)
                        baseline_performances.append(1.0)
                elif 'vidmae' in args.model or 'mme' in args.model:
                    pre_video = dataset[video_idx].permute(1, 0, 2, 3)
                    if pre_video.shape[-1] != 224:
                        pre_video = vcd.post_resize_smooth(dataset[video_idx].permute(1, 0, 2, 3))
                    # normalize video with imagenet stats
                    video = torch.stack([torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(vid) for vid in pre_video],
                                        dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    if 'pre' in args.model:
                        # get target, which is the normalized video
                        unnorm_videos = pre_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                                                   p0=2, p1=16, p2=16, t=8, h=14, w=14)
                        videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                                       ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)').cuda()
                        # forward pass
                        pred, _ = model(video, mask, return_features=False)
                        loss = mse_loss_func(input=pred, target=videos_patch)
                        baseline_performances.append(-loss.item())
                    else:
                        pred, _ = model(video)
                        pred = pred.argmax(dim=1).item()
                        baseline_performances.append(1) if pred == target_class_idx else baseline_performances.append(0)
                elif 'intern' in args.model:
                    # video
                    video = dataset[video_idx]
                    if video.shape[-1] != 224:
                        video = vcd.post_resize_smooth(video)
                    video = model.transform(video).unsqueeze(0).cuda()
                    video_features, _ = model.encode_video(video)
                    # text
                    text = model.tokenize(vcd.args.target_class).cuda()
                    text_features = model.encode_text(text)
                    # norm
                    text_features = torch.nn.functional.normalize(text_features, dim=1)
                    video_features = torch.nn.functional.normalize(video_features, dim=1)
                    t = model.logit_scale.exp()
                    logit = (video_features @ text_features.T * t)

                    baseline_performances.append(logit[0].item())
                else:
                    print('Model not recognized - need to implement the forward pass/metric here!')
                    raise NotImplementedError

                cris_concepts_removed = []
                performance_list = []

                # CRIS: mask many concepts and compute performance after each mask
                for i in range(args.num_masks):
                    # sample concepts to mask with CRIS
                    concepts_to_remove_and_idx = random.sample(all_concepts, num_concepts_to_mask)
                    concepts_to_remove = [x[1] for x in concepts_to_remove_and_idx]
                    concepts_to_remove_idx = torch.tensor([x[0] for x in concepts_to_remove_and_idx])
                    all_layer_hook_dict = {layer : {'heads': [], 'masks': [], 'cluster_subject': vcd.args.cluster_subject} for layer in vcd.dic.keys()}
                    for concept_idx in concepts_to_remove:
                        # get all masks that belong to concept for this video
                        layer = int(concept_idx.split('-')[0].split('Layer')[-1])
                        head = int(concept_idx.split('-')[1].split('Head')[-1])
                        # put in layer dict for hook
                        if not concept_idx in all_concept_masks[video_idx].keys():
                            continue
                        all_layer_hook_dict[layer]['heads'].append(head)
                        all_layer_hook_dict[layer]['masks'].append(all_concept_masks[video_idx][concept_idx])
                    # load model with hooks
                    model = load_model(args, hook=remove_concepts, hook_layer=list(vcd.dic.keys()), model=model, hook_dict=all_layer_hook_dict)
                    if 'timesformer' in args.model:
                        if 'kub' in path_to_dataset:
                            output_mask, output_flags, target_mask, features, model_retval = \
                                tcow_timesformer_forward(dataset,model,video_idx,keep_all=False)
                            model_retval = {
                                'output_mask': output_mask.unsqueeze(0),
                                'target_mask': target_mask.unsqueeze(0)}
                            metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                          source_name='kubric')
                            # put results from cuda to cpu
                            metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                            new_performance = metrics_retval['mean_snitch_iou'].item()
                        elif 'ssv2' in path_to_dataset:
                            video = vcd.post_resize_smooth(dataset[video_idx]).unsqueeze(0).cuda()
                            seeker_query_mask = vcd.post_resize_nearest(vcd.seeker_query_labels[video_idx].squeeze(0)).unsqueeze(0).cuda()
                            (output_mask, output_flags, features) = model(video, seeker_query_mask)
                            pred = (output_mask > 0.5).cpu().detach()
                            # compare pred with baseline mask
                            new_performance = compute_iou(pred, baseline_mask)
                    elif 'vidmae' in args.model:
                        if 'pre' in args.model:
                            pred, _ = model(video, mask, return_features=False)
                            loss = mse_loss_func(input=pred, target=videos_patch)
                            new_performance = -loss.item()
                        else:
                            pred, _ = model(video)
                            pred = pred.argmax(dim=1).item()
                            new_performance = 1.0 if pred == target_class_idx else 0.0
                    elif 'intern' in args.model:
                        # video
                        video = dataset[video_idx]
                        video = model.transform(video).unsqueeze(0).cuda()
                        video_features, _ = model.encode_video(video)
                        video_features = torch.nn.functional.normalize(video_features, dim=1)
                        t = model.logit_scale.exp()
                        logit = (video_features @ text_features.T * t)
                        new_performance = logit[0].item()
                    else:
                        print('Model not recognized - need to implement the forward pass/metric here!')
                        raise NotImplementedError

                    # convert global_idx_heads_to_remove to 0s and 1s indicating which heads to remove
                    bool_concepts_to_remove = torch.zeros(number_total_concepts)
                    bool_concepts_to_remove[concepts_to_remove_idx] = 1
                    bool_concepts_kept = 1 - bool_concepts_to_remove
                    cris_concepts_removed.append(bool_concepts_kept)
                    performance_list.append(new_performance)

                # compute cris importance through a weighted sum of performance and the heads removed to achieve that performance (higher performance * heads kept means more important!)
                cris_concepts_removed = torch.stack(cris_concepts_removed)
                performance_list = torch.tensor(performance_list)
                concept_importance = performance_list @ cris_concepts_removed
                concept_importances.append(concept_importance)
            # average over all videos
            concept_importance =  torch.stack(concept_importances).mean(0)
            baseline_performance = np.mean(baseline_performances)
            MostLeastConceptImportance = concept_importance.argsort(descending=True)
            LeastMostConceptImportance = concept_importance.argsort()
            del model
    else:
        print('Loading results from {}'.format(results_path))
        with open(results_path, 'rb') as f:
            all_results = pickle.load(f)
        concept_importance = all_results['concept_importance']
        MostLeastConceptImportance = concept_importance.argsort(descending=True)
        LeastMostConceptImportance = concept_importance.argsort()
        baseline_performance = all_results['most_to_least'][0]

        # need baseline masks for tcow-ssv2
        baseline_masks = []
        model = load_model(args)
        with torch.no_grad():
            for video_idx in range(num_videos):
                if 'timesformer' in args.model and 'ssv2' in path_to_dataset:
                    # if ssv2, we don't have labels, so just assume the baseline is 1.0 (i.e., the prediction is always correct)
                    video = vcd.post_resize_smooth(dataset[video_idx]).unsqueeze(0).cuda()
                    seeker_query_mask = vcd.post_resize_nearest(vcd.seeker_query_labels[video_idx].squeeze(0)).unsqueeze(0).cuda()

                    (output_mask, output_flags, features) = model(video, seeker_query_mask)
                    baseline_mask = (output_mask > 0.5).cpu().detach()
                    baseline_masks.append(baseline_mask)

    model = load_model(args)
    # compute attribution when removing heads in order from head_importance
    print('1/3: most to least important')
    # most_to_least
    MostLeastResults = concept_removal_performance_curve(vcd, args, model, dataset, all_concept_masks,  all_concepts, MostLeastConceptImportance, target_class_idx,baseline_masks)
    print('2/3: least to most important')
    # least_to_most
    LeastMostResults = concept_removal_performance_curve(vcd, args, model, dataset, all_concept_masks, all_concepts, LeastMostConceptImportance, target_class_idx,baseline_masks)
    print('3/3: random')
    # random
    RandomConceptImportance = torch.randperm(len(all_concepts))
    RandomResults = concept_removal_performance_curve(vcd, args, model, dataset, all_concept_masks, all_concepts, RandomConceptImportance, target_class_idx,baseline_masks)


    # add baseline performance to results if not from saved results
    MostLeastResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), MostLeastResults])
    LeastMostResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), LeastMostResults])
    RandomResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), RandomResults])

    # convert head_importance to dict where the keys are the layer and the values are the head importances
    concept_importance_list = [all_concepts[x][1] for x in MostLeastConceptImportance]
    results = {
        'concept_importance': concept_importance,
        'concept_importance_most_to_least': concept_importance_list,
        'most_to_least': MostLeastResults,
        'least_to_most': LeastMostResults,
        'random': RandomResults,
    }

    # save results
    with open(results_path, 'wb') as f:
        print('Saving results to {}'.format(results_path))
        pickle.dump(results, f)

    if os.path.exists(results_path):
        print('Loading results from {}'.format(results_path))
        with open(results_path, 'rb') as f:
            all_results = pickle.load(f)

        # quantify what layers the most important concepts are in
        layer_scores = {layer: [] for layer in range(12)}
        for idx, concept in enumerate(all_results['concept_importance_most_to_least']):
            layer = int(concept.split('-')[0].split('Layer')[-1])
            score = (len(all_results['concept_importance_most_to_least']) - idx) / len(
                all_results['concept_importance_most_to_least'])
            layer_scores[layer].append(score)

        # average scores
        for layer in layer_scores.keys():
            layer_scores[layer] = np.mean(layer_scores[layer])

        # sort layers by score
        sorted_layer_scores = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
        print('Most important layers:')
        for layer, score in sorted_layer_scores:
            print('Layer {}: {:.3f}'.format(layer, score))

        # plot results
        results = {
            'most_to_least': all_results['most_to_least'],
            'least_to_most': all_results['least_to_most'],
            'random': all_results['random'],
        }
        plot_save_results(args, results)


def plot_save_results(args, results):
    plt.style.use('dark_background')
    colors = sns.color_palette("husl", 3)

    # plot results for each ordering
    for i, (key, result) in enumerate(results.items()):
        frac_concepts_removed = [i / len(result) for i in range(len(result))]
        auc = np.trapz(result, frac_concepts_removed)
        plt.plot(frac_concepts_removed, result, color=colors[i], label=key + ' (AUC: {:.3f})'.format(auc))

    # save data as csv
    csv_data_path = 'results/{}/Readable_ConceptImportance_{}Masks{}.csv'.format(args.exp_name, args.num_masks, args.results_name)
    with open(csv_data_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['concepts_removed', 'standard_scores', 'random_scores', 'inverse_scores'])
        # write every 12 rows
        for i in range(0, len(frac_concepts_removed)):
            writer.writerow([frac_concepts_removed[i], results['most_to_least'][i], results['random'][i], results['least_to_most'][i]])

    plt.xlabel('Concepts removed')
    plt.ylabel('Performance')
    plt.legend()
    plt.title('Concept removal performance curve ({} Masks)'.format(args.num_masks))
    # save and show plot
    plt.savefig('results/{}/AttributionCurve_{}Masks.png'.format(args.exp_name, args.num_masks))
    plt.show()


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

def concept_removal_performance_curve(vcd, args, model, dataset, all_concept_masks, all_concepts, concept_importance, target_class_idx=None, baseline_masks=None):
    ConceptsRemovedEachStep = args.heads_removed_each_step
    NumRemovedStep = (len(all_concepts) // ConceptsRemovedEachStep) + 1
    num_videos = dataset['pv_rgb_tf'].shape[0] if 'kub' in vcd.cached_file_path else dataset.shape[0]
    if 'pre' in vcd.args.model:
        import torch.nn as nn
        mse_loss_func = nn.MSELoss()
        mask = torch.zeros((1, 1568)).cuda().type(torch.bool)
    PerVideoResults = []
    with (torch.no_grad()):
        for video_idx in tqdm(range(num_videos)):
            # compute attribution when removing heads
            all_layer_hook_dict = {layer: {'heads': [], 'masks': [], 'cluster_subject': vcd.args.cluster_subject} for layer in range(12)}
            PerStepResults = []
            for i in range(NumRemovedStep):
                concepts_to_remove_idx = concept_importance[i * ConceptsRemovedEachStep: (i + 1) * ConceptsRemovedEachStep]
                # sample concepts to mask with CRIS
                concepts_to_remove = [all_concepts[x][1] for x in concepts_to_remove_idx]
                for concept_idx in concepts_to_remove:
                    # get all masks that belong to concept for this video
                    layer = int(concept_idx.split('-')[0].split('Layer')[-1])
                    head = int(concept_idx.split('-')[1].split('Head')[-1])

                    # put in layer dict for hook
                    if not concept_idx in all_concept_masks[video_idx].keys():
                        continue
                    all_layer_hook_dict[layer]['heads'].append(head)
                    all_layer_hook_dict[layer]['masks'].append(all_concept_masks[video_idx][concept_idx])


                model = load_model(args, hook=remove_concepts, hook_layer=list(range(12)), model=model,hook_dict=all_layer_hook_dict)
                if 'timesformer' in args.model:
                    if 'kub' in vcd.cached_file_path:
                        output_mask, output_flags, target_mask, features, model_retval = \
                        tcow_timesformer_forward(dataset, model, video_idx, keep_all=False)
                        model_retval = {
                            'output_mask': output_mask.unsqueeze(0),
                            'target_mask': target_mask.unsqueeze(0)}
                        metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                      source_name='kubric')
                        # put results from cuda to cpu
                        metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                        new_performance = metrics_retval['mean_snitch_iou'].item()
                    elif 'ssv2' in vcd.cached_file_path:
                        video = vcd.post_resize_smooth(dataset[video_idx]).unsqueeze(0).cuda()
                        seeker_query_mask = vcd.post_resize_nearest(
                            vcd.seeker_query_labels[video_idx].squeeze(0)).unsqueeze(0).cuda()
                        (output_mask, output_flags, features) = model(video, seeker_query_mask)
                        pred = (output_mask > 0.5).cpu().detach()
                        # compare pred with baseline mask
                        new_performance = compute_iou(pred, baseline_masks[video_idx])
                elif 'vidmae' in args.model or 'mme' in args.model:
                    pre_video = dataset[video_idx].permute(1, 0, 2, 3)

                    # normalize video with imagenet stats
                    video = torch.stack([torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(vid) for vid in pre_video],
                                        dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    if 'pre' in args.model:
                        unnorm_videos = pre_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
                        videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                                                   p0=2, p1=16, p2=16, t=8, h=14, w=14)
                        videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                                       ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                        # we find that the mean is about 0.48 and standard deviation is about 0.08.
                        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)').cuda()
                        pred, _ = model(video, mask, return_features=False)
                        loss = mse_loss_func(input=pred, target=videos_patch)
                        new_performance = -loss.item()
                    else:
                        pred, _ = model(video)
                        pred = pred.argmax(dim=1).item()
                        new_performance = 1.0 if pred == target_class_idx else 0.0
                elif 'intern' in args.model:
                    # video
                    video = dataset[video_idx]
                    video = model.transform(video).unsqueeze(0).cuda()
                    video_features, _ = model.encode_video(video)
                    # text
                    text = model.tokenize(vcd.args.target_class).cuda()
                    text_features = model.encode_text(text)
                    # norm
                    text_features = torch.nn.functional.normalize(text_features, dim=1)
                    video_features = torch.nn.functional.normalize(video_features, dim=1)
                    t = model.logit_scale.exp()
                    logit = (video_features @ text_features.T * t)
                    new_performance = logit[0].item()
                else:
                    print('Model not recognized - need to implement the forward pass/metric here!')
                    raise NotImplementedError
                PerStepResults.append(new_performance)
            PerVideoResults.append(np.array(PerStepResults))

    # average over all videos
    PerVideoResults = np.stack(PerVideoResults).mean(0)
    return PerVideoResults


def remove_concepts(module, input, output):
    # grab hook_dict info
    heads = module.hook_dict['heads']
    masks = module.hook_dict['masks']
    cluster_subject = module.hook_dict['cluster_subject']

    B, N, C = input[0].shape

    # rearrange to get head dimension
    qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]
    if cluster_subject == 'keys':
        feature = k
    elif cluster_subject == 'values':
        feature = v
    elif cluster_subject == 'queries':
        feature = q
    else:
        raise NotImplementedError

    # initialize mask with ones for all heads
    concept_mask = torch.ones_like(feature)
    # combine, reshape, and rearrange masks to match qkv with all heads
    for i, head in enumerate(heads):
        # reshape single head mask
        single_mask = rearrange(masks[i], 't h w -> (t h w)').cuda()
        # tcow vs. other models
        if output.shape[1] == 1568:
            concept_mask[0, head] = torch.mul(concept_mask[0, head].T, single_mask).T
        else:
            single_mask = rearrange(masks[i], 't h w -> t (h w)').cuda()
            concept_mask[:, head, 1:, :] = torch.mul(concept_mask[:, head, 1:, :].T, single_mask.T).T

    # multiply masks with qkv values with all heads
    feature = torch.mul(feature, concept_mask)

    # rearrange back to original shape
    if cluster_subject == 'keys':
        qkv = torch.stack([q, feature, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
    elif cluster_subject == 'values':
        qkv = torch.stack([q, k, feature]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
    elif cluster_subject == 'queries':
        qkv = torch.stack([feature, k, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)

    return qkv

def tcow_timesformer_forward(dataset, model, vid_idx, keep_all=False):
        # hard coded stuff
        qt_idx = 0
        b = 0
        B = 1
        Qs = 1
        seeker_input = dataset['pv_rgb_tf'][vid_idx].unsqueeze(0).cuda()
        all_segm = dataset['pv_segm_tf'][vid_idx].unsqueeze(0).cuda()
        all_div_segm = dataset['pv_div_segm_tf'][vid_idx].unsqueeze(0).cuda()
        inst_count = dataset['pv_inst_count'][vid_idx].unsqueeze(0)
        target_desirability = torch.tensor(dataset['traject_retval_tf'][vid_idx]['desirability_tf']).unsqueeze(0)
        occl_fracs = torch.tensor(dataset['traject_retval_tf'][vid_idx]['occl_fracs_tf']).unsqueeze(0)
        occl_cont_dag = torch.tensor(dataset['traject_retval_tf'][vid_idx]['occl_cont_dag_tf']).unsqueeze(0)
        scene_dp = dataset['full_scene_dp'][vid_idx]

        # Sample either random or biased queries.
        sel_query_inds = tcow.utils.my_utils.sample_query_inds(
            B, Qs, inst_count, target_desirability, model.train_args, 'cuda', 'test')
        query_idx = sel_query_inds[:, 0]
        seeker_query_mask = torch.zeros_like(all_segm, dtype=torch.uint8)  # (B, 1, T, Hf, Wf).
        seeker_query_mask[b, 0, qt_idx] = (all_segm[b, 0, qt_idx] == query_idx[b] + 1)

        # Prepare query mask and ground truths.
        (seeker_query_mask, snitch_occl_by_ptr, full_occl_cont_id, target_mask,
         target_flags) = tcow.data.data_utils.fill_kubric_query_target_mask_flags(
            all_segm, all_div_segm, query_idx, qt_idx, occl_fracs, occl_cont_dag, scene_dp,
            None, model.train_args, 'cuda', 'test')

        del full_occl_cont_id, all_segm, all_div_segm, occl_cont_dag, scene_dp

        # add things to dataset for logging if they were missed, this works with cacheing and non-cached datasets
        if keep_all:
            try:
                dataset['seeker_query_mask'].append(seeker_query_mask.cpu())
                dataset['target_mask'].append(target_mask.cpu())
            except:
                dataset['seeker_query_mask'] = [seeker_query_mask.cpu()]
                dataset['target_mask'] = [target_mask.cpu()]


        # forward pass:
        (output_mask, output_flags, features) = model(seeker_input, seeker_query_mask)

        # prepare model_retval for metrics
        model_retval = {}
        model_retval['target_flags'] = torch.stack([target_flags], dim=1).cuda()  # (B, Qs, T, 3).
        model_retval['snitch_occl_by_ptr'] = torch.stack([snitch_occl_by_ptr], dim=1).cuda()
        cur_occl_fracs = occl_fracs[:, query_idx, :, :].diagonal(0, 0, 1)
        cur_occl_fracs = rearrange(cur_occl_fracs, 'T V B -> B T V')  # (B, T, 3).
        sel_occl_fracs = torch.stack([cur_occl_fracs], dim=1)  # (B, Qs, T, 3).
        model_retval['sel_occl_fracs'] = sel_occl_fracs.cuda()  # (B, Qs, T, 3).
        return output_mask, output_flags, target_mask, features, model_retval

def vcd_args():

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--exp_name', default='test', type=str, help='Experiment used in run_vcd.py')
    parser.add_argument('--overwrite_importance', action='store_true', help='Overwrite importance results')
    parser.add_argument('--overwrite_attribution', action='store_true', help='Overwrite attribution results')
    parser.add_argument('--results_name', default='', type=str, help='Postfix figure name used for saving')
    parser.add_argument('--recompute_performance_curves', action='store_true', help='Load results but recompute performance curves.')
    parser.add_argument('--use_saved_results', action='store_true', help='Use saved results.')

    # concept importance
    parser.add_argument('--removal_type', default='cris', help='Type of attribution removal to do. [perlay | alllay | alllayhead | cris | gradient]')
    parser.add_argument('--num_masks', default=4000, type=int, help='Number of masks to forward pass during random head removal for cris.')
    parser.add_argument('--heads_removed_each_step', default=100, type=int, help='Number of passes during random head removal for cris.')
    parser.add_argument('--masking_ratio', default=0.5, type=float, help='Ratio of concepts to mask for cris.')


    # computation and reproducibility
    parser.add_argument('--max_num_workers', default=16, type=int,help='Maximum number of workers for clustering')
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