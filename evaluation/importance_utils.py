
import os
import sys
import time
import random
import gc
import argparse
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
sys.path.append('/home/ubuntu/video_concept_discovery/')
from utilities.utils import load_model
from models.hide_seek.tcow.eval.metrics import calculate_metrics_mask_track


import torch.nn.functional as F


def compute_concept_importance(args, vcd, break_point=None):
    '''
    Compute concept importance for each video in the dataset.
    Args:
        args: arguments
        vcd: video concept class
        model: model to use for importance
    Returns:
        concept_importance: dictionary of concept importance scores
        {layer: {head: {concept: [score]}}}
        or
        {layer: {head: {concept: [{metric: score}]}}}
    '''

    if args.cat_method == 'integrated_gradients':
        concept_importance = {layer: {head: {concept: [] for concept in vcd.dic[layer][head]['concepts']} for head in args.attn_head} for layer in args.cluster_layer}
    else:
        concept_importance = {layer: {head: {concept: [] for concept in ['baseline'] + vcd.dic[layer][head]['concepts']} for head in args.attn_head} for layer in args.cluster_layer}


    # pass video through first part of model
    # assign every token to nearest concept (or with cnmf assignment matrix)
    # reconstruct video features from concepts
    # pass video features through second part of model
    # occlude token (for now, then implement ig
    # calculate change in metric and then save score to dict

    for layer in args.cluster_layer:
        for head in args.attn_head:
            print('layer: {}, head: {}'.format(layer, head))
            num_vids = vcd.dataset['pv_rgb_tf'].shape[0] if args.dataset == 'kubric' else vcd.dataset.shape[0]
            videos = vcd.dataset['pv_rgb_tf'] if args.dataset == 'kubric' else vcd.dataset
            for vid_idx in tqdm(range(num_vids)):
                if args.debug:
                    if vid_idx == 2:
                        break
                if vid_idx == break_point:
                    break
                input = videos[vid_idx]
                # forward pass and calculate impact of each concept
                results = compute_single_video_importance(args, input, vcd, vid_idx, layer, head)
                for concept in concept_importance[layer][head]:
                    concept_importance[layer][head][concept].append(results[concept])

    # accumulate results
    # if args.cat_method == 'integrated_gradients':
    accum_concept_importance = {layer: {head: {concept: {} for concept in vcd.dic[layer][head]['concepts']} for head in args.attn_head} for layer in args.cluster_layer}
    # else:
        # accum_concept_importance = {layer: {head: {concept: {} for concept in ['baseline'] + vcd.dic[layer][head]['concepts']} for head in args.attn_head} for layer in args.cluster_layer}


    for layer in args.cluster_layer:
        for head in args.attn_head:
            for concept in accum_concept_importance[layer][head]:
                # need to subtract concept importance from baseline importance
                for metric in concept_importance[layer][head][concept][0].keys():
                    if 'occlusion' in args.cat_method:
                        baseline = (sum([x[metric][0] for x in concept_importance[layer][head]['baseline']])/len(concept_importance[layer][head]['baseline']))
                        importance_score = (sum([x[metric][0] for x in concept_importance[layer][head][concept]])/len(concept_importance[layer][head][concept]))
                        accum_concept_importance[layer][head][concept][metric] = baseline - importance_score
                        # accum_concept_importance[layer][head][concept][metric] = baseline - importance_score
                        # if args.cat_method == 'occlusion_hard':
                        #     accum_concept_importance[layer][head][concept][metric] = baseline - importance_score
                        # else:
                        #     accum_concept_importance[layer][head][concept][metric] = importance_score
                    elif args.cat_method == 'integrated_gradients':
                        importance_score = (sum([x[metric][0] for x in concept_importance[layer][head][concept]])/len(concept_importance[layer][head][concept]))
                        # since we sort from smallest to largest, need to make negative
                        accum_concept_importance[layer][head][concept][metric] = -importance_score

    return accum_concept_importance

def compute_single_video_importance(args, input, vcd, vid_idx, layer, head):
    concepts = ['baseline'] + vcd.dic[layer][head]['concepts']

    if args.cat_method == 'integrated_gradients':
        per_concept_results = {concept: {} for concept in vcd.dic[layer][head]['concepts']}
        concept_range = range(len(concepts) - 1)
    else:
        per_concept_results = {concept: {metric: [] for metric in args.attribution_evaluation_metric} for concept in ['baseline'] + vcd.dic[layer][head]['concepts']}
        concept_range = range(-1, len(concepts) - 1)
    for concept_idx in concept_range:

        if args.cat_method == 'integrated_gradients':
            # grads = calculate_outputs_and_gradients(input, vcd)
            # grads is contribution of each input pixel to output
            # need to calculate contribution of each token belonging to concept to output
            # need to use hook, then calculate gradient of output wrt to concept
            hook_dict = {
                'head': head,
                'layer': layer,
                'zero_features': args.zero_features,
                'concept_idx': [concept_idx],
                'vcd': vcd
            }

            # get masks for concept
            mask_paths = [[x, idx] for idx, x in enumerate(vcd.segment_dataset_mask[layer][head]) if
                          x.split('/')[-2].split('_')[1] == str(vid_idx)]
            min_idx = min([x[1] for x in mask_paths])
            max_idx = max([x[1] for x in mask_paths])
            min_max_idx = [min_idx, max_idx]
            masks = np.stack([np.load(mask_path[0]) for mask_path in mask_paths])
            masks = torch.tensor(masks)
            hook_fn = ig_hook
            hook_dict['masks'] = masks
            hook_dict['min_max_idx'] = min_max_idx


            # clear memory
            torch.cuda.empty_cache()
            # clear cpu memory

            scales = [(float(i) / args.ig_steps) for i in range(0, args.ig_steps + 1)]
            gradients = []
            for scale_idx, scale in enumerate(scales):
                hook_dict['scale'] = scale
                vcd.model = load_model(args, hook=hook_fn, hook_layer=str(layer), model=vcd.model, hook_dict=hook_dict, enable_grad=True)
                # vcd.model = load_model(vcd.args)
                # clear grad
                vcd.model.zero_grad()
                # pass video through model
                if 'timesformer' in vcd.args.model:
                    output_mask, output_flags, target_mask, features, model_retval = vcd.tcow_timesformer_forward(vid_idx)
                    # prepare model retval
                    model_retval['output_mask']= output_mask.unsqueeze(0)
                    model_retval['target_mask'] = target_mask.unsqueeze(0)
                    model_retval['output_flags'] = output_flags.unsqueeze(0)
                    # prepare data retval
                    data_retval = {}
                    data_retval['kubric_retval'] = {}
                    data_retval['kubric_retval']['pv_rgb_tf'] = vcd.dataset['pv_rgb_tf'][vid_idx].unsqueeze(0)
                    data_retval['source_name'] = ['kubric']
                    data_retval['kubric_retval']['traject_retval_tf'] = {'query_time': [torch.tensor([0])]}

                    # using predefined loss functions here if possible because it will allow us
                    # to compare losses with snitch, container, occluder, during event losses, etc.
                    loss_retval = vcd.model.loss.per_example(data_retval, model_retval, progress=None, metrics_only=False)

                    # grab loss
                    loss = loss_retval[args.importance_loss]
                    loss.backward()
                    # grab gradients
                    gradient = hook_feature[0].grad.detach().cpu().numpy()[0]
                    gradients.append(gradient)
                elif 'vidmae' in vcd.args.model or 'mme' in vcd.args.model:
                    # if concept_idx == 0 and scale_idx == 0:
                    video = input.permute(1, 0, 2, 3)
                    video = torch.stack([vcd.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    # input = torch.tensor(input, dtype=torch.float32, device='cuda', requires_grad=True)
                    pred, _ = vcd.model(video)
                    output = F.softmax(pred, dim=1)
                    index = np.ones((output.size()[0], 1)) * vcd.target_label_id
                    index = torch.tensor(index, dtype=torch.int64).cuda()
                    output = output.gather(1, index)
                    output.backward()
                    # grab gradients
                    gradient = hook_feature[0].grad.detach().cpu().numpy()[0]
                    gradients.append(-gradient)
                    # gradients.append(gradient)

            gradients = np.array(gradients)
            avg_grads = np.average(gradients[:-1], axis=0)
            # grab baseline
            baseline = torch.cat(baselines).mean(0)

            # grab input features
            input_feature = input_features[0]
            # calculate integrated gradients
            delta_X = (input_feature - baseline).detach().squeeze(0).cpu().numpy()
            integrated_grad = delta_X * avg_grads
            integrated_grad = integrated_grad.sum() # sum over all pixels
            per_concept_results[concepts[concept_idx + 1]]['ig'] = [integrated_grad.item()]


        else:
            hook_dict = {
                'head': head,
                'layer': layer,
                'zero_features': args.zero_features,
                'concept_idx': [concept_idx],
                'vcd': vcd
            }

            if args.cat_method == 'occlusion_hard':
                masked_assignment = vcd.dic[layer][head]['cnmf']['H'].copy()
                # don't mask for baseline
                if concept_idx > -1:
                    masked_assignment[concept_idx] = 0
                feature_hat = np.matmul(masked_assignment.T, vcd.dic[layer][head]['cnmf']['W'].T)
                hook_dict['feature_hat'] = feature_hat
                hook_fn = occlude_feature_hard
            elif args.cat_method == 'occlusion_soft':
                # get masks for concept
                mask_paths = [[x, idx] for idx, x in enumerate(vcd.segment_dataset_mask[layer][head]) if
                              x.split('/')[-2].split('_')[1] == str(vid_idx)]
                min_idx = min([x[1] for x in mask_paths])
                max_idx = max([x[1] for x in mask_paths])
                min_max_idx = [min_idx, max_idx]
                masks = np.stack([np.load(mask_path[0]) for mask_path in mask_paths])
                masks = torch.tensor(masks)
                hook_fn = occlude_feature_soft
                hook_dict['masks'] = masks
                hook_dict['min_max_idx'] = min_max_idx

            # clear memory
            torch.cuda.empty_cache()
            # clear cpu memory
            if concept_idx == -1:
                vcd.model = load_model(args)
            else:
                vcd.model = load_model(args, hook=hook_fn, hook_layer=str(layer), model=vcd.model, hook_dict=hook_dict)

            # pass video through model
            if 'timesformer' in vcd.args.model:
                output_mask, output_flags, target_mask, features, _ = vcd.tcow_timesformer_forward(vid_idx)
                model_retval = {
                    'output_mask': output_mask.unsqueeze(0),
                    'target_mask': target_mask.unsqueeze(0)
                }
                metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                              source_name='kubric')
                # put results from cuda to cpu
                metrics_retval = {k: v.cpu().item() for k, v in metrics_retval.items()}
                # add results to results dict
                for metric in args.attribution_evaluation_metric:
                    per_concept_results[concepts[concept_idx+1]][metric].append(metrics_retval[metric])
            elif 'vidmae' in vcd.args.model or 'mme' in vcd.args.model:
                video = input.permute(1, 0, 2, 3)
                video = torch.stack([vcd.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                video = video.unsqueeze(0).cuda()
                pred, _ = vcd.model(video)
                pred = pred.argmax(dim=1)
                per_concept_results[concepts[concept_idx + 1]]['acc'] = [pred.item() == vcd.target_label_id]
    gc.collect()

    return per_concept_results



def compute_concept_attribution(args, vcd, concept_importance, mode='insertion', baseline=None):
    '''
    Compute concept importance attribution.
    Args:
        args: arguments
        vcd: vcd object
        concept_importance: concept importance dictionary where larger scores imply more important concepts
        mode: 'insertion' or 'deletion'
    Returns:
        results: dictionary of results
        {layer: {head: {concept: {metric: acc}}}}} where acc is a list of len(num_concepts) of accuracies [acc1, acc2, ...]
    '''

    # for layer
    # for head
    # for concept in concepts:
    # get_acc_without_concept
    results = {layer: {head: {metric: {num_concept_perturb: [] for num_concept_perturb in range(len(concept_importance[layer][head]) + 1)} for metric in args.attribution_evaluation_metric} for head in args.attn_head} for layer in args.cluster_layer}
    # load new model without hooks
    # vcd.model = load_model(args)

    for layer in args.cluster_layer:
        for head in args.attn_head:
            print('Computing attribution for layer {} head {}'.format(layer, head))
            # if args.cat_method == 'integrated_gradients':
            #     all_concepts = [x for x in range(len(concept_importance[layer][head].keys()))]
            # else:
            #     all_concepts = [x for x in range(len(concept_importance[layer][head].keys()))]
            all_concepts = [x for x in range(len(concept_importance[layer][head].keys()))]
            # for metric in args.attribution_evaluation_metric:
            for metric in concept_importance[layer][head]['concept_1'].keys():
            # obtain ranked conceept ranking based on importance
                ranked_concepts = sorted(concept_importance[layer][head].items(), key=lambda x: x[1][metric], reverse=False)
                if baseline == 'random':
                    ranked_concepts = random.sample(ranked_concepts, len(ranked_concepts))
                elif baseline == 'inverse':
                    ranked_concepts = list(reversed(ranked_concepts))
                num_vids = vcd.dataset['pv_rgb_tf'].shape[0] if args.dataset == 'kubric' else vcd.dataset.shape[0]
                for vid_idx in tqdm(range(num_vids)):
                    if args.debug:
                        if vid_idx == 2:
                            break
                    concept_indices = []
                    for num_concept_perturb, concept in enumerate([('baseline', 0)] + ranked_concepts):
                        # get masks for concept
                        mask_paths = [[x, idx] for idx, x in enumerate(vcd.segment_dataset_mask[layer][head]) if x.split('/')[-2].split('_')[1] == str(vid_idx)]
                        min_idx = min([x[1] for x in mask_paths])
                        max_idx = max([x[1] for x in mask_paths])
                        min_max_idx = [min_idx, max_idx]
                        masks = np.stack([np.load(mask_path[0]) for mask_path in mask_paths])
                        masks = torch.tensor(masks)

                        if mode == 'insertion':
                            if concept[0] != 'baseline':
                                concept_indices.append(int(concept[0].split('_')[-1])-1)
                                remove_indices = [x for x in all_concepts if not x in concept_indices]
                            else:
                                remove_indices = [i for i in range(len(all_concepts))]

                        elif mode == 'deletion':
                            if concept[0] != 'baseline':
                                concept_indices.append(int(concept[0].split('_')[-1])-1)
                                remove_indices = concept_indices
                            else:
                                remove_indices = []

                        hook_dict = {
                            'head': head,
                            'layer': layer,
                            'zero_features': args.zero_features,
                            'masks': masks,
                            'min_max_idx': min_max_idx,
                            'concept_idx': remove_indices,
                            'vcd': vcd
                        }

                        vcd.model = load_model(args, hook=occlude_feature_soft, hook_layer=str(layer), model=vcd.model, hook_dict=hook_dict)

                        # pass video through model
                        if 'timesformer' in vcd.args.model:
                            output_mask, output_flags, target_mask, features, _ = vcd.tcow_timesformer_forward(vid_idx)
                            model_retval = {
                                'output_mask': output_mask.unsqueeze(0),
                                'target_mask': target_mask.unsqueeze(0)
                            }
                            metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                          source_name='kubric')
                            # put results from cuda to cpu
                            metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                            # add results to results dict
                            for metric in args.attribution_evaluation_metric:
                                results[layer][head][metric][num_concept_perturb].append(metrics_retval[metric].item())

                            # print('Concept {} ------- mIoU == {}'.format(concept, metrics_retval[metric].item()))

                        elif 'vidmae' in vcd.args.model or 'mme' in vcd.args.model:
                            input = vcd.dataset[vid_idx]
                            video = input.permute(1, 0, 2, 3)
                            video = torch.stack([vcd.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                            video = video.unsqueeze(0).cuda()
                            pred, _ = vcd.model(video)
                            pred = pred.argmax(dim=1)
                            results[layer][head][metric][num_concept_perturb].append(float(pred.item() == vcd.target_label_id))
                        else:
                            raise NotImplementedError
    return results



def compute_concept_attribution_all_layers(args, vcd, concept_importance, mode='insertion', baseline=None):
    '''
    Compute concept importance attribution.
    Args:
        args: arguments
        vcd: vcd object
        concept_importance: concept importance dictionary where larger scores imply more important concepts
        mode: 'insertion' or 'deletion'
    Returns:
        results: dictionary of results
        {layer: {head: {concept: {metric: acc}}}}} where acc is a list of len(num_concepts) of accuracies [acc1, acc2, ...]
    '''

    # get maximum number of concepts in any layer
    # initialize results dictionary, one for all layers
    # results = {layer: {head: {metric: {num_concept_perturb: [] for num_concept_perturb in range(len(concept_importance[layer][head]) + 1)} for metric in args.attribution_evaluation_metric} for head in args.attn_head} for layer in args.cluster_layer}
    all_layers = [x for x in concept_importance.keys()]
    results = {}
    # for layer in args.cluster_layer:
    for head in args.attn_head:
        max_num_concepts = max([len(concept_importance[layer][head].keys()) for layer in args.cluster_layer])
        per_head_results = {metric: {num_concept_perturb: [] for num_concept_perturb in range(max_num_concepts + 1)} for metric
                   in args.attribution_evaluation_metric}

        print('Computing attribution for layers {} head {}'.format(all_layers, head))
        if args.cat_method == 'integrated_gradients':
            # all_concepts = [x for x in range(len(concept_importance[layer][head].keys()))]
            all_concepts = list(range(max_num_concepts))
        else:
            # all_concepts = [x for x in range(len(concept_importance[layer][head].keys())-1)]
            all_concepts = list(range(max_num_concepts-1))
        # for metric in args.attribution_evaluation_metric:
        for importance_loss in concept_importance[all_layers[0]][head]['concept_1'].keys():
        # obtain ranked conceept ranking based on importance
            ranked_concepts = {}
            for layer in all_layers:
                ranked_concepts[layer] = sorted(concept_importance[layer][head].items(), key=lambda x: x[1][importance_loss], reverse=False)
                if baseline == 'random':
                    ranked_concepts[layer] = random.sample(ranked_concepts[layer], len(ranked_concepts[layer]))
                elif baseline == 'inverse':
                    ranked_concepts[layer] = list(reversed(ranked_concepts[layer]))
            # ranked_concepts = sorted(concept_importance[layer][head].items(), key=lambda x: x[1][metric], reverse=False)
            # if baseline == 'random':
            #     ranked_concepts = random.sample(ranked_concepts, len(ranked_concepts))
            # elif baseline == 'inverse':
            #     ranked_concepts = list(reversed(ranked_concepts))
            num_vids = vcd.dataset['pv_rgb_tf'].shape[0] if args.dataset == 'kubric' else vcd.dataset.shape[0]
            for vid_idx in tqdm(range(num_vids)):
                if args.debug:
                    if vid_idx == 2:
                        break
                concept_indices = {layer: [] for layer in all_layers}
                for num_concept_perturb in range(len(all_concepts) + 1):
                    # get masks for concept
                    all_layer_hook_dict = {}
                    all_min_max_idx = {}
                    all_layer_masks = {}
                    for layer in all_layers:

                        if num_concept_perturb != 0:
                            if num_concept_perturb <= len(ranked_concepts[layer]):
                                # if we have already inserted all concepts, then don't insert any more, else insert
                                concept_indices[layer].append(int(ranked_concepts[layer][num_concept_perturb - 1][0].split('_')[-1])-1)
                            remove_indices = [x for x in all_concepts if not x in concept_indices[layer]] if mode == 'insertion' else concept_indices[layer]
                        else:
                            remove_indices = [i for i in range(len(all_concepts))] if mode == 'insertion' else []

                        # if mode == 'insertion':
                        #     if num_concept_perturb != 0:
                        #         if num_concept_perturb > len(ranked_concepts[layer]):
                        #             # if we have already inserted all concepts, then don't insert any more
                        #             pass
                        #         else:
                        #             concept_indices[layer].append(int(ranked_concepts[layer][num_concept_perturb - 1][0].split('_')[-1])-1)
                        #             remove_indices = [x for x in all_concepts if not x in concept_indices[layer]]
                        #     else:
                        #         remove_indices = [i for i in range(len(all_concepts))]
                        #
                        # elif mode == 'deletion':
                        #     if num_concept_perturb != 0:
                        #         if num_concept_perturb > len(ranked_concepts[layer]):
                        #             # if we have already deleted all concepts, then don't insert any more
                        #             pass
                        #         else:
                        #             concept_indices[layer].append(int(ranked_concepts[layer][num_concept_perturb - 1][0].split('_')[-1])-1)
                        #             remove_indices = concept_indices[layer]
                        #     else:
                        #         remove_indices = []
                        # else:
                        #     raise NotImplementedError


                        mask_paths = [[x, idx] for idx, x in enumerate(vcd.segment_dataset_mask[layer][head]) if x.split('/')[-2].split('_')[1] == str(vid_idx)]
                        min_idx = min([x[1] for x in mask_paths])
                        max_idx = max([x[1] for x in mask_paths])
                        min_max_idx = [min_idx, max_idx]
                        masks = np.stack([np.load(mask_path[0]) for mask_path in mask_paths])
                        masks = torch.tensor(masks)
                        all_layer_masks[layer] = masks
                        all_min_max_idx[layer] = min_max_idx

                        # set per-layer remove_indices to be up to the total number of concepts in that layer
                        max_val = max(remove_indices) if len(remove_indices) > 0 else 0
                        if max_val >= len(concept_importance[layer][head].keys()):
                            # remove all indices that are greater than the number of concepts in that layer
                            layer_remove_indices = [x for x in remove_indices if x < len(concept_importance[layer][head].keys())]
                        else:
                            layer_remove_indices = remove_indices

                        all_layer_hook_dict[layer] = {
                            'head': head,
                            'zero_features': args.zero_features,
                            'masks': all_layer_masks[layer],
                            'min_max_idx': all_min_max_idx[layer],
                            'concept_idx': layer_remove_indices,
                            'layer': layer,
                            'vcd': vcd
                        }
                    # hook_dict = {
                    #     'head': head,
                    #     'zero_features': args.zero_features,
                    #     'masks': all_layer_masks,
                    #     'min_max_idx': all_min_max_idx,
                    #     'concept_idx': remove_indices,
                    #     'vcd': vcd
                    # }

                    vcd.model = load_model(args, hook=occlude_feature_soft, hook_layer=all_layers, model=vcd.model, hook_dict=all_layer_hook_dict)

                    # pass video through model
                    if 'timesformer' in vcd.args.model:
                        output_mask, output_flags, target_mask, features, _ = vcd.tcow_timesformer_forward(vid_idx)
                        model_retval = {
                            'output_mask': output_mask.unsqueeze(0),
                            'target_mask': target_mask.unsqueeze(0)
                        }
                        metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                      source_name='kubric')
                        # put results from cuda to cpu
                        metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                        # add results to results dict
                        for metric in args.attribution_evaluation_metric:
                            per_head_results[metric][num_concept_perturb].append(metrics_retval[metric].item())
                    elif 'vidmae' in vcd.args.model or 'mme' in vcd.args.model:
                        input = vcd.dataset[vid_idx]
                        video = input.permute(1, 0, 2, 3)
                        video = torch.stack([vcd.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                        video = video.unsqueeze(0).cuda()
                        pred, _ = vcd.model(video)
                        pred = pred.argmax(dim=1)
                        per_head_results[importance_loss][num_concept_perturb].append(float(pred.item() == vcd.target_label_id))
                    else:
                        raise NotImplementedError
        results[head] = per_head_results
    results = {'All': results}
    return results


def compute_concept_attribution_all_layers_all_heads(args, vcd, concept_importance, mode='insertion', baseline=None):
    '''
    Compute concept importance attribution.
    Args:
        args: arguments
        vcd: vcd object
        concept_importance: concept importance dictionary where larger scores imply more important concepts
        mode: 'insertion' or 'deletion'
    Returns:
        results: dictionary of results
        {layer: {head: {concept: {metric: acc}}}}} where acc is a list of len(num_concepts) of accuracies [acc1, acc2, ...]
    '''

    # get maximum number of concepts in any layer
    # initialize results dictionary, one for all layers



    all_layers = [x for x in concept_importance.keys()]

    # sort all concepts in all layers and heads together in a single list
    concept_score_global = {}
    for importance_loss in concept_importance[all_layers[0]][0]['concept_1'].keys():
        for layer in concept_importance:
            for head in concept_importance[layer]:
                head_average_score = np.mean([concept_importance[layer][head][concept][importance_loss] for concept in concept_importance[layer][head]])
                concept_score_global['layer_{}-head_{}'.format(layer, head)] = {importance_loss: head_average_score}

    num_vids = vcd.dataset['pv_rgb_tf'].shape[0] if args.dataset == 'kubric' else vcd.dataset.shape[0]
    results = {metric: {num_head: [] for num_head in range(len(concept_score_global))} for metric in args.attribution_evaluation_metric}

    # sort concepts by score
    for importance_loss in concept_importance[all_layers[0]][0]['concept_1'].keys():
        sorted_head_scores = sorted(concept_score_global.items(), key=lambda x: x[1][importance_loss], reverse=True)
        if baseline == 'random':
            sorted_head_scores = random.sample(sorted_head_scores, len(sorted_head_scores))
        elif baseline == 'inverse':
            sorted_head_scores = list(reversed(sorted_head_scores))

        # iterate through videos and remove heads consecutively in order of importance
        for vid_idx in tqdm(range(num_vids)):
            if args.debug:
                if vid_idx == 2:
                    break

            layer_head_hook_dict = {}
            for num_head_removed, target in enumerate(sorted_head_scores):
                layer = int(target[0].split('-')[0].split('_')[1])
                if layer not in layer_head_hook_dict:
                    layer_head_hook_dict[layer] = {}
                    layer_head_hook_dict[layer]['heads'] = [int(target[0].split('-')[1].split('_')[1])]
                    layer_head_hook_dict[layer]['layer'] = layer
                    layer_head_hook_dict[layer]['cluster_subject'] = vcd.args.cluster_subject
                    layer_head_hook_dict[layer]['zero_features'] = args.zero_features
                else:
                    layer_head_hook_dict[layer]['heads'].append(int(target[0].split('-')[1].split('_')[1]))



                hook_layers = list(layer_head_hook_dict.keys())

                vcd.model = load_model(args, hook=remove_heads_from_layer, hook_layer=hook_layers, model=vcd.model, hook_dict=layer_head_hook_dict)

                if 'timesformer' in vcd.args.model:
                    output_mask, output_flags, target_mask, features, _ = vcd.tcow_timesformer_forward(vid_idx)
                    model_retval = {
                        'output_mask': output_mask.unsqueeze(0),
                        'target_mask': target_mask.unsqueeze(0)
                    }
                    metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                  source_name='kubric')
                    # put results from cuda to cpu
                    metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                    # add results to results dict
                    for metric in args.attribution_evaluation_metric:
                        results[metric][num_head_removed].append(metrics_retval[metric].item())
                elif 'vidmae' in vcd.args.model or 'mme' in vcd.args.model:
                    input = vcd.dataset[vid_idx]
                    video = input.permute(1, 0, 2, 3)
                    video = torch.stack([vcd.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    pred, _ = vcd.model(video)
                    pred = pred.argmax(dim=1)
                    results[list(results.keys())[0]][num_head_removed].append(float(pred.item() == vcd.target_label_id))
                else:
                    raise NotImplementedError

    results = {'All': {'All': results}}
    return results


# importance methods
def occlude_feature_hard(module, input, output):
    '''
    Perturb the output of a given layer by adding noise.
    Args:
        module: the layer to perturb
        input: input to the layer
        output: output of the layer to perturb
        eps: noise level
        direction: 'reverse' or 'forward' or 'random'
    Returns:
        perturbed output at that layer
    '''

    layer = module.hook_dict['layer']
    head = module.hook_dict['head']
    concept_idx = module.hook_dict['concept_idx']
    vcd = module.hook_dict['vcd']

    # temporal shape = 300,12,30,64
    # spatial shape = 30,12,301,64
    B, N, C = input[0].shape
    if vcd.args.cluster_subject == 'tokens':
        feature = rearrange(output[0], 't s (nh m) -> t s nh m', m=768 // 12, nh=12)
    elif vcd.args.cluster_subject == 'block_token':
        feature = output[0]
    elif vcd.args.cluster_subject in ['queries', 'keys', 'values']:
        qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if vcd.args.cluster_subject == 'keys':
            feature = k
        elif vcd.args.cluster_subject == 'values':
            feature = v
        elif vcd.args.cluster_subject == 'queries':
            feature = q
        else:
            raise NotImplementedError
    else:
        feature = output[0]

    # old method
    # interpolate masks to same size as feature_hat
    # masks = F.interpolate(masks.unsqueeze(0).type(torch.uint8), size=(30, 15, 20), mode='nearest').squeeze(0).cuda()
    # reshape feature and masks
    # masks = rearrange(masks, 'b t h w -> b (t h w)')
    # grab features only for this video
    # feature_hat = module.feature_hat[min_max_idx[0]:min_max_idx[1] + 1]
    # replace each channel with the corresponding feature hat vector
    # reconstruct per-pixel features using reconstruction matrix

    # divide by the number of 1's in the mask

    asg = torch.cdist(feature[:, 1:, :], torch.tensor(vcd.dic[layer][head]['cnmf']['W'].T).unsqueeze(0).cuda().float(), p=2).argmax(2).squeeze(0)

    # reconstructed_feature = ((masks.T.float() @ torch.tensor(feature_hat).cuda().float()).T / masks.sum(0)).T
    # if not vcd.args.cluster_subject == 'block_token':
    #     reconstructed_feature = rearrange(reconstructed_feature, '(t h w) m -> t (h w) m', h=15, w=20, t=30)
    # if vcd.args.cluster_subject == 'tokens':
    #     feature[:, 1:, head] = reconstructed_feature
    #     return rearrange(feature, 't s nh m -> t s (nh m)'), output[1]
    # elif vcd.args.cluster_subject == 'keys':
    #     k[:, head, 1:, :] = reconstructed_feature
    # elif vcd.args.cluster_subject == 'values':
    #     v[:, head, 1:, :] = reconstructed_feature
    # elif vcd.args.cluster_subject == 'queries':
    #     q[:, head, 1:, :] = reconstructed_feature
    if vcd.args.cluster_subject == 'block_token':
        index_to_mask = torch.where(asg == concept_idx[0])[0].type(torch.LongTensor)
        feature[:, 1:][:, index_to_mask, :] = 0
        return feature, output[1]
    else:
        raise NotImplementedError

    qkv = torch.stack([q, k, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
    return qkv

def occlude_feature_soft(module, input, output):
    '''
    Perturb the output of a given layer by adding noise.
    Args:
        module: the layer to perturb
        input: input to the layer
        output: output of the layer to perturb
        eps: noise level
        direction: 'reverse' or 'forward' or 'random'
    Returns:
        perturbed output at that layer
    '''

    masks = module.hook_dict['masks']
    min_max_idx = module.hook_dict['min_max_idx']
    layer = module.hook_dict['layer']
    head = module.hook_dict['head']
    concept_idx = module.hook_dict['concept_idx']
    vcd = module.hook_dict['vcd']

    # temporal shape = 300,12,30,64
    # spatial shape = 30,12,301,64
    B, N, C = input[0].shape
    if vcd.args.cluster_subject == 'tokens':
        feature = rearrange(output[0], 't s (nh m) -> t s nh m', m=768 // 12, nh=12)
    elif vcd.args.cluster_subject == 'block_token':
        feature = output[0]
    elif vcd.args.cluster_subject in ['queries', 'keys', 'values']:
        qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if vcd.args.cluster_subject == 'keys':
            feature = k
        elif vcd.args.cluster_subject == 'values':
            feature = v
        elif vcd.args.cluster_subject == 'queries':
            feature = q
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # current method
    # hard code shape of tensor between models... (not very nice, should use module attribute)
    if output[0].shape[1] == 1568 or output[0].shape[0] == 1568:
        masks = F.interpolate(masks.unsqueeze(0).type(torch.uint8), size=(8, 14, 14), mode='nearest').squeeze(0).cuda().float()
    else:
        masks = F.interpolate(masks.unsqueeze(0).type(torch.uint8), size=(30, 15, 20), mode='nearest').squeeze(0).cuda().float()
    # masks = rearrange(masks, 'b t h w -> b (t h w)')
    masks = rearrange(masks, 'b t h w -> b (h w t)')
    G = torch.tensor(vcd.dic[layer][head]['cnmf']['G'])
    G_curr = G[min_max_idx[0]:min_max_idx[1] + 1]  # .softmax(1)
    # scale G_curr between 0-1 over concept dimension (i.e., to enforce that removing all concepts = 0)
    G_curr = (G_curr - G_curr.min(1)[0].unsqueeze(1)) / (G_curr.max(1)[0] - G_curr.min(1)[0]).unsqueeze(1)
    G_inv = 1 - G_curr[:, concept_idx].cuda().float()
    # replace masks with values of G_inv
    mask_weight = (masks.T.float() @ G_inv.float()).T
    # feature_copy = feature.clone()
    for i in range(mask_weight.shape[0]):
        # multiple feature with mask_weight
        # deal with cls token
        if vcd.args.cluster_subject == 'block_token':
            if mask_weight.shape[1] == feature.shape[1]:
                feature[0] = torch.mul(feature[0].T, mask_weight[i]).T
            else:
                feature[0, 1:] = torch.mul(feature[0, 1:].T, mask_weight[i]).T
        else:

            if feature[0,head].T.shape[1] == mask_weight[i].shape[0]:
                feature[0,head] = torch.mul(feature[0,head].T, mask_weight[i]).T
            else:
                new_feat = rearrange(feature[:, :, 1:], '(b t) n (h w) c -> b n (h w t) c', t=30, h=15, w=20)
                return_feat = torch.mul(new_feat[0,head].T, mask_weight[i]).T
                return_feat = rearrange(return_feat.unsqueeze(0), 'b (h w t) c -> (b t) (h w) c', t=30, h=15, w=20)
                feature[:, head, 1:] = return_feat

    if vcd.args.cluster_subject == 'block_token':
        return feature, output[1]
    else:
        if vcd.args.cluster_subject == 'keys':
            qkv = torch.stack([q, feature, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        elif vcd.args.cluster_subject == 'values':
            qkv = torch.stack([q, k, feature]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        elif vcd.args.cluster_subject == 'queries':
            qkv = torch.stack([feature, k, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        return qkv


def remove_heads_from_layer(module, input, output):
    '''
    Perturb the output of a given layer by adding noise.
    Args:
        module: the layer to perturb
        input: input to the layer
        output: output of the layer to perturb
        eps: noise level
        direction: 'reverse' or 'forward' or 'random'
    Returns:
        perturbed output at that layer
    '''

    layer = module.hook_dict['layer']
    head = module.hook_dict['heads']
    cluster_subject = module.hook_dict['cluster_subject']

    # temporal shape = 300,12,30,64
    # spatial shape = 30,12,301,64
    B, N, C = input[0].shape
    if cluster_subject == 'tokens':
        feature = rearrange(output[0], 't s (nh m) -> t s nh m', m=768 // 12, nh=12)
    elif cluster_subject == 'block_token':
        feature = output[0]
    elif cluster_subject in ['queries', 'keys', 'values']:
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
    else:
        raise NotImplementedError

    # set all heads to zero
    feature[:, head] = 0

    if cluster_subject == 'block_token':
        return feature, output[1]
    else:
        if cluster_subject == 'keys':
            qkv_out = torch.stack([q, feature, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        elif cluster_subject == 'values':
            qkv_out = torch.stack([q, k, feature]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        elif cluster_subject == 'queries':
            qkv_out = torch.stack([feature, k, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        return qkv_out

hook_feature = []
baselines = []
input_features = []
def ig_hook(module, input, output):
    # pop all elements in hook_feature, baselines, input_features
    hook_feature.clear()
    baselines.clear()
    input_features.clear()

    masks = module.hook_dict['masks']
    min_max_idx = module.hook_dict['min_max_idx']
    layer = module.hook_dict['layer']
    head = module.hook_dict['head']
    concept_idx = module.hook_dict['concept_idx']
    vcd = module.hook_dict['vcd']
    scale = module.hook_dict['scale']


    # need to replace tokens belonging to concept with scaled version of input - baseline
    # scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(0, steps + 1)]

    # temporal shape = 300,12,30,64
    # spatial shape = 30,12,301,64
    # temporal shape = 300,12,30,64
    # spatial shape = 30,12,301,64
    B, N, C = input[0].shape
    if vcd.args.cluster_subject == 'tokens':
        feature = rearrange(output[0], 't s (nh m) -> t s nh m', m=768 // 12, nh=12)
    elif vcd.args.cluster_subject == 'block_token':
        feature = output[0]
        cls_token = feature[:, 0]
    elif vcd.args.cluster_subject in ['queries', 'keys', 'values', 'tokens']:
        qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if vcd.args.cluster_subject == 'keys':
            feature = k
        elif vcd.args.cluster_subject == 'values':
            feature = v
        elif vcd.args.cluster_subject == 'queries':
            feature = q
        elif vcd.args.cluster_subject == 'tokens':
            print('tokens')
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # current method
    # hard code shape of tensor between models... (not very nice, should use module attribute)
    if output[0].shape[1] == 1568 or output[0].shape[0] == 1568:
        masks = F.interpolate(masks.unsqueeze(0).type(torch.uint8), size=(8, 14, 14), mode='nearest').squeeze(0).cuda().float()
    else:
        masks = F.interpolate(masks.unsqueeze(0).type(torch.uint8), size=(30, 15, 20), mode='nearest').squeeze(0).cuda().float()
    # masks = rearrange(masks, 'b t h w -> b (t h w)')
    masks = rearrange(masks, 'b t h w -> b (h w t)')
    G = torch.tensor(vcd.dic[layer][head]['cnmf']['G'])
    G_curr = G[min_max_idx[0]:min_max_idx[1] + 1]
    asg = G_curr.argmax(1)
    if vcd.args.cluster_subject == 'block_token':

        # get locations of tokens not belonging to concept
        index_to_mask = torch.where(asg != concept_idx[0])[0].type(torch.LongTensor)
        # set all mask values to 0 if they do not belong to concept
        masks[index_to_mask] = 0
        # sum over B dimension to get mask
        video_mask = masks.sum(0)
        # expand mask to match feature dimension
        video_mask_ig = video_mask.unsqueeze(1).expand(masks.shape[1], 768).unsqueeze(0)
        # invert mask
        video_mask_inv = torch.where(video_mask_ig == 1, 0, 1)
        input_features.append(feature)
        # get baseline (0 feature)
        baseline = 0 * feature
        baselines.append(baseline)
        # calculate scaled feature for all concepts
        return_feature = baseline + scale * (feature - baseline)

        # set requires_grad to true for part of feature belonging to concept
        return_feature = torch.tensor(return_feature.cpu(), device='cuda', requires_grad=True)
        hook_feature.append(return_feature)

        # combine return_feature and feature based on mask and invert mask
        # if mask_weight.shape[1] == feature.shape[1]:
        #     feature[0] = torch.mul(feature[0].T, mask_weight[i]).T
        # else:
        #     feature[0, 1:] = torch.mul(feature[0, 1:].T, mask_weight[i]).T

        if video_mask_inv.shape[1] == feature.shape[1]:
            return_feature = video_mask_ig * return_feature + video_mask_inv * feature
        else:
            return_feature = video_mask_ig * return_feature[:,1:] + video_mask_inv * feature[:,1:]
            return_feature = torch.cat((cls_token.unsqueeze(1), return_feature), dim=1)
        return return_feature, output[1]
    else:

        # get locations of tokens not belonging to concept
        index_to_mask = torch.where(asg != concept_idx[0])[0].type(torch.LongTensor)
        # set all mask values to 0 if they do not belong to concept
        masks[index_to_mask] = 0
        # sum over B dimension to get mask
        video_mask = masks.sum(0)
        # expand mask to match feature dimension (64 for individual heads))
        video_mask = video_mask.unsqueeze(1).expand(masks.shape[1], feature.shape[-1]).unsqueeze(0)
        # expand mask to match number of heads
        video_mask = video_mask.unsqueeze(1).expand(1, k.shape[1], video_mask.shape[1], video_mask.shape[2])
        # set all other heads to 0
        other_heads = torch.where(torch.arange(k.shape[1]) != head)[0].type(torch.LongTensor)
        # no idea why this is necessary, but it is (otherwise video_mask_ig is all zeros)
        video_mask_ig = video_mask.clone()
        video_mask_ig[:, other_heads] = 0
        # invert mask
        video_mask_inv = torch.where(video_mask_ig == 1, 0, 1)
        input_features.append(feature)
        # get baseline (0 feature)
        baseline = 0 * feature
        baselines.append(baseline)
        # calculate scaled feature for all concepts
        return_feature = baseline + scale * (feature - baseline)

        # set requires_grad to true for part of feature belonging to concept
        return_feature = torch.tensor(return_feature.cpu(), device='cuda', requires_grad=True)
        hook_feature.append(return_feature)

        if 'timesformer' in vcd.args.model:
            video_mask_ig = rearrange(video_mask_ig, 'b n (h w t) c -> (b t) n (h w) c', t=30, h=15, w=20)
            video_mask_inv = rearrange(video_mask_inv, 'b n (h w t) c -> (b t) n (h w) c', t=30, h=15, w=20)
            cls_token = feature[:, :, 0]
            return_feature = video_mask_ig * return_feature[:, :, 1:] + video_mask_inv * feature[:, :, 1:]
            return_feature = torch.cat((cls_token.unsqueeze(2), return_feature), dim=2)
        else: # videomae
            # combine return_feature and feature based on mask and invert mask (differentiable)
            if video_mask_inv.shape[1] == feature.shape[1]:
                return_feature = video_mask_ig * return_feature + video_mask_inv * feature
            else:
                return_feature = video_mask_ig * return_feature[:,1:] + video_mask_inv * feature[:,1:]
                return_feature = torch.cat((cls_token.unsqueeze(1), return_feature), dim=1)

        # stack qkv and reshape and return
        if vcd.args.cluster_subject == 'keys':
            qkv = torch.stack([q, return_feature, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        elif vcd.args.cluster_subject == 'values':
            qkv = torch.stack([q, k, return_feature]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        elif vcd.args.cluster_subject == 'queries':
            qkv = torch.stack([return_feature, k, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
        return qkv

def calculate_outputs_and_gradients(input, vcd, steps=4):
    # do the pre-processing
    baseline = 0 * input    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (input - baseline) for i in range(0, steps + 1)]
    gradients = []
    for input in scaled_inputs:

        if 'timesformer' in vcd.args.model:
            raise NotImplementedError
        elif 'vidmae' in vcd.args.model or 'mme' in vcd.args.model:
            input = input.permute(1, 0, 2, 3)
            input = torch.stack([vcd.normalize(vid) for vid in input], dim=0).permute(1, 0, 2, 3)
            input = input.unsqueeze(0)
            input = torch.tensor(input, dtype=torch.float32, device='cuda', requires_grad=True)
            pred, _ = vcd.model(input)
            output = F.softmax(pred, dim=1)
            index = np.ones((output.size()[0], 1)) * vcd.target_label_id
            index = torch.tensor(index, dtype=torch.int64).cuda()
            output = output.gather(1, index)
            # clear grad
            vcd.model.zero_grad()
            output.backward()
            gradient = input.grad.detach().cpu().numpy()[0]
            gradients.append(gradient)
    gradients = np.array(gradients)
    avg_grads = np.average(gradients[:-1], axis=0)
    # avg_grads = np.transpose(avg_grads, (1,2,3,0))
    delta_X = (input - baseline.cuda()).detach().squeeze(0).cpu().numpy()
    integrated_grad = delta_X * avg_grads
    return integrated_grad
