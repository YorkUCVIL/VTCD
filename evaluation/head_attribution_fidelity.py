
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
sys.path.append('/home/ubuntu/VTCD/')
from utilities.utils import load_model
from models.hide_seek.tcow.eval.metrics import calculate_metrics_mask_track
from evaluation.importance_utils import *
import torch.nn.functional as F
import torchvision
import models.hide_seek.tcow as tcow
from decord import VideoReader, cpu

def main(args):
    '''
    Compute concept importance attribution.
    Args:
        args: arguments from command line

    Returns:
        results: dictionary of results

    pseudocode:
    load dataset

    # Gradient-based
    for each video, target in dataset:
        load model
        model.train()
        out = model(video)
        loss = criterion(out, target)
        loss.backward()
        for head in model:
            head_importance = head.grad * head
    '''
    num_videos = args.dataset_pkl_path.split('Max')[1].split('.')[0]
    if not len(args.target_class_idxs) > 1:
        results_path = 'results/{}/HeadImportance_{}Masks.pkl'.format(args.exp_name, args.num_masks)
    else:
        results_path = 'results/{}/HeadImportance_{}Masks_{}.pkl'.format(args.exp_name, args.num_masks, '_'.join([str(x) for x in sorted(args.target_class_idxs)]))

    if args.use_saved_results and not args.recompute_performance_curves:
        if os.path.exists(results_path):
            print('Loading results from {}'.format(results_path))
            with open(results_path, 'rb') as f:
                all_results = pickle.load(f)

            readable_ranking = [('head {}'.format(x//12), 'layer {}'.format(x%12)) for x in all_results['head_importance_list']]
            # save as csv
            with open('evaluation/head_importance/HeadRanking_{}_{}Masks_{}Vids.csv'.format(args.model, args.num_masks, num_videos), 'w') as f:
                for item in readable_ranking:
                    f.write("{}, {}\n".format(item[0], item[1]))
            # plot results
            results = {
                'most_to_least': all_results['most_to_least'],
                'least_to_most': all_results['least_to_most'],
                'random': all_results['random'],
            }
            plot_results(args, results, num_videos)
            exit()
        else:
            print('No results found at {}'.format(results_path))


    # load vcd with pickle
    print('Loading dataset from {}'.format(args.dataset_pkl_path))
    with open(args.dataset_pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    if 'ssv2' in args.dataset_pkl_path:
        ssv2_label_path = '/data/ssv2/something-something-v2-labels.json'
        with open(ssv2_label_path, 'r') as f:
            ssv2_labels = json.load(f)

            # set up multiclass setting
        if len(args.target_class_idxs) > 1:
            multiclass = True
            target_class_idx = '_'.join([str(x) for x in sorted(args.target_class_idxs)])
            # load vcd to get labels
            vcd_path = os.path.join('results', args.exp_name, 'vcd', 'vcd.pkl')
            print('Loading VCD from {}...'.format(vcd_path))
            vcd = pickle.load(open(vcd_path, 'rb'))

        else:
            target_class_name = args.target_class
            target_class_idx = int(ssv2_labels[target_class_name])
            multiclass = False
        # if vcd.multiclass:
        #     target_class_idx = vcd.args.target_class_idxs
        # else:
        #     target_class_name = path_to_dataset.split('v1_')[-1].split('_Max')[0].replace('_', ' ')
        #     target_class_idx = int(ssv2_labels[target_class_name])
    else:
        target_class_idx = None
        multiclass = False
        vcd = None

    num_videos = dataset['pv_rgb_tf'].shape[0] if 'kub' in args.dataset_pkl_path else dataset.shape[0]
    if not args.recompute_performance_curves:
        head_importances = []
        baseline_performances = []
        print('Computing importance of heads with {} masks'.format(args.num_masks))
        for video_idx in tqdm(range(num_videos)):
            if args.debug:
                if video_idx == 2:
                    break
            # load model
            model = load_model(args)
            if 'timesformer' in args.model:
                output_mask, output_flags, target_mask, features, model_retval = tcow_timesformer_forward(dataset, model, video_idx, keep_all=True)
                model_retval = {
                    'output_mask': output_mask.unsqueeze(0),
                    'target_mask': target_mask.unsqueeze(0)}
                metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,source_name='kubric')
                # put results from cuda to cpu
                metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                baseline_performance = metrics_retval['mean_snitch_iou'].item()
                baseline_performances.append(baseline_performance)
            elif 'vidmae' in args.model or 'mme' in args.model:
                video = dataset[video_idx]
                video = video.permute(1, 0, 2, 3)
                # normalize video with imagenet stats
                video = torch.stack([torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])(vid) for vid in video],
                                    dim=0).permute(1, 0, 2, 3)
                video = video.unsqueeze(0).cuda()
                pred, _ = model(video)
                pred = pred.argmax(dim=1).item()
                if multiclass:
                    baseline_performances.append(1) if pred == vcd.labels[video_idx] else baseline_performances.append(0)
                else:
                    baseline_performances.append(1) if pred == target_class_idx else baseline_performances.append(0)
            else:
                raise NotImplementedError

            rish_heads_removed = []
            performance_list = []

            for i in range(args.num_masks):
                # todo: don't hardcode this
                global_idx_heads_to_remove = random.sample(range(144), int(144/2))
                head_layers = [x % 12 for x in global_idx_heads_to_remove]
                head_heads = [x // 12 for x in global_idx_heads_to_remove]
                # set up hook_dict with all layers and all heads to remove
                all_layer_hook_dict = {}
                # calculate which head to remove (12 heads, 12 layers)
                for layer in range(12):
                    heads_to_remove = torch.tensor(head_heads)[torch.where(torch.tensor(head_layers) == layer)]

                    all_layer_hook_dict[layer] = {
                        'heads_to_remove': heads_to_remove,
                            }

                model = load_model(args, hook=remove_heads, hook_layer=list(range(12)), model=model, hook_dict=all_layer_hook_dict)
                if 'timesformer' in args.model:
                    output_mask, output_flags, target_mask, features, model_retval = tcow_timesformer_forward(dataset,
                                                                                                              model,
                                                                                                              video_idx,
                                                                                                              keep_all=False)
                    model_retval = {
                        'output_mask': output_mask.unsqueeze(0),
                        'target_mask': target_mask.unsqueeze(0)}
                    metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                  source_name='kubric')
                    # put results from cuda to cpu
                    metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                    new_performance = metrics_retval['mean_snitch_iou'].item()
                elif 'vidmae' in args.model or 'mme' in args.model:
                    video = dataset[video_idx]
                    video = video.permute(1, 0, 2, 3)
                    # normalize video with imagenet stats
                    video = torch.stack([torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(vid) for vid in video],
                                        dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    pred, _ = model(video)
                    if multiclass:
                        new_performance = pred[0][vcd.args.target_class_idxs].cpu().detach()
                    else:
                        pred = pred.argmax(dim=1).item()
                        new_performance = 1.0 if pred == target_class_idx else 0.0

                else:
                    raise NotImplementedError

                # convert global_idx_heads_to_remove to 0s and 1s indicating which heads to remove
                bool_heads_to_remove = torch.zeros(144)
                bool_heads_to_remove[global_idx_heads_to_remove] = 1
                bool_heads_kept = 1 - bool_heads_to_remove
                rish_heads_removed.append(bool_heads_kept)
                performance_list.append(new_performance)
            # compute rish importance through a weighted sum of performance and the heads removed to achieve that performance (higher performance * heads kept means more important!)
            rish_heads_removed = torch.stack(rish_heads_removed)
            if vcd.multiclass:
                performance_list = torch.stack(performance_list).sum(dim=1)
            else:
                performance_list = torch.tensor(performance_list)
            head_importance = performance_list @ rish_heads_removed
            head_importances.append(head_importance)
        # average over all videos
        head_importance =  torch.stack(head_importances).mean(0)
        baseline_performance = np.mean(baseline_performances)
        MostLeastHeadImportance = head_importance.argsort(descending=True)
        LeastMostHeadImportance = head_importance.argsort()
    else:
        print('Loading results from {}'.format(results_path))
        with open(results_path, 'rb') as f:
            all_results = pickle.load(f)
        MostLeastHeadImportance = torch.tensor(all_results['head_importance_list']).argsort()
        LeastMostHeadImportance = torch.tensor(all_results['head_importance_list']).argsort(descending=True)
        baseline_performance = all_results['most_to_least'][0]
        model = load_model(args)

    if args.full_validation:
        dataset, labels = get_validation_set(args, model=model, multiclass=multiclass)
        vcd.labels = labels

    # compute attribution when removing heads in order from head_importance
    print('1/3: most to least important')
    # most_to_least
    MostLeastResults = head_removal_performance_curve(args, model, dataset, MostLeastHeadImportance, target_class_idx, multiclass, vcd)
    print('2/3: least to most important')
    # least_to_most
    LeastMostResults = head_removal_performance_curve(args, model, dataset, LeastMostHeadImportance, target_class_idx, multiclass, vcd)
    print('3/3: random')
    # random
    RandomHeadImportance = torch.randperm(144)
    RandomResults = head_removal_performance_curve(args, model, dataset, RandomHeadImportance, target_class_idx, multiclass, vcd)

    # add baseline performance to results
    MostLeastResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), MostLeastResults])
    LeastMostResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), LeastMostResults])
    RandomResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), RandomResults])

    if not args.full_validation:
        # convert head_importance to dict where the keys are the layer and the values are the head importances
        head_importance_dict = {}
        for i in range(12):
            head_importance_dict[i] = head_importance[i*12:(i+1)*12]
        head_importance_list = list(x.item() for x in MostLeastHeadImportance)
        results = {
            'head_importance': head_importance,
            'head_importance_dict': head_importance_dict,
            'head_importance_list': head_importance_list,
            'most_to_least': MostLeastResults,
            'least_to_most': LeastMostResults,
            'random': RandomResults,
        }
        # save results
        with open(results_path, 'wb') as f:
            print('Saving results to {}'.format(results_path))
            pickle.dump(results, f)
        # plot results
        results = {
            'most_to_least': results['most_to_least'],
            'least_to_most': results['least_to_most'],
            'random': results['random'],
        }
    else:


        head_importance_dict = {}
        for i in range(12):
            head_importance_dict[i] = head_importance[i*12:(i+1)*12]
        head_importance_list = list(x.item() for x in MostLeastHeadImportance)
        results = {
            'head_importance': head_importance,
            'head_importance_dict': head_importance_dict,
            'head_importance_list': head_importance_list,
            'most_to_least': MostLeastResults,
            'least_to_most': LeastMostResults,
            'random': RandomResults,
        }
        # save results
        with open(results_path, 'wb') as f:
            print('Saving results to {}'.format(results_path))
            pickle.dump(results, f)

        results = {
            'most_to_least': MostLeastResults,
            'least_to_most': LeastMostResults,
            'random': RandomResults,
        }
    plot_results(args, results, num_videos)

def get_validation_set(args, model, multiclass=False, frame_width = 224, frame_height = 224):
    # get class names
    label_path = os.path.join(args.ssv2_path, 'something-something-v2-labels.json')

    with open(label_path, 'r') as f:
        label_dict = json.load(f)
    idx_to_label = {v: k for k, v in label_dict.items()}
    if multiclass:
        cls_idx = [x for i, x in enumerate(label_dict) if i in args.target_class_idxs]
    else:
        cls_idx = label_dict[args.target_class]
        target_label_id = int(cls_idx)
    # open validation file
    data_path = os.path.join(args.ssv2_path, 'something-something-v2-validation.json')
    with open(data_path, 'r') as f:
        data_dict = json.load(f)

    # get videos for target class
    video_ids = []
    video_labels = []
    if multiclass:
        for idx in args.target_class_idxs:
            target_class = idx_to_label[str(idx)]
            video_ids += [x['id'] for x in data_dict if
                          x['template'].replace('[something]', 'something').replace('[something in it]',
                                                                                    'something in it') == target_class]
            video_labels += [idx for x in data_dict if
                             x['template'].replace('[something]', 'something').replace('[something in it]',
                                                                                       'something in it') == target_class]
    else:
        video_ids = [x['id'] for x in data_dict if
                     x['template'].replace('[something]', 'something') == args.target_class]
        video_labels = [cls_idx for x in data_dict if
                        x['template'].replace('[something]', 'something') == args.target_class]
    videos = [('{}/20bn-something-something-v2/{}.webm'.format(args.ssv2_path, video_ids[x]), video_labels[x]) for
              x in range(len(video_ids))]

    random.shuffle(videos)
    dataset = []
    labels = []
    for vid_num, data in enumerate(videos):
        video = data[0]
        label = data[1]

        try:
            vr = VideoReader(video, num_threads=1, ctx=cpu(0), width=frame_width, height=frame_height)
        except:
            continue
        # a file like object works as well, for in-memory decoding

        # 1. the simplest way is to directly access frames
        frames = []
        for i in range(len(vr)):
            # the video reader will handle seeking and skipping in the most efficient manner
            frame = vr[i]
            try:
                frames.append(torch.tensor(frame.asnumpy()))
            except:
                frames.append(torch.tensor(frame))

        # sample frames every model.args.sampling_rate frames
        frames = frames[::model.sampling_rate]
        if len(frames) < model.num_frames:
            continue
        frames = frames[:model.num_frames]

        rgb_video = torch.stack(frames).permute(3, 0, 1, 2) / 255.0
        dataset.append(rgb_video)
        labels.append(label)

    # dataset = [post_resize_smooth(video) for video in dataset]
    dataset = torch.stack(dataset, dim=0)  # n x c x t x h x w
    return dataset, labels

def plot_results(args, results, num_videos):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('dark_background')
    colors = sns.color_palette("husl", 3)

    for i, (key, result) in enumerate(results.items()):
        frac_heads_removed = [i / len(result) for i in range(len(result))]
        auc = np.trapz(result, frac_heads_removed)
        plt.plot(frac_heads_removed, result, color=colors[i], label=key + ' (AUC: {:.2f})'.format(auc))

    plt.xlabel('Heads removed')
    plt.ylabel('Snitch mIoU' if 'timesformer' in args.model else 'Acc')
    plt.legend()
    plt.title('Head removal performance curve ({} Videos - {} Masks)'.format(num_videos, args.num_masks))
    plt.savefig('results/{}/AttributionCurve_{}Masks.png'.format(args.exp_name, args.num_masks))
    # plt.show()

def head_removal_performance_curve(args, model, dataset, head_importance, target_class_idx=None, multiclass=False, vcd=None):
    HeadsRemovedEachStep = args.heads_removed_each_step
    NumRemovedStep = (144 // HeadsRemovedEachStep) + 1
    num_videos = dataset['pv_rgb_tf'].shape[0] if 'kub' in args.dataset_pkl_path else dataset.shape[0]
    PerVideoResults = []
    for video_idx in tqdm(range(num_videos)):
        # compute attribution when removing heads
        if args.debug:
            if video_idx == 2:
                break
        all_layer_hook_dict = {}
        PerStepResults = []
        for i in range(NumRemovedStep):
            IdxHeadRemove = head_importance[i * HeadsRemovedEachStep: (i + 1) * HeadsRemovedEachStep]
            HeadLayers = [x % 12 for x in IdxHeadRemove]
            HeadHeads = [x // 12 for x in IdxHeadRemove]
            for layer in range(12):
                heads_to_remove = torch.tensor(HeadHeads)[torch.where(torch.tensor(HeadLayers) == layer)]
                if i == 0:
                    all_layer_hook_dict[layer] = {'heads_to_remove': heads_to_remove}
                else:
                    all_layer_hook_dict[layer] = {'heads_to_remove': torch.cat([all_layer_hook_dict[layer]['heads_to_remove'],heads_to_remove])}

            model = load_model(args, hook=remove_heads, hook_layer=list(range(12)), model=model,
                               hook_dict=all_layer_hook_dict)
            if 'timesformer' in args.model:
                output_mask, output_flags, target_mask, features, model_retval = tcow_timesformer_forward(dataset,
                                                                                                          model,
                                                                                                          video_idx,
                                                                                                          keep_all=False)
                model_retval = {
                    'output_mask': output_mask.unsqueeze(0),
                    'target_mask': target_mask.unsqueeze(0)}
                metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                              source_name='kubric')
                # put results from cuda to cpu
                metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                new_performance = metrics_retval['mean_snitch_iou'].item()
            elif 'vidmae' in args.model or 'mme' in args.model:
                video = dataset[video_idx]
                video = video.permute(1, 0, 2, 3)
                # normalize video with imagenet stats
                video = torch.stack([torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])(vid) for vid in video],
                                    dim=0).permute(1, 0, 2, 3)
                video = video.unsqueeze(0).cuda()
                pred, _ = model(video)
                pred = pred.argmax(dim=1).item()
                if multiclass:
                    new_performance = 1.0 if pred == vcd.labels[video_idx] else 0.0
                else:
                    new_performance = 1.0 if pred == target_class_idx else 0.0
            else:
                raise NotImplementedError
            PerStepResults.append(new_performance)
        PerVideoResults.append(np.array(PerStepResults))

    # average over all videos
    PerVideoResults = np.stack(PerVideoResults).mean(0)
    return PerVideoResults


def remove_heads(module, input, output):
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

    heads_to_remove = module.hook_dict['heads_to_remove']

    # temporal shape = 300,12,30,64
    # spatial shape = 30,12,301,64
    B, N, C = input[0].shape

    # rearrange to get head dimension
    qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)

    # remove heads
    qkv[:,:,heads_to_remove,:,:] = 0

    # rearrange back to original shape
    output = qkv.permute(1, 3, 0, 2, 4).reshape(B, N, -1)

    return output
    # if vcd.args.cluster_subject == 'tokens':
    #     feature = rearrange(output[0], 't s (nh m) -> t s nh m', m=768 // 12, nh=12)
    # elif vcd.args.cluster_subject == 'block_token':
    #     feature = output[0]
    # elif vcd.args.cluster_subject in ['queries', 'keys', 'values']:
    #     qkv = output.reshape(B, N, 3, 12, C // 12).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]
    #     if vcd.args.cluster_subject == 'keys':
    #         feature = k
    #     elif vcd.args.cluster_subject == 'values':
    #         feature = v
    #     elif vcd.args.cluster_subject == 'queries':
    #         feature = q
    #     else:
    #         raise NotImplementedError
    # else:
    #     raise NotImplementedError

    # current method
    # hard code shape of tensor between models... (not very nice, should use module attribute)
    # if output[0].shape[1] == 1568 or output[0].shape[0] == 1568:
    #     masks = F.interpolate(masks.unsqueeze(0).type(torch.uint8), size=(8, 14, 14), mode='nearest').squeeze(0).cuda().float()
    # else:
    #     masks = F.interpolate(masks.unsqueeze(0).type(torch.uint8), size=(30, 15, 20), mode='nearest').squeeze(0).cuda().float()
    # # masks = rearrange(masks, 'b t h w -> b (t h w)')
    # masks = rearrange(masks, 'b t h w -> b (h w t)')
    # G = torch.tensor(vcd.dic[layer][head]['cnmf']['G'])
    # G_curr = G[min_max_idx[0]:min_max_idx[1] + 1]  # .softmax(1)
    # # scale G_curr between 0-1 over concept dimension (i.e., to enforce that removing all concepts = 0)
    # G_curr = (G_curr - G_curr.min(1)[0].unsqueeze(1)) / (G_curr.max(1)[0] - G_curr.min(1)[0]).unsqueeze(1)
    # G_inv = 1 - G_curr[:, concept_idx].cuda().float()
    # # replace masks with values of G_inv
    # mask_weight = (masks.T.float() @ G_inv.float()).T
    # # feature_copy = feature.clone()
    # for i in range(mask_weight.shape[0]):
    #     # multiple feature with mask_weight
    #     # deal with cls token
    #     if vcd.args.cluster_subject == 'block_token':
    #         if mask_weight.shape[1] == feature.shape[1]:
    #             feature[0] = torch.mul(feature[0].T, mask_weight[i]).T
    #         else:
    #             feature[0, 1:] = torch.mul(feature[0, 1:].T, mask_weight[i]).T
    #     else:
    #
    #         if feature[0,head].T.shape[1] == mask_weight[i].shape[0]:
    #             feature[0,head] = torch.mul(feature[0,head].T, mask_weight[i]).T
    #         else:
    #             new_feat = rearrange(feature[:, :, 1:], '(b t) n (h w) c -> b n (h w t) c', t=30, h=15, w=20)
    #             return_feat = torch.mul(new_feat[0,head].T, mask_weight[i]).T
    #             return_feat = rearrange(return_feat.unsqueeze(0), 'b (h w t) c -> (b t) (h w) c', t=30, h=15, w=20)
    #             feature[:, head, 1:] = return_feat
    #
    # if vcd.args.cluster_subject == 'block_token':
    #     return feature, output[1]
    # else:
    #     if vcd.args.cluster_subject == 'keys':
    #         qkv = torch.stack([q, feature, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
    #     elif vcd.args.cluster_subject == 'values':
    #         qkv = torch.stack([q, k, feature]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
    #     elif vcd.args.cluster_subject == 'queries':
    #         qkv = torch.stack([feature, k, v]).permute(1, 3, 0, 2, 4).reshape(B, N, -1)
    #     return qkv

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

            # debug - visualize the output of the model
            # t = 13
            # plt.imshow(seeker_input[0][:, t].permute(1, 2, 0).cpu());plt.show()
            # plt.imshow(output_mask[0].sigmoid()[0][t].cpu());plt.show()

        model_retval = {}
        # all_target_flags.append(target_flags)  # (B, T, 3).
        # target_flags = torch.stack([target_flags], dim=1)  # (B, Qs, T, 3).
        model_retval['target_flags'] = torch.stack([target_flags], dim=1).cuda()  # (B, Qs, T, 3).

        # snitch_occl_by_ptr = torch.stack([snitch_occl_by_ptr], dim=1)  # (B, Qs, 1, T, Hf, Wf).
        model_retval['snitch_occl_by_ptr'] = torch.stack([snitch_occl_by_ptr], dim=1).cuda()

        cur_occl_fracs = occl_fracs[:, query_idx, :, :].diagonal(0, 0, 1)
        cur_occl_fracs = rearrange(cur_occl_fracs, 'T V B -> B T V')  # (B, T, 3).
        sel_occl_fracs = torch.stack([cur_occl_fracs], dim=1)  # (B, Qs, T, 3).
        model_retval['sel_occl_fracs'] = sel_occl_fracs.cuda()  # (B, Qs, T, 3).

        return output_mask, output_flags, target_mask, features, model_retval

def vcd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name' ,default='videomae_ssv2_keys_spilling_6class_fgc_train60', type=str, help='experiment name (used for saving)')

    # general
    parser.add_argument('--dataset', default='ssv2', type=str,help='dataset to use')
    # parser.add_argument('--dataset_pkl_path', default='/data/ssv2/v1_Closing_something_Max30.pkl', type=str,help='dataset to use')
    # parser.add_argument('--dataset_pkl_path', default='/data/ssv2/v1_Rolling_something_on_a_flat_surface_Max30.pkl', type=str,help='dataset to use')
    parser.add_argument('--dataset_pkl_path', default='/data/ssv2/v1_60_136_137_138_159_163_Max60_train.pkl', type=str,help='dataset to use')
    # parser.add_argument('--dataset_pkl_path', default='/data/kubcon_v10/val/v1_Max30.pkl', type=str,help='dataset to use')
    # parser.add_argument('--dataset', default='kinetics', type=str,help='dataset to use')
    # parser.add_argument('--dataset', default='ssv2', type=str,help='dataset to use')
    parser.add_argument('--kubric_path', default='/data/kubcon_v10', type=str,help='kubric path')
    parser.add_argument('--uvo_path', default='/home/matthewkowal/data/uvo', type=str,help='kubric path')
    parser.add_argument('--ssv2_path', default='/data/ssv2', type=str,help='kubric path')
    parser.add_argument('--kinetics_path', default='/data/kinetics400', type=str,help='kubric path')
    # parser.add_argument('--target_class', default='archery', type=str,help='target class name for classification dataset')
    parser.add_argument('--target_class', default='Dropping something into something', type=str,help='target class name for classification dataset')
    parser.add_argument('--custom_path', default='data/sample', type=str,help='path to custom dataset')
    parser.add_argument('--force_reload_videos', action='store_true',help='Maximum number of videos to use during clustering.')
    parser.add_argument('--cache_name', default='v1', type=str,help='experiment name (used for saving)')
    parser.add_argument('--fig_save_name', default='attribution_plot',help='figure name (used for saving)')
    parser.add_argument('--compare_multiclass', action='store_true', help='if true, average over multiple classes and then compare ')
    parser.add_argument('--overwrite_importance', action='store_true', help='Overwrite importance results')
    parser.add_argument('--overwrite_attribution', action='store_true', help='Overwrite addtribution results')
    parser.add_argument('--results_name', default='', type=str,help='figure name (used for saving)')
    parser.add_argument('--use_saved_results', action='store_true', help='Use saved results.')

    # model
    # parser.add_argument('--model', default='timesformer_occ_v1', type=str,help='Model to run.')
    parser.add_argument('--model', default='vidmae_ssv2_ft', type=str,help='Model to run.')
    parser.add_argument('--checkpoint_path', default='', type=str,help='Override checkpoint path.')
    parser.add_argument('--concept_clustering', action='store_true', help='Flag to perform concept clustering.')
    parser.add_argument('--cluster_layer', nargs='+', default=[], type=int,help='Layers to perform clustering at (timseformer: 0-11 / aot: 0-3).')
    parser.add_argument('--cluster_subject', default='tokens', type=str,help='Subject to cluster)', choices=['block_token', 'keys', 'values', 'queries', 'tokens', 'attn', 'attn_caus', 'attn_sft'])
    parser.add_argument('--cluster_memory', default='curr', type=str,help='Subject to cluster)', choices=['tokens', 'curr', 'long', 'short'])
    parser.add_argument('--use_temporal_attn', action='store_true', help='Flag to use temporal feature maps for timesformer.')
    parser.add_argument('--attn_head', nargs='+', default=[], type=int, help='Which heads to use to cluster attention maps (-1 is mean | use 0 if using entire feature).')
    parser.add_argument('--target_class_idxs', nargs='+', default=[60,136,137,138,159,163], type=int,help='target class idx for multiple target class setting')


    # concept importance
    parser.add_argument('--removal_type', default='rish', help='type of attribution removal to do. [perlay | alllay | alllayhead || rish | gradient]')
    parser.add_argument('--num_masks', default=25, type=int, help='Number of masks to forward pass during random head removal for RISH.')
    parser.add_argument('--heads_removed_each_step', default=10, type=int, help='Number of passes during random head removal for RISH.')
    parser.add_argument('--random_importance', action='store_true', help='Use random concept importance.')
    parser.add_argument('--baseline_compare', action='store_true', help='Compare with random and inverse baselines.')
    parser.add_argument('--importance_loss', default='track', type=str,help='Loss to use for importance [track | occl_pct].')
    parser.add_argument('--recompute_performance_curves', action='store_true', help='Load results but recompute performance curves.')
    parser.add_argument('--full_validation', action='store_true', help='Use full validation set during AUC curve removal.')

    # attribution settings
    parser.add_argument('--attribution_evaluation_metric', nargs='+', default=['mean_snitch_iou'], type=str, help='Metrics to use during attribution calculation.')
    parser.add_argument('--zero_features', action='store_true', help='Zero out all other features during attribution.')

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