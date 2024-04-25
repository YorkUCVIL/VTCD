
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
from einops import rearrange
sys.path.append('/home/ubuntu/VTCD/')
from utilities.utils import load_model
from models.hide_seek.tcow.eval.metrics import calculate_metrics_mask_track
from evaluation.importance_utils import *
import torch.nn.functional as F
import torchvision
import models.hide_seek.tcow as tcow
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
def main(args):
    '''
    Script to prune heads from models while using the full dataset to rank heads, prune them, and evaluate.
    '''

    if not os.path.exists('results/{}'.format(args.exp_name)):
        os.makedirs('results/{}'.format(args.exp_name))

    results_path = 'results/{}/{}_{}Epoch_{}Masks_{}.pkl'.format(args.exp_name, args.model, args.epochs, args.num_masks, '_'.join([str(x) for x in sorted(args.target_class_idxs)]))
    print('Results will be saved to {}'.format(results_path))
    if args.use_saved_results and not args.recompute_performance_curves:
        if os.path.exists(results_path):
            print('Loading results from {}'.format(results_path))
            with open(results_path, 'rb') as f:
                all_results = pickle.load(f)

            readable_ranking = [('layer {}'.format(x%12),'head {}'.format(x//12)) for x in all_results['head_importance_list']]
            # plot results
            results = {
                'most_to_least': all_results['most_to_least'],
                'least_to_most': all_results['least_to_most'],
                'random': all_results['random'],
            }
            plot_results(args, results)
            exit()
        else:
            print('No results found at {}'.format(results_path))

    # loading model
    model = load_model(args)

    # load vcd with pickle
    print('Loading dataloader...')

    if args.dataset == 'ssv2':
        train_dataset = SSV2(args, model, 'train')
        validation_dataset = SSV2(args, model, 'validation')
    else:
        raise NotImplementedError
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if not args.recompute_performance_curves:
        head_importances = []
        for epoch in range(args.epochs):
            print('Epoch {}/{}'.format(epoch+1, args.epochs))
            # iterate through all videos in dataset (either for all classes or a subset) while masking heads
            for batch_idx, data in enumerate(tqdm(train_loader)):
                if args.debug:
                    if batch_idx > 5:
                        break

                video = data[0].cuda()
                target = torch.tensor([int(x) for x in data[1]])


                # forward pass with all heads masked
                rish_heads_removed = []
                performance_list = []
                for i in range(args.num_masks):

                    global_idx_heads_to_remove = random.sample(range(144), int(144 / 2))
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

                    model = load_model(args, hook=remove_heads, hook_layer=list(range(12)), model=model,
                                       hook_dict=all_layer_hook_dict)
                    if 'vidmae' in args.model:
                        pred, _ = model(video)
                        pred = pred.argmax(dim=1).cpu().detach()
                        performance = (pred == target).sum().item() / len(pred)
                    else:
                        raise NotImplementedError

                    bool_heads_to_remove = torch.zeros(144)
                    bool_heads_to_remove[global_idx_heads_to_remove] = 1
                    bool_heads_kept = 1 - bool_heads_to_remove
                    rish_heads_removed.append(bool_heads_kept)
                    performance_list.append(performance)

                rish_heads_removed = torch.stack(rish_heads_removed)
                performance_list = torch.tensor(performance_list)
                head_importance = performance_list @ rish_heads_removed
                head_importances.append(head_importance)
        # average over all videos
        head_importance = torch.stack(head_importances).mean(0)
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




    # compute attribution when removing heads in order from head_importance

    # load model to get rid of hooks
    model = load_model(args)

    # baseline performance
    print('0/3: baseline performance')
    baseline_performance = single_validation_epoch(args, model, validation_loader)

    print('1/3: most to least important')
    # most_to_least
    MostLeastResults = head_removal_performance_curve(args, model, validation_loader, MostLeastHeadImportance)
    print('2/3: least to most important')
    # least_to_most
    LeastMostResults = head_removal_performance_curve(args, model, validation_loader, LeastMostHeadImportance)
    print('3/3: random')
    # random
    RandomHeadImportance = torch.randperm(144)
    RandomResults = head_removal_performance_curve(args, model, validation_loader, RandomHeadImportance)

    # add baseline performance to results
    MostLeastResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), MostLeastResults])
    LeastMostResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), LeastMostResults])
    RandomResults = np.concatenate([np.expand_dims(baseline_performance, axis=0), RandomResults])

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

    results = {
        'most_to_least': MostLeastResults,
        'least_to_most': LeastMostResults,
        'random': RandomResults,
    }


    plot_results(args, results)


class SSV2(Dataset):
    def __init__(self, args, model, split='train', multiclass=False, frame_width = 224, frame_height = 224):
        self.args = args
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sampling_rate = model.sampling_rate
        self.num_frames = model.num_frames
        # get class names
        label_path = os.path.join(args.ssv2_path, 'something-something-v2-labels.json')

        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        idx_to_label = {v: k for k, v in label_dict.items()}
        if not len(args.target_class_idxs) == 0:
            self.cls_idx = [x for i, x in enumerate(label_dict) if i in args.target_class_idxs]

        # open data file
        data_path = os.path.join(args.ssv2_path, 'something-something-v2-{}.json'.format(split))
        with open(data_path, 'r') as f:
            data_dict = json.load(f)

        # get videos for target class
        video_ids = []
        video_labels = []
        if len(args.target_class_idxs) == 0:
            video_ids = [x['id'] for x in data_dict]
            video_labels += [label_dict[x['template'].replace('[', '').replace(']', '')] for x in data_dict]
        else:
            for idx in args.target_class_idxs:
                target_class = idx_to_label[str(idx)]
                video_ids += [x['id'] for x in data_dict if x['template'].replace('[', '').replace(']', '') == target_class]
                video_labels += [idx for x in data_dict if x['template'].replace('[', '').replace(']', '') == target_class]
        self.videos = [('{}/20bn-something-something-v2/{}.webm'.format(args.ssv2_path, video_ids[x]), video_labels[x]) for
                  x in range(len(video_ids))]

        self.transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])


    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        video = self.videos[idx][0]
        label = torch.tensor(int(self.videos[idx][1]))

        vr = VideoReader(video, num_threads=1, ctx=cpu(0), width=self.frame_width, height=self.frame_height)
        # a file like object works as well, for in-memory decoding

        # 1. the simplest way is to directly access frames
        frames = []
        for i in range(len(vr)):
            # the video reader will handle seeking and skipping in the most efficient manner
            frame = vr[i]
            frame = torch.tensor(frame.asnumpy())
            frame = frame.permute(2, 0, 1)/255.0
            frame = self.transform(frame)
            frames.append(frame)

        # sample frames every model.args.sampling_rate frames
        frames = frames[::self.sampling_rate]
        frames = frames[:self.num_frames]

        # if the video is too short, repeat the last frame
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))

        rgb_video = torch.stack(frames).permute(1, 0, 2, 3)
        return rgb_video, torch.tensor(int(label))


def plot_results(args, results):
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
    plt.title('Head removal performance curve ({} Masks)'.format(args.num_masks))
    plt.savefig('results/{}/HeadsFullDataset_AttributionCurve_{}Masks.png'.format(args.exp_name, args.num_masks))
    plt.show()


def single_validation_epoch(args, model, validation_loader):

    model.eval()
    results = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(validation_loader)):
            if args.debug:
                if batch_idx > 5:
                    break

            video = data[0].cuda()
            target = torch.tensor([int(x) for x in data[1]])

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
                pred, _ = model(video)
                pred = pred.argmax(dim=1).cpu().detach()
                performance = (pred == target).sum().item() / len(pred)
            else:
                raise NotImplementedError
            results.append(performance)

    # average over all videos
    final_average_acc = np.mean(results)
    return final_average_acc



def head_removal_performance_curve(args, model, validation_loader, head_importance):
    HeadsRemovedEachStep = args.heads_removed_each_step
    NumRemovedStep = (144 // HeadsRemovedEachStep) + 1
    PerVideoResults = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(validation_loader)):
            if args.debug:
                if batch_idx > 5:
                    break

            video = data[0].cuda()
            target = torch.tensor([int(x) for x in data[1]])

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
                if 'vidmae' in args.model:
                    pred, _ = model(video)
                    pred = pred.argmax(dim=1).cpu().detach()
                    performance = (pred == target).sum().item() / len(pred)
                else:
                    raise NotImplementedError
                PerStepResults.append(performance)
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
    parser.add_argument('--exp_name' ,default='VideoMAE_SSv2_Spilling', type=str, help='experiment name (used for saving)')

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
    # parser.add_argument('--target_class_idxs', nargs='+', default=[], type=int,help='target class idx for multiple target class setting')
    parser.add_argument('--target_class_idxs', nargs='+', default=[60,136,137,138,159,163], type=int,help='target class idx for multiple target class setting')


    # concept importance
    parser.add_argument('--removal_type', default='rish', help='type of attribution removal to do. [perlay | alllay | alllayhead || rish | gradient]')
    parser.add_argument('--num_masks', default=1, type=int, help='Number of masks to forward pass during random head removal for RISH.')
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
    parser.add_argument('--epochs', default=1, type=int,help='')
    parser.add_argument('--num_workers', default=0, type=int,help='')
    parser.add_argument('--batch_size', default=2, type=int,help='')

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

'''
CUDA_VISIBLE_DEVICES=2 python evaluation/prune_heads_full_dataset.py --dataset ssv2 --model vidmae_ssv2_ft --num_masks 10 --epochs 10 --num_workers 16 --batch_size 4
CUDA_VISIBLE_DEVICES=2 python evaluation/prune_heads_full_dataset.py --dataset ssv2 --model vidmae_ssv2_ft --target_class_idxs 60 136 137 138 159 163 --num_masks 10 --epochs 10 --batch_size 4
'''
