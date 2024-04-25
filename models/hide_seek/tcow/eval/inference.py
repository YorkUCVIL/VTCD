'''
Evaluation tools.
Created by Basile Van Hoorick, Jun 2022.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'eval/'))
sys.path.insert(0, os.getcwd())

from models.hide_seek.tcow.__init__ import *

# Internal imports.
import models.hide_seek.tcow.utils.my_utils as my_utils
import models.hide_seek.tcow.seeker.seeker as seeker
import models.hide_seek.tcow.pipeline as pipeline


def load_networks(checkpoint_path, device, logger, epoch=-1, args=None, random=False):
    '''
    :param checkpoint_path (str): Path to model checkpoint folder or file.
    :param epoch (int): If >= 0, desired checkpoint epoch to load.
    :return (networks, train_args, dset_args, model_args, epoch).
        networks (dict).
        train_args (dict).
        train_dset_args (dict).
        model_args (dict).
        epoch (int).
    '''
    print_fn = logger.info if logger is not None else print
    
    # TODX DRY: This overlaps with args.py, and the passed value is always a file.
    if os.path.isdir(checkpoint_path):
        model_fn = f'model_{epoch}.pth' if epoch >= 0 else 'checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_path, model_fn)
    # print_fn('Loading weights from: ' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint['seeker_args']['aot_pretrain_path'] = ''

    # Load all arguments for later use.
    train_args = checkpoint['train_args']
    train_dset_args = checkpoint['dset_args']

    # Get network instance parameters.
    seeker_args = checkpoint['seeker_args']
    model_args = {'seeker': seeker_args}

    # Fix outdated arguments from older checkpoints.
    # (For dset_args, it is the responsibility of data.py)
    if 'occl_cont_zero_weight' not in train_args:
        train_args.occl_cont_zero_weight = 0.02
    if 'hard_negative_factor' not in train_args:
        train_args.hard_negative_factor = 3.0
    if 'xray_query' not in train_args:
        train_args.xray_query = False
    if 'annot_visible_pxl_only' not in train_args:
        train_args.annot_visible_pxl_only = False
    if 'is_figs' not in train_args:
        train_args.is_figs = False
    if 'flag_channels' not in seeker_args:
        seeker_args['flag_channels'] = 0
    if 'aot_max_gap' not in seeker_args:
        seeker_args['aot_max_gap'] = 1
    if 'aot_max_gap' not in seeker_args:
        seeker_args['aot_max_gap'] = 1
    if 'aot_max_gap' not in seeker_args:
        seeker_args['aot_max_gap'] = 1
    seeker_args['concept_clustering'] = args.concept_clustering
    seeker_args['cluster_layer'] = args.cluster_layer
    seeker_args['cluster_subject'] = args.cluster_subject
    # seeker_args['cluster_memory'] = args.cluster_memory

    # Instantiate networks.
    seeker_net = seeker.Seeker(logger, **seeker_args)
    seeker_net = seeker_net.to(device)
    if not random:
        print('Loading weights from: ' + checkpoint_path)
        seeker_net.load_state_dict(checkpoint['net_seeker'])
    networks = {'seeker': seeker_net}
    epoch = checkpoint['epoch']
    # print_fn('=> Loaded epoch (1-based): ' + str(epoch + 1))

    # else:
    #     perfect_net = seeker.PerfectBaseline(logger, perfect_baseline, **seeker_args)
    #     networks = {'perfect': perfect_net}
    #     model_args['perfect'] = {'which_baseline:', perfect_baseline}
    #     epoch = -2
    #     print_fn(f'=> Loaded perfect baseline: {perfect_baseline}')

    return (networks, train_args, train_dset_args, model_args, epoch)


def perform_inference(data_retval, networks, device, logger, all_args, cur_step):
                    #   **pipeline_args):
    '''
    Generates test time predictions.
    :param data_retval (dict): Data loader element.
    :param all_args (dict): train, test, train_dset, test_dset, model.
    '''
    # Following DRY, prepare pipeline instance, *BUT* take care of shared args by updating them.
    used_args = copy.deepcopy(all_args['train'])
    used_args.num_queries = all_args['test'].num_queries

    my_pipeline = pipeline.MyTrainPipeline(used_args, logger, networks, device)
    my_pipeline.set_phase('test')  # This calls eval() on all submodules.

    perfect_baseline = all_args['test'].perfect_baseline
    include_loss = not(all_args['test'].for_stats)
    metrics_only = (data_retval['source_name'][0] == 'plugin' or perfect_baseline)
    no_pred = all_args['test'].for_stats

    # Communicate arguments from test options to modules.
    if all_args['train'].tracker_arch == 'aot' and all_args['test'].aot_eval_size_fix:
        logger.warning('Enabling AOT eval_size_fix such that model input shapes are (577, 1041)!')
        networks['seeker'].seeker.eval_size_fix = True

    if used_args.tracker_arch == 'timesformer':
        my_pipeline.networks['seeker'].seeker.tracker_backbone.cluster_layer = all_args['test'].cluster_layer
        my_pipeline.networks['seeker'].seeker.tracker_backbone.cluster_subject = all_args['test'].cluster_subject


    temp_st = time.time()  # DEBUG
    (model_retval, loss_retval, cluster_viz) = my_pipeline(
    # (model_retval, loss_retval) = my_pipeline(
        data_retval, cur_step, cur_step, 0, 1.0, include_loss, metrics_only, perfect_baseline,
        no_pred)
    logger.debug(f'(perform_inference) my_pipeline: {time.time() - temp_st:.3f}s')  # DEBUG

    # Calculate various evaluation metrics.
    loss_retval = my_pipeline.process_entire_batch(
        data_retval, model_retval, loss_retval, cur_step, cur_step, 0, 1.0) \
            if loss_retval is not None else None

    # (B, Q, T) = model_retval['output_traject'].shape[:3]
    # (H, W) = data_retval['kubric_retval']['pv_rgb_tf'].shape[-2:]
    
    # assert B == 1
    # metric_retval = defaultdict(list)

    # if 'kubric_retval_others' in data_retval:
    #     # TODX uncertainty evaluation
    #     pass

    # TODX update in light of other seeker_output_type
    # ^ see inference_old.py

    # Organize and return relevant info, moving stuff to CPU and/or converting to numpy as needed.
    inference_retval = dict()
    inference_retval['model_retval'] = model_retval
    inference_retval['loss_retval'] = loss_retval
    inference_retval = my_utils.dict_to_cpu(inference_retval)

    return inference_retval, cluster_viz
