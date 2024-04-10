import pickle
import os
import torch
import numpy as np
import decord
from decord import VideoReader, cpu
import models.hide_seek.tcow.loss as tcow_loss
import yaml

def load_model(args, hook=None, hook_layer=None, feature_hat=None, model=None, hook_dict=None, enable_grad=False):
    if 'timesformer' in args.model:
        if model is None:
            from models.hide_seek.tcow.args import test_args
            from models.hide_seek.tcow.eval import inference

            tcow_test_args = test_args()
            tcow_test_args.data_path = args.kubric_path
            tcow_test_args.resume = 'checkpoints/v111/checkpoint.pth'

            if len(args.checkpoint_path) > 0:
                tcow_test_args.resume = args.checkpoint_path

            if args.model == 'timesformer_random':
                tcow_test_args.resume = 'checkpoints/v111/checkpoint.pth'
                (networks, train_args, train_dset_args, model_args, epoch) = \
                    inference.load_networks(tcow_test_args.resume, device='cuda', logger=None, epoch=tcow_test_args.epoch, args=args, random=True)
            else:
                (networks, train_args, train_dset_args, model_args, epoch) = inference.load_networks(tcow_test_args.resume, device='cuda', logger=None, epoch=tcow_test_args.epoch, args=args)

            for (k, v) in networks.items():
                v.eval()
            for (k, v) in networks.items():
                if k == 'seeker':
                    v.set_phase('test')

            model = networks['seeker']
            model.seeker.tracker_backbone.cluster_layer = args.cluster_layer
            model.seeker.tracker_backbone.cluster_subject = args.cluster_subject
            model.seeker.tracker_backbone.use_temporal_attn = args.use_temporal_attn
            model.seeker.tracker_backbone.attn_head = args.attn_head

            model.args = args
            model.train_args = train_args
            model.train_dset_args = train_dset_args
            model.model_args = model_args
            model.test_args = tcow_test_args
            model.num_frames = 30
            model.sampling_rate = 1 # just for rosetta concepts

        if hook is not None:
            # if args.cluster_subject == 'tokens':
            #     model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].register_forward_hook(hook)
            #     if centroids is not None:
            #         model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].centroid = \
            #         centroids[hook_layer]
            # else:
            #     # model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer]._modules['attn']._modules['qkv'].register_forward_hook(hook)

            # remove existing hook if it exists
            if isinstance(hook_layer, list):
                if args.removal_type == 'alllayhead':
                    if args.cluster_subject == 'block_token':
                        # remove all hooks
                        for lay in range(12):
                            if len(model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._forward_hooks) > 0:
                                model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._forward_hooks.clear()
                        # register hook
                        model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].register_forward_hook(hook)
                        model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].hook_dict = hook_dict if hook_dict is not None else None
                    else:
                        for lay in range(12):
                            if len(model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                                model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                        # register hook for all layers
                        for lay in hook_layer:
                            model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv'].register_forward_hook(hook)
                            model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv'].hook_dict = hook_dict[lay] if hook_dict is not None else None
                elif args.removal_type == 'rish':
                    for lay in range(12):
                        if len(model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[
                                   str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                            model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[
                                str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                    # register hook for all layers
                    for lay in hook_layer:
                        model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules[
                            'attn']._modules['qkv'].register_forward_hook(hook)
                        model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules[
                            'attn']._modules['qkv'].hook_dict = hook_dict[lay] if hook_dict is not None else None
                else:
                    if args.cluster_subject == 'block_token':
                        # remove all hooks
                        for lay in range(12):
                            if len(model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._forward_hooks) > 0:
                                model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._forward_hooks.clear()
                        # register hook
                        model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].register_forward_hook(hook)
                        model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].hook_dict = hook_dict if hook_dict is not None else None
                    else:
                        for lay in range(12):
                            if len(model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                                model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                        # register hook for all layers
                        for lay in hook_layer:
                            model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv'].register_forward_hook(hook)
                            model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv'].hook_dict = hook_dict[lay] if hook_dict is not None else None
            else:
                if args.cluster_subject == 'block_token':
                    # remove all hooks
                    for lay in range(12):
                        if len(model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._forward_hooks) > 0:
                            model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._forward_hooks.clear()
                    # register hook
                    model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].register_forward_hook(hook)
                    model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[hook_layer].hook_dict = hook_dict if hook_dict is not None else None
                else:
                    for lay in range(12):
                        if len(model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                            model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                    model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(hook_layer)]._modules['attn']._modules['qkv'].register_forward_hook(hook)
                    model.seeker.tracker_backbone.timesformer.model._modules['blocks']._modules[str(hook_layer)]._modules['attn']._modules['qkv'].hook_dict = hook_dict if hook_dict is not None else None

                # add loss to model if neccessary
                if args.cat_method == 'integrated_gradients':
                    from models.hide_seek.tcow.args import train_args
                    tcow_train_args = train_args()
                    loss = tcow_loss.MyLosses(tcow_train_args, logger=None, phase='val')
                    model.loss = loss
                    model.loss.train_args = tcow_train_args
        torch.set_grad_enabled(enable_grad)
    elif 'vidmae' in args.model:
        if model is None:
            from timm.models import create_model

            if 'pre' in args.model:
                from models.VideoMAE.run_videomae_vis import get_args
                vidmae_args = get_args()
                if 'ssv2' in args.model:
                    model_path = 'checkpoints/videomae_vitb_pretrain_ssv2.pth'
                    vidmae_args.nb_classes = 174
                    vidmae_args.sampling_rate = 2
                vidmae_args.model = 'pretrain_videomae_base_patch16_224'
                model = create_model(
                    vidmae_args.model,
                    pretrained=False,
                    drop_path_rate=vidmae_args.drop_path,
                    drop_block_rate=None,
                    decoder_depth=vidmae_args.decoder_depth
                )
                patch_size = model.encoder.patch_embed.patch_size
                vidmae_args.window_size = (vidmae_args.num_frames // 2, vidmae_args.input_size // patch_size[0],
                                           vidmae_args.input_size // patch_size[1])
                vidmae_args.patch_size = patch_size

                model.args = vidmae_args

                model.encoder.cluster_layer = args.cluster_layer
                model.encoder.cluster_subject = args.cluster_subject
                model.encoder.attn_head = args.attn_head
                model.sampling_rate = model.args.sampling_rate
                model.num_frames = model.args.num_frames
                model.cuda()

                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model'])
            elif 'ft' in args.model:
                from models.VideoMAE.run_class_finetuning import get_args
                from models.VideoMAE.modeling_finetune import vit_base_patch16_224
                vidmae_args = get_args()
                if 'k400' in args.model:
                    model_path = 'checkpoints/videomae_vitb_ft_k400.pth'
                    vidmae_args.nb_classes = 400
                    vidmae_args.sampling_rate = 5
                elif 'ssv2' in args.model:
                    model_path = 'checkpoints/videomae_vitb_ft_ssv2.pth'
                    vidmae_args.nb_classes = 174
                    vidmae_args.sampling_rate = 2
                vidmae_args.model = 'vit_base_patch16_224'
                model = vit_base_patch16_224(
                    pretrained=False,
                    num_classes=vidmae_args.nb_classes,
                    all_frames=vidmae_args.num_frames * vidmae_args.num_segments,
                    tubelet_size=vidmae_args.tubelet_size,
                    fc_drop_rate=vidmae_args.fc_drop_rate,
                    drop_rate=vidmae_args.drop,
                    drop_path_rate=vidmae_args.drop_path,
                    attn_drop_rate=vidmae_args.attn_drop_rate,
                    use_mean_pooling=vidmae_args.use_mean_pooling,
                    init_scale=vidmae_args.init_scale,
                )
                patch_size = (16,16)
                vidmae_args.window_size = (vidmae_args.num_frames // 2, vidmae_args.input_size // patch_size[0],
                                           vidmae_args.input_size // patch_size[1])
                vidmae_args.patch_size = patch_size

                model.args = vidmae_args

                model.cluster_layer = args.cluster_layer
                model.cluster_subject = args.cluster_subject
                model.attn_head = args.attn_head
                model.sampling_rate = model.args.sampling_rate
                model.num_frames = model.args.num_frames

                model.cuda()

                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['module'])
                model.eval()


        if hook is not None:
            # remove existing hook if it exists
            if isinstance(hook_layer, list):
                if args.cluster_subject == 'block_token':
                    # remove all hooks
                    for lay in range(12):
                        if len(model.blocks._modules[str(lay)]._forward_hooks) > 0:
                            model.blocks._modules[str(lay)]._forward_hooks.clear()
                    # register hook
                    model.blocks._modules[str(hook_layer)].register_forward_hook(hook)
                    model.blocks._modules[str(hook_layer)].hook_dict = hook_dict if hook_dict is not None else None
                else:
                    if 'pre' in args.model:
                        for lay in range(12):
                            if len(model.encoder.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                                model.encoder.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                        # register hook for all layers
                        for lay in hook_layer:
                            model.encoder.blocks._modules[str(lay)]._modules['attn']._modules['qkv'].register_forward_hook(hook)
                            model.encoder.blocks._modules[str(lay)]._modules['attn']._modules['qkv'].hook_dict = hook_dict[lay] if hook_dict is not None else None
                    else:
                        for lay in range(12):
                            if len(model.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                                model.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                        # register hook for all layers
                        for lay in hook_layer:
                            model.blocks._modules[str(lay)]._modules['attn']._modules['qkv'].register_forward_hook(hook)
                            model.blocks._modules[str(lay)]._modules['attn']._modules['qkv'].hook_dict = hook_dict[lay] if hook_dict is not None else None
            else:
                if args.cluster_subject == 'block_token':
                    # remove all hooks
                    for lay in range(12):
                        if len(model.blocks._modules[str(lay)]._forward_hooks) > 0:
                            model.blocks._modules[str(lay)]._forward_hooks.clear()
                    # register hook
                    model.blocks._modules[str(hook_layer)].register_forward_hook(hook)
                    model.blocks._modules[str(hook_layer)].hook_dict = hook_dict if hook_dict is not None else None
                else:
                    if 'pre' in args.model:
                        for lay in range(12):
                            if len(model.encoder.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                                model.encoder.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                        model.encoder.blocks._modules[str(hook_layer)]._modules['attn']._modules['qkv'].register_forward_hook(hook)
                        model.encoder.blocks._modules[str(hook_layer)]._modules['attn']._modules['qkv'].hook_dict = hook_dict if hook_dict is not None else None
                    else:
                        for lay in range(12):
                            if len(model.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks) > 0:
                                model.blocks._modules[str(lay)]._modules['attn']._modules['qkv']._forward_hooks.clear()
                        model.blocks._modules[str(hook_layer)]._modules['attn']._modules['qkv'].register_forward_hook(hook)
                        model.blocks._modules[str(hook_layer)]._modules['attn']._modules['qkv'].hook_dict = hook_dict if hook_dict is not None else None


        # load training args if using kubric (i.e., target mask, etc)
        if args.dataset == 'kubric':
            # use baseline occ tracking model
            train_args_ckpt = 'checkpoints/v111/checkpoint.pth'
            train_args = torch.load(train_args_ckpt, map_location='cpu')['train_args']
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
            train_args.query_bias = 'none'

            model.train_args = train_args
    elif 'intern' in args.model:
        if model is None:
            import models.InternVideo.Pretrain.MultiModalitiesPretraining.InternVideo as InternVideo
            from torchvision import transforms
            from models.InternVideo.Pretrain.MultiModalitiesPretraining.InternVideo import video_transform

            model = InternVideo.load_model("checkpoints/InternVideo-MM-B-16.ckpt").cuda()
            model.tokenize = InternVideo.tokenize

            model.visual.transformer.cluster_layer = args.cluster_layer
            model.visual.transformer.cluster_subject = args.cluster_subject
            model.visual.transformer.attn_head = args.attn_head
            model.sampling_rate = 2 # same as video mae
            model.num_frames = 16 # same as video mae

            # transforms
            input_mean = [0.48145466, 0.4578275, 0.40821073]
            input_std = [0.26862954, 0.26130258, 0.27577711]
            trans = transforms.Compose([
                video_transform.Normalize(mean=input_mean, std=input_std)
            ])
            model.transform = trans
        if hook is not None:
            # remove existing hook if it exists
            if isinstance(hook_layer, list):
                if args.cluster_subject == 'block_token':
                    # remove all hooks
                    for lay in range(12):
                        if len(model.blocks._modules[str(lay)]._forward_hooks) > 0:
                            model.blocks._modules[str(lay)]._forward_hooks.clear()
                    # register hook
                    model.blocks._modules[str(hook_layer)].register_forward_hook(hook)
                    model.blocks._modules[str(hook_layer)].hook_dict = hook_dict if hook_dict is not None else None
                else:
                    for lay in range(12):
                        # model.visual.transformer.resblocks._modules[str(lay)]._modules['attn']
                        if hasattr(model.visual.transformer.resblocks._modules[str(lay)]._modules['attn'], 'hook_dict'):
                            delattr(model.visual.transformer.resblocks._modules[str(lay)]._modules['attn'], 'hook_dict')
                    # register hook for all layers
                    for lay in hook_layer:
                        model.visual.transformer.resblocks._modules[str(lay)]._modules['attn'].hook_dict = hook_dict[lay] if hook_dict is not None else None
    else:
        raise NotImplementedError
    model.eval()
    return model


def prepare_directories(args):
    save_dir = os.path.join('results', args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    args.vcd_dir = os.path.join(save_dir, 'vcd')
    if not os.path.exists(args.vcd_dir):
        os.mkdir(args.vcd_dir)
    args.save_dir = save_dir

    if args.cluster_subject == 'block_token':
        args.attn_head = [0]

    return args

def save_vcd(vcd):

    vcd_path = os.path.join(vcd.args.vcd_dir, 'vcd.pkl')

    # clean up dic

    # save dataset as pickle file
    with open(vcd.cached_file_path, 'wb') as f:
        pickle.dump(vcd.dataset, f)
    try:
        delattr(vcd, 'model')
    except:
        pass
    try:
        delattr(vcd, 'num_intra_clusters')
    except:
        pass
    try:
        delattr(vcd, 'dataset')
    except:
        pass
    # save vcd as pickle file
    with open(vcd_path, 'wb') as f:
        pickle.dump(vcd, f)
