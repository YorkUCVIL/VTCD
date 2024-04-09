'''
Neural network architecture description.
Created by Basile Van Hoorick, Oct 2022.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'seeker/'))
sys.path.insert(0, os.path.join(os.getcwd(), 'third_party/aot-benchmark/'))
sys.path.insert(0, os.getcwd())


from sklearn.cluster import KMeans
# from yellowbrick.cluster.elbow import kelbow_visualizer

from models.hide_seek.tcow.__init__ import *

# Internal imports.
import models.hide_seek.tcow.model.resnet as resnet
import models.hide_seek.tcow.model.vision_tf as vision_tf

# External imports.
from models.hide_seek.tcow.third_party.aot_benchmark.configs.default import DefaultEngineConfig
from models.hide_seek.tcow.third_party.aot_benchmark.networks.engines.aot_engine import AOTEngine, AOTInferEngine
from models.hide_seek.tcow.third_party.aot_benchmark.networks.models.aot import AOT


def _load_aot_network(net, checkpoint_path):
    # Adapted from checkpoint.py in aot-benchmark repo.
    pretrained = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'state_dict' in pretrained.keys():
        pretrained_dict = pretrained['state_dict']
    elif 'model' in pretrained.keys():
        pretrained_dict = pretrained['model']
    else:
        pretrained_dict = pretrained

    existing_dict = net.state_dict()
    update_dict = {}
    updated_keys = []
    removed_keys = []
    for k, v in pretrained_dict.items():
        if k in existing_dict:
            update_dict[k] = v
            updated_keys.append(k)
        elif k[:7] == 'module.':
            if k[7:] in existing_dict:
                update_dict[k[7:]] = v
                updated_keys.append(k[7:])
        else:
            removed_keys.append(k)

    existing_dict.update(update_dict)
    net.load_state_dict(existing_dict)

    updated_keys = sorted(updated_keys)
    removed_keys = sorted(removed_keys)

    del pretrained
    return (net, updated_keys, removed_keys)


class AOTWrapper(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger, num_frames=24, frame_height=224, frame_width=288, max_frame_gap=1,
                 aot_arch='aott', pretrain_path=''):
        '''
        X
        '''
        super().__init__()
        self.logger = logger
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.max_frame_gap = max_frame_gap
        self.aot_arch = aot_arch
        self.pretrain_path = pretrain_path
        self.eval_size_fix = False  # May be overwritten later at inference.

        # See also 20_aot_shapes.py.
        # Model choices: aott / aots / aotb / aotl.
        # https://github.com/yoxu515/aot-benchmark/blob/main/MODEL_ZOO.md
        self.aot_cfg = DefaultEngineConfig(exp_name='stub', model=self.aot_arch)
        self.aot_model = AOT(self.aot_cfg, encoder=self.aot_cfg.MODEL_ENCODER)
        self.aot_engine_train = AOTEngine(self.aot_model, gpu_id=0)
        self.aot_engine_test = AOTInferEngine(self.aot_model, gpu_id=0)

        # Process pretrained model (see trainer.py).
        if len(self.pretrain_path) > 0:
            self.logger.info(f'(AOTWrapper) Loading pretrained model from {self.pretrain_path}')
            self.aot_cfg.PRETRAIN_FULL = True
            self.aot_cfg.PRETRAIN_MODEL = self.pretrain_path
            (self.aot_model, updated_keys, removed_keys) = _load_aot_network(
                self.aot_model, self.pretrain_path)
            self.logger.info(f'(AOTWrapper) updated_keys: {updated_keys[:3]} ... {updated_keys[-3:]}')
            self.logger.info(f'(AOTWrapper) removed_keys: {removed_keys[:3]} ... {removed_keys[-3:]}')

    def set_phase(self, phase):
        '''
        Must be called when switching between train / validation / test phases.
        '''
        self.phase = phase
        self.logger.info(f'(AOTWrapper) Set phase: {phase}')

    def forward(self, input_frames, query_or_target):
        if 'train' in self.phase or 'val' in self.phase:
            return self.forward_train(input_frames, query_or_target)
        elif 'test' in self.phase:
            return self.forward_test(input_frames, query_or_target)
        else:
            raise ValueError(f'Invalid phase: {self.phase}')

    def forward_train(self, input_frames, target_mask):  # , gt_frames_given):
        '''
        Assumes input frames are already blacked out as appropriate.
        :param input_frames (B, 3, T, Hf, Wf) tensor.
        :param target_mask (B, 1, T, Hf, Wf) tensor.
        :param gt_frames_given (int) in range [U - 1, T].
        :return (output_mask, aot_loss).
            output_mask (B, 1, T, Hf, Wf) tensor.
            aot_loss (1, 3) tensor.
            NOTE: May seem a bit unnatural, but output_mask consists of integer (ID) values here,
            with 0 = background. 
        '''
        (B, _, T, Hf, Wf) = input_frames.shape
        U = 5  # Matches original AOT network: 1 ref + 4 cur (incl. 1 prev).

        output_mask = torch.zeros_like(target_mask)  # (B, 1, T, Hf, Wf) of float32.
        aot_loss = []

        # Instead of going through the entire video, sample 3 subclips per example.
        # This means we process roughly the equivalent number of frames as our own model.
        for _ in range(3):
            frame_gaps = 1 + np.random.choice(self.max_frame_gap, size=U - 1)
            frame_inds = np.concatenate([[0], np.cumsum(frame_gaps)])
            frame_start = np.random.randint(0, T - frame_inds[-1])
            frame_inds += frame_start

            input_clip = input_frames[:, :, frame_inds]  # (B, 3, U, Hf, Wf).
            target_clip = target_mask[:, :, frame_inds]  # (B, 1, U, Hf, Wf).

            all_frames = rearrange(input_clip, 'B C U H W -> (U B) C H W')
            # (U * B, 3, H, W).
            all_labels = rearrange(target_clip, 'B C U H W -> (U B) C H W')
            # (U * B, 1, H, W).
            obj_nums = [1] * B

            # See AOT trainer.py.
            self.aot_engine_train.restart_engine(batch_size=B, enable_id_shuffle=False)
            (loss, all_pred_mask, all_frame_loss, boards) = \
                self.aot_engine_train(all_frames, all_labels, B, obj_nums)
            # all_pred_mask = list-U of (B, H, W) tensors of int64.

            # Save prediction only at subclip frames where nothing has been written yet.
            # NOTE: aot_loss is going to be used for backpropagation, not output_mask, so we can
            # basically do whatever, but I am doing what makes most sense for visualization.
            for i, t in enumerate(frame_inds):
                if output_mask[:, 0, t].abs().sum() == 0.0:
                    output_mask[:, 0, t] = all_pred_mask[i]

            aot_loss.append(loss)

            # Sanity checks.
            assert np.all(frame_gaps >= 1)
            assert np.all(frame_gaps <= self.max_frame_gap)
            assert np.all(frame_inds >= 0)
            assert np.all(frame_inds < T)

        aot_loss = torch.stack(aot_loss).unsqueeze(0)  # (1, 3) tensor.

        return (output_mask, aot_loss)

    def forward_test(self, input_frames, query_mask, cluster_subject=None, cluster_memory=None, cluster_viz=None):
        '''
        Assumes input frames are already blacked out as appropriate.
        :param input_frames (B, 3, T, Hf, Wf) tensor.
        :param query_mask (B, 1, T, Hf, Wf) tensor.
        :return output_mask (B, 1, T, Hf, Wf) tensor.
            NOTE: Only non-zero for frames after query_mask, due to causal structure of AOT!
        '''
        (B, _, T, Hf, Wf) = input_frames.shape
        U = 5  # Matches original AOT network: 1 ref + 4 cur (incl. 1 prev).
        (He, We) = (577, 1041)  # For untrained variants; see AOT evaluator.py comments.

        query_time = query_mask.sum(dim=(0, 1, 3, 4)).argmax().item()
        output_mask = torch.ones_like(query_mask) * (-1.0)
        obj_nums = [1] * B

        if self.eval_size_fix:
            # This fixes the spatial misalignment issue that I've been seeing at epoch -1.
            cur_input = torch.nn.functional.interpolate(
                input_frames[:, :, query_time], size=(He, We), mode='bilinear')
            cur_query = torch.nn.functional.interpolate(
                query_mask[:, :, query_time], size=(He, We), mode='bilinear')
        else:
            cur_input = input_frames[:, :, query_time]
            cur_query = query_mask[:, :, query_time]

        # See AOT evaluator.py.
        self.aot_engine_test.restart_engine()
        self.aot_engine_test.add_reference_frame(cur_input, cur_query, obj_nums, frame_step=0)

        # concept cluster stuff
        cluster_viz = []
        all_memory_embeddings = self.aot_engine_test.aot_engines[0].curr_lstt_output
        if self.cluster_subject == 'tokens' or self.cluster_memory == 'tokens':
            single_step_cluster_viz = all_memory_embeddings[0]
        else:
            if self.cluster_memory == 'curr':
                all_layer_embeddings = all_memory_embeddings[1]
            elif self.cluster_memory == 'long':
                all_layer_embeddings = all_memory_embeddings[2]
            elif self.cluster_memory == 'short':
                all_layer_embeddings = all_memory_embeddings[3]
            else:
                raise NotImplementedError  # shapes don't work
            if self.cluster_subject == 'keys':
                single_step_cluster_viz = [x[0] for x in all_layer_embeddings]
            elif self.cluster_subject == 'values':
                single_step_cluster_viz = [x[0] for x in all_layer_embeddings]
            else:
                raise NotImplementedError
        if self.cluster_memory == 'short':
            single_step_cluster_viz = [rearrange(x, 'B C H W -> (H W) B C') for x in single_step_cluster_viz]
        cluster_viz.append(single_step_cluster_viz)

        for t in range(query_time + 1, T):
            
            if self.eval_size_fix:
                cur_input = torch.nn.functional.interpolate(
                    input_frames[:, :, t], size=(He, We), mode='bilinear')
            else:
                cur_input = input_frames[:, :, t]
            self.aot_engine_test.match_propogate_one_frame(cur_input)

            # add embeddings

            output_logits_multi = self.aot_engine_test.decode_current_logits(output_size=(Hf, Wf))
            # (B, 11, Hf, Wf).
            output_logits_binary = output_logits_multi[:, 1] - output_logits_multi[:, 0]
            # (B, Hf, Wf).
            output_mask[:, 0, t] = output_logits_binary

            # concept cluster stuff
            all_memory_embeddings = self.aot_engine_test.aot_engines[0].curr_lstt_output

            # use this to get the embedding type
            if self.cluster_subject == 'tokens' or self.cluster_memory == 'tokens':
                single_step_cluster_viz = all_memory_embeddings[0]
            else:
                if self.cluster_memory == 'curr':
                    all_layer_embeddings = all_memory_embeddings[1]
                elif self.cluster_memory == 'long':
                    all_layer_embeddings = all_memory_embeddings[2]
                elif self.cluster_memory == 'short':
                    all_layer_embeddings = all_memory_embeddings[3]
                else:
                    raise NotImplementedError  # shapes don't work
                if self.cluster_subject == 'keys':
                    single_step_cluster_viz = [x[0] for x in all_layer_embeddings]
                elif self.cluster_subject == 'values':
                    single_step_cluster_viz = [x[0] for x in all_layer_embeddings]
                else:
                    raise NotImplementedError
            if self.cluster_memory == 'short':
                single_step_cluster_viz = [rearrange(x, 'B C H W -> (H W) B C') for x in single_step_cluster_viz]
            cluster_viz.append(single_step_cluster_viz)

        # Copy query to output directly as fake logits.
        output_mask[:, 0, query_time] = query_mask[:, 0, query_time] * 20.0 - 10.0


        # aot embedding format:
        # lstt_embs (lay1, lay2, lay3),
        # lstt_curr_memories (lay1 (key, val), lay2 (key, val), lay3 (key, val)),
        # lstt_long_memories (lay1 (key, val), lay2 (key, val), lay3 (key, val)),
        # lstt_short_memories (lay1 (key, val), lay2 (key, val), lay3 (key, val)),

        # use all layers for aot and stack along time dimension
        cluster_viz = [torch.stack([x[0] for x in cluster_viz], dim=0).detach().cpu()
            ,torch.stack([x[1] for x in cluster_viz], dim=0).detach().cpu()
            ,torch.stack([x[2] for x in cluster_viz], dim=0).detach().cpu()]
        # todo: fix hard coding of 37 and 66
        # cluster_viz = [rearrange(x, 'T (H W) B C -> (B T) H W C', H=37, W=66) for x in cluster_viz]
        # cluster_viz = [torch.nn.functional.interpolate(x, size=(He, We), mode='bilinear') for x in cluster_viz]
        if cluster_viz[0].shape[1] == 2442:
            cluster_viz = [rearrange(x, 'T (H W) B C -> B C T H W', H=37, W=66) for x in cluster_viz]
        else:
            cluster_viz = [rearrange(x, 'T (H W) B C -> B C T H W', H=15, W=20) for x in cluster_viz]

        # # cluster
        # cluster_viz = [rearrange(x, 'T (H W) B C -> B (T H W) C', H=37, W=66) for x in cluster_viz]
        # concept_cluster = []
        # # max_num_clusters = 5
        # num_clusters = 2
        # for cluster_feature in cluster_viz:
        #     # og method
        #     cluster_feature = cluster_feature[0]  # remove batch dim
        #     # clustering_alg = KMeans(n_init='auto')
        #     # elbow_alg = kelbow_visualizer(clustering_alg, np.array(cluster_feature.cpu()), k=(2, max_num_clusters),
        #     #                               metric='silhouette', show=False, timings=False)
        #     # # elbow_alg = kelbow_visualizer(clustering_alg, np.array(cluster_feature.cpu()), k=(2, 10), metric='distortion',show=False, timings=False)
        #     # # elbow_alg.fit(y[0].cpu())
        #     # num_clusters = elbow_alg.elbow_value_
        #     # if num_clusters is None:
        #     #     num_clusters = max_num_clusters + 1
        #     # clust_out = KMeans(n_clusters=2, random_state=0, n_init='auto').fit_transform(cluster_feature)
        #     clust_out = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit_transform(cluster_feature.cpu())
        #     clust_out = torch.from_numpy(clust_out)
        #     # rearrange and interpolate
        #     clust_out = rearrange(clust_out.unsqueeze(0), 'B (T H W) K -> (K B) T H W', T=T, H=37, W=66, K=num_clusters)
        #     clust_out = torch.nn.functional.interpolate(clust_out, size=(Hf, Wf), mode='bilinear')
        #     clust_out = rearrange(clust_out, '(K B) T H W -> K B T H W', B=B, K=int(clust_out.shape[0] / B))
        #     concept_cluster.append(clust_out)


        # NOTE: output_mask consists of logits here.
        return output_mask, cluster_viz
