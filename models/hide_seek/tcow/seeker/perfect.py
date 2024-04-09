'''
Stub module for test-time baselines.
Created by Basile Van Hoorick, Nov 2022.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'seeker/'))

from models.hide_seek.tcow.__init__ import *

# Internal imports.
import models.hide_seek.tcow.data.data_utils as data_utils
import models.hide_seek.tcow.utils.my_utils as my_utils


def run_baseline(data_retval, which_baseline, logger, train_args):
    '''
    :return model_retval (dict).
    '''
    device = torch.device('cpu')
    phase = 'test'

    assert train_args.which_seeker == 'mask_track_2d'
    source_name = data_retval['source_name'][0]

    if source_name == 'kubric':
        kubric_retval = data_retval['kubric_retval']
        all_segm = kubric_retval['pv_segm_tf']
        # (B, 1, T, Hf, Wf).
        all_div_segm = kubric_retval['pv_div_segm_tf']
        # (B, M, T, Hf, Wf).
        inst_count = kubric_retval['pv_inst_count']
        # (B, 1); acts as Qt value per example.
        query_time = kubric_retval['traject_retval_tf']['query_time']
        # (B).
        occl_fracs = kubric_retval['traject_retval_tf']['occl_fracs_tf']
        # (B, M, T, 3) with (f, v, t).
        occl_cont_dag = kubric_retval['traject_retval_tf']['occl_cont_dag_tf']
        # (B, T, M, M, 3) with (c, od, of).
        # NOTE: ^ Based on non-cropped data, so could be inaccurate near edges!
        target_desirability = kubric_retval['traject_retval_tf']['desirability_tf']
        # (B, M, 7).
        scene_dp = data_retval['scene_dp']

        (B, _, T, H, W) = all_segm.shape
        assert T == train_args.num_frames
        Qs = train_args.num_queries

        # Sample top few hardest queries.
        sel_query_inds = my_utils.sample_query_inds(
            B, Qs, inst_count, target_desirability, train_args, device, phase)

        # Loop over selected queries and accumulate fake results & targets.
        # NOTE: Looking at calculate_metrics_mask_track(), we only need to return output_mask and
        # target_mask, though logvis also needs seeker_query_mask.
        all_seeker_query_mask = []
        all_target_mask = []
        all_output_mask = []

        for q in range(Qs):

            # Get query info.
            # NOTE: query_idx is still a (B) tensor, so don't forget to select index.
            # query_idx[b] refers directly to the snitch instance ID we are tracking.
            query_idx = sel_query_inds[:, q]  # (B).
            qt_idx = query_time[0].item()

            # Prepare query mask and ground truths.
            (seeker_query_mask, snitch_occl_by_ptr, full_occl_cont_id, target_mask,
                target_flags) = data_utils.fill_kubric_query_target_mask_flags(
                all_segm, all_div_segm, query_idx, qt_idx, occl_fracs, occl_cont_dag, scene_dp,
                logger, train_args, device, phase)

            output_mask = torch.zeros_like(target_mask)  # (B, 3, T, Hf, Wf).

            for b in range(B):

                currently_tracking = query_idx[b]
                occluded_frames = 0

                for t in range(T):

                    if which_baseline == 'query':
                        output_mask[b, 0, t] = seeker_query_mask[b, 0, qt_idx]

                    elif which_baseline == 'static':
                        if occl_fracs[b, query_idx[b], t, 0] < train_args.front_occl_thres:
                            # Not or just partially occluded.
                            # Copy ground truth directly.
                            # output_mask[b, 0, t] = target_mask[b, 0, t]
                            # Copy visible mask directly.
                            output_mask[b, 0, t] = (all_segm[b, 0, t] == query_idx[b] + 1)
                            occluded_frames = 0
                        
                        else:
                            # Fully (or nearly so, by a small tolerance) occluded.
                            # Copy last seen position directly.
                            if not(t >= 1):
                                # This frame is not eligible (too early, or snitch out of frame).
                                continue
                            
                            if occluded_frames == 0:
                                output_mask[b, 0, t] = target_mask[b, 0, t - 1]
                            else:
                                output_mask[b, 0, t] = output_mask[b, 0, t - 1]
                            
                            occluded_frames += 1

                    elif which_baseline == 'linear':
                        if occl_fracs[b, query_idx[b], t, 0] < train_args.front_occl_thres:
                            # Not or just partially occluded.
                            # Copy ground truth directly.
                            # output_mask[b, 0, t] = target_mask[b, 0, t]
                            # Copy visible mask directly.
                            output_mask[b, 0, t] = (all_segm[b, 0, t] == query_idx[b] + 1)
                            occluded_frames = 0

                        else:
                            # Fully (or nearly so, by a small tolerance) occluded.
                            # Measure velocity via t-2 and t-1 by comparing centers of gravity, and
                            # then propagate mask of t-1 with same speed in same direction.
                            if not(t >= 2 and output_mask[b, 0, t - 2].any()
                                   and output_mask[b, 0, t - 1].any()):
                                # This frame is not eligible (too early, or snitch out of frame).
                                continue

                            if occluded_frames == 0:
                                use_mask1 = target_mask
                                use_mask2 = target_mask
                            elif occluded_frames == 1:
                                use_mask1 = target_mask
                                use_mask2 = output_mask
                            else:
                                use_mask1 = output_mask
                                use_mask2 = output_mask

                            (y, x) = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                            norm1 = use_mask1[b, 0, t - 2].sum()
                            norm2 = use_mask2[b, 0, t - 1].sum()
                            cog1_x = (x * use_mask1[b, 0, t - 2]).sum() / norm1
                            cog1_y = (y * use_mask1[b, 0, t - 2]).sum() / norm1
                            cog2_x = (x * use_mask2[b, 0, t - 1]).sum() / norm2
                            cog2_y = (y * use_mask2[b, 0, t - 1]).sum() / norm2
                            dx = int(round((cog2_x - cog1_x).item()))
                            dy = int(round((cog2_y - cog1_y).item()))
                            if abs(dx) < W and abs(dy) < H:
                                output_mask[b, 0, t,
                                            max(0, dy):min(H, H + dy),
                                            max(0, dx):min(W, W + dx)] = \
                                    use_mask2[b, 0, t - 1,
                                                max(0, -dy):min(H, H - dy),
                                                max(0, -dx):min(W, W - dx)]

                            occluded_frames += 1

                    elif which_baseline == 'jump':
                        # NOTE: Take care to use currently_tracking here, not query_idx!
                        cur_vis_mask = (all_segm[b, 0, t] == currently_tracking + 1)

                        if occl_fracs[b, currently_tracking, t, 0] < train_args.front_occl_thres:
                            # Not or just partially occluded.
                            # Copy visible mask directly, but either as snitch or occluder mask
                            # depending on which instance we are marking right now.
                            if currently_tracking == query_idx[b]:
                                output_mask[b, 0, t] = cur_vis_mask
                            else:
                                output_mask[b, 1, t] = cur_vis_mask
                            occluded_frames = 0

                        else:
                            # Fully (or nearly so, by a small tolerance) occluded.
                            # Look at *xray* mask of current instance, and switch to other instance
                            # whose *visible* mask has biggest overlap.
                            cur_xray_mask = (all_div_segm[b, currently_tracking, t] == 1)
                            overlap_counts = []
                            for k in range(inst_count):
                                if k == currently_tracking:
                                    overlap_counts.append(-1)
                                else:
                                    overlap_counts.append(np.logical_and(
                                        all_segm[b, 0, t] == k + 1, cur_xray_mask).sum())
                            currently_tracking = np.argmax(overlap_counts)

                            cur_vis_mask = (all_div_segm[b, currently_tracking, t] == 1)
                            output_mask[b, 1, t] = cur_vis_mask

                            occluded_frames += 1

            all_seeker_query_mask.append(seeker_query_mask)  # (B, 1, T, Hf, Wf).
            all_target_mask.append(target_mask)  # (B, 3, T, Hf, Wf).
            all_output_mask.append(output_mask)  # (B, 3, T, Hf, Wf).

        seeker_query_mask = torch.stack(all_seeker_query_mask, dim=1)  # (B, Qs, 1, T, Hf, Wf).
        target_mask = torch.stack(all_target_mask, dim=1)  # (B, Qs, 3, T, Hf, Wf).
        output_mask = torch.stack(all_output_mask, dim=1)  # (B, Qs, 3, T, Hf, Wf).

    elif source_name == 'plugin':
        assert which_baseline == 'query'
        
        all_query = data_retval['pv_query_tf']  # (B, 1, T, Hf, Wf).
        all_target = data_retval['pv_target_tf']  # (B, 3, T, Hf, Wf).
        (T, H, W) = all_target.shape[-3:]
        
        seeker_query_mask = all_query.type(torch.float32)
        target_mask = all_target.type(torch.float32)
        
        qt_idx = seeker_query_mask.sum(dim=(0, 1, 3, 4)).argmax().item()
        output_mask = torch.zeros_like(target_mask)  # (B, 3, T, Hf, Wf).
        
        for t in range(T):
            output_mask[:, 0, t] = all_query[:, 0, qt_idx]

    else:
        raise ValueError('Unsupported source: %s' % source_name)

    # Copy snitch channel to all other (occluder & container) channels to get cross-channel metrics
    # (even if we don't use this information, since otherwise we just have 0 IOU everywhere).
    # NOTE / DEBUG: temporarily disabled for green colored out_oc / extra4 vis
    # if which_baseline != 'jump' and output_mask.shape[-4] == 3:
    #     output_mask[..., 1, :, :, :] = output_mask[..., 0, :, :, :]
    #     output_mask[..., 2, :, :, :] = output_mask[..., 0, :, :, :]

    # Convert perfect probits into fake logits for correct metrics and visualization.
    output_mask = output_mask * 20.0 - 10.0

    # Organize & return relevant info.
    model_retval = dict()
    model_retval['seeker_query_mask'] = seeker_query_mask  # (B, Qs?, 1, T, Hf, Wf).
    model_retval['target_mask'] = target_mask  # (B, Qs?, 3, T, Hf, Wf).
    model_retval['output_mask'] = output_mask  # (B, Qs?, 3, T, Hf, Wf).

    return model_retval
