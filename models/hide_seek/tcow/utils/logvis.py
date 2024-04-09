'''
Logging and visualization logic.
Created by Basile Van Hoorick, Jun 2022.
'''

from __init__ import *
import json

# Internal imports.
import logvisgen
import visualization
import concepts

class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context, log_level=None):
        if 'batch_size' in args:
            if args.is_debug:
                self.step_interval = max(16 // args.batch_size, 2)
            elif args.single_scene:
                self.step_interval = max(32 // args.batch_size, 2)
            else:
                self.step_interval = max(64 // args.batch_size, 2)
        else:
            if args.is_debug:
                self.step_interval = 4
            elif args.single_scene:
                self.step_interval = 8
            else:
                self.step_interval = 16
        self.half_step_interval = self.step_interval // 2

        # With odd strides, we can interleave info from two data sources.
        if self.step_interval % 2 == 0:
            self.step_interval += 1
        if self.half_step_interval % 2 == 0:
            self.half_step_interval += 1
        self.concept_metrics = None
        # TEMP / DEBUG:
        # self.step_interval = 2

        super().__init__(log_dir=args.log_path, context=context, log_level=log_level)

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval, train_args, test_args, features_to_cluster=None):

        if not(('train' in phase and cur_step % self.step_interval == 0) or
               ('val' in phase and cur_step % self.half_step_interval == 0) or
                ('test' in phase)):
            return

        if 'point_track' in train_args.which_seeker:
            pass
            # self.handle_train_step_point_track(
            #     epoch, phase, cur_step, total_step, steps_per_epoch,
            #     data_retval, model_retval, loss_retval, train_args)

        elif 'mask_track' in train_args.which_seeker:
            # mp.Process(target=MyLogger.handle_train_step_mask_track,
            #               args=(self, epoch, phase, cur_step, total_step, steps_per_epoch,
            #               data_retval, model_retval, loss_retval, train_args)).start()
            file_name_suffix = self.handle_train_step_mask_track(
                epoch, phase, cur_step, total_step, steps_per_epoch,
                data_retval, model_retval, loss_retval, train_args, test_args, features_to_cluster, test_args.cluster_layer)

        return file_name_suffix

    def handle_train_step_mask_track(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                                     data_retval, model_retval, loss_retval, train_args, test_args, features_to_cluster=None, cluster_layer=None):
        source_name = data_retval['source_name'][0]
        scene_idx = data_retval['scene_idx'][0].item()
        if source_name == 'kubric':
            kubric_retval = data_retval['kubric_retval']
            frame_first = kubric_retval['frame_inds_load'][0, 0].item()
            frame_last = kubric_retval['frame_inds_load'][0, -1].item()

        # Obtain friendly short name for this step for logging, video saving, and CSV export.
        if not('test' in phase):
            file_name_suffix = ''
            file_name_suffix += f'e{epoch}_p{phase}_s{cur_step}_{source_name[:2]}_d{scene_idx}'

            if source_name == 'kubric':
                file_name_suffix += f'_f{frame_first}_l{frame_last}'
                if kubric_retval['augs_params']['reverse'][0]:
                    file_name_suffix += '_rev'
                if kubric_retval['augs_params']['palindrome'][0]:
                    file_name_suffix += '_pal'

        else:
            # NOTE: Here, we are now also exporting itemized results to CSV files.
            file_name_suffix = ''
            if source_name == 'plugin':
                plugin_name = str(pathlib.Path(data_retval['src_path'][0]).name).split('.')[0]
                frame_start = data_retval['frame_start'][0].item()
                frame_stride = data_retval['frame_stride'][0].item()
                file_name_suffix += f'{plugin_name}_i{frame_stride}_f{frame_start}_s{cur_step}'
            else:
                file_name_suffix += f's{cur_step}_{source_name[:2]}_d{scene_idx}'
                if source_name == 'kubric':
                    file_name_suffix += f'_f{frame_first}_l{frame_last}'

        # Log informative line including loss values & metrics in console.
        to_print = f'[Step {cur_step} / {steps_per_epoch}]  {source_name}  scn: {scene_idx}  '

        if model_retval is None:
            to_print += f'dlo: {train_args.data_loop_only}  '

        if source_name == 'plugin':
            to_print += f'name: {plugin_name}  f_stride: {frame_stride}  f_start: {frame_start}  '

        # NOTE: All wandb stuff for reporting scalars is handled in loss.py.
        # Assume loss may be missing (e.g. at test time).
        if loss_retval is not None:

            if len(loss_retval.keys()) >= 2:
                total_loss_seeker = loss_retval['total_seeker'].item()
                loss_track = loss_retval['track']
                loss_occl_mask = loss_retval['occl_mask']
                loss_cont_mask = loss_retval['cont_mask']
                loss_voc_cls = loss_retval['voc_cls']
                loss_occl_pct = loss_retval['occl_pct']
                to_print += (f'tot: {total_loss_seeker:.3f}  '
                             f'sn_t: {loss_track:.3f}  '
                             f'fo_t: {loss_occl_mask:.3f}  '
                             f'oc_t: {loss_cont_mask:.3f}  '
                             f'cls: {loss_voc_cls:.3f}  '
                             f'pct: {loss_occl_pct:.3f}  ')

            # Assume metrics are always present (even if count = 0).
            metrics_retval = loss_retval['metrics']
            snitch_iou = metrics_retval['mean_snitch_iou']
            occl_mask_iou = metrics_retval['mean_occl_mask_iou']
            cont_mask_iou = metrics_retval['mean_cont_mask_iou']
            to_print += (f'sn_iou: {snitch_iou:.3f}  '
                         f'fo_iou: {occl_mask_iou:.3f}  '
                         f'oc_iou: {cont_mask_iou:.3f}  ')

        self.info(to_print)

        log_rarely = 0 if 'test' in phase else train_args.log_rarely
        if log_rarely > 0 and cur_step % (self.step_interval * 16) != self.step_interval * 8:
            return file_name_suffix

        temp_st = time.time()  # DEBUG

        if model_retval is None:
            return  # Stop early if data_loop_only.

        no_gt_frames = None
        risky_frames = None
        target_mask = None
        output_occl_pct = None
        target_occl_pct = None
        snitch_weights = None

        # Save input, prediction, and ground truth data.
        if source_name == 'kubric':
            all_rgb = rearrange(kubric_retval['pv_rgb_tf'][0],
                                'C T H W -> T H W C').detach().cpu().numpy()
        else:
            all_rgb = rearrange(data_retval['pv_rgb_tf'][0],
                                'C T H W -> T H W C').detach().cpu().numpy()
        # (T, H, W, 3).
        if 'seeker_input' in model_retval:
            seeker_rgb = rearrange(model_retval['seeker_input'][0, 0:3],
                                'C T H W -> T H W C').detach().cpu().numpy()
        else:
            seeker_rgb = all_rgb
        # (T, H, W, 3).
        seeker_query_mask = model_retval['seeker_query_mask'][0].detach().cpu().numpy()
        # (Qs, 1, T, H, W).
        if train_args.tracker_arch == 'aot' and not('test' in phase):
            output_mask = model_retval['output_mask'][0].detach().cpu().numpy()  # Already int.
        else:
            output_mask = model_retval['output_mask'][0].sigmoid().detach().cpu().numpy()
        # (Qs, 3, T, H, W).
        target_mask = model_retval['target_mask'][0].detach().cpu().numpy()
        # (Qs, 3, T, H, W).
        if 'output_flags' in model_retval and train_args.tracker_arch != 'aot':
            output_occl_pct = model_retval['output_flags'][0, ..., 2].detach().cpu().numpy()
        # (Qs, T).
        if 'target_flags' in model_retval and source_name == 'kubric':
            target_occl_pct = model_retval['target_flags'][0, ..., 2].detach().cpu().numpy()
        # (Qs, T).
        if 'snitch_weights' in model_retval:
            snitch_weights = model_retval['snitch_weights'][0].detach().cpu().numpy()
        # (Qs, 1, T, H, W).

        frame_rate = train_args.kubric_frame_rate // train_args.kubric_frame_stride

        if source_name in ['ytvos', 'plugin']:
            # Add fake query count dimension (Qs = 1).
            seeker_query_mask = seeker_query_mask[None]  # Add fake query count dimension (Qs = 1).
            output_mask = output_mask[None]
            target_mask = target_mask[None]  # Add fake query count dimension (Qs = 1).
            if output_occl_pct is not None:
                output_occl_pct = output_occl_pct[None]
            if target_occl_pct is not None:
                target_occl_pct = target_occl_pct[None]

            if source_name == 'ytvos':
                no_gt_frames = model_retval['no_gt_frames'][0].detach().cpu().numpy()  # (T).
                risky_frames = model_retval['risky_frames'][0].detach().cpu().numpy()  # (T).
                frame_rate = train_args.ytvos_frame_rate // train_args.ytvos_frame_stride

            elif source_name == 'plugin':
                # We want to slow down plugin videos according to how much we are subsampling them
                # temporally for the model, but not too drastically.
                frame_stride = data_retval['frame_stride'][0].item()
                used_frame_stride = (frame_stride +
                                     test_args.plugin_prefer_frame_stride) / 2.5
                frame_rate = int(round(test_args.plugin_frame_rate / used_frame_stride))

        dimmed_rgb = (all_rgb + seeker_rgb) / 2.0  # Indicates when input becomes black.
        # (T, H, W, 3).
        if output_mask is not None:
            
            # DEBUG / TEMP: force out_oc vis generation for AOT baseline! => Cmo is always = 3.
            while output_mask.shape[1] < 3:
                output_mask = np.concatenate(
                    [output_mask, np.zeros_like(output_mask[:, 0:1])], axis=1)
            
            (Qs, Cmo) = output_mask.shape[:2]
        else:
            Cmo = 0
        if target_mask is not None:
            (Qs, Cmt) = target_mask.shape[:2]
        else:
            Cmt = 0
        # NOTE: Typically, Cm == 3 except for ytvos and baselines, where Cm == 1.
        
        self.logger.debug(
            f'logvis tensor to cpu: {time.time() - temp_st:.3f}s')  # DEBUG

        # Superimpose input video, predicted mask, and borders of query & ground truth masks.
        for q in range(Qs):

            temp_st = time.time()  # DEBUG

            # Construct query & target outlines.
            query_border = visualization.draw_segm_borders(
                seeker_query_mask[q, 0][..., None], fill_white=False)  # (T, H, W).
            snitch_border = visualization.draw_segm_borders(
                target_mask[q, 0][..., None], fill_white=False) \
                if Cmt >= 1 else np.zeros_like(output_mask[q, 0], dtype=np.bool)  # (T, H, W).
            frontmost_border = visualization.draw_segm_borders(
                target_mask[q, 1][..., None], fill_white=False) \
                if Cmt >= 2 else np.zeros_like(output_mask[q, 0], dtype=np.bool)  # (T, H, W).
            outermost_border = visualization.draw_segm_borders(
                target_mask[q, 2][..., None], fill_white=False) \
                if Cmt >= 3 else np.zeros_like(output_mask[q, 0], dtype=np.bool)  # (T, H, W).

            # self.logger.debug(
            #     f'logvis video creation 1: {time.time() - temp_st:.3f}s')  # DEBUG
            
            # Draw annotated model input.
            # NOTE: This exact part sometimes takes ~10s!
            vis_input = visualization.create_model_input_video(
                dimmed_rgb, seeker_query_mask[q, 0], query_border)
            # (T, H, W, 3).

            # self.logger.debug(
            #     f'logvis video creation 2: {time.time() - temp_st:.3f}s')  # DEBUG

            # Draw annotated model output at snitch level.
            # NOTE: This exact part sometimes takes ~10-70s!
            vis_snitch = visualization.create_model_output_snitch_video(
                seeker_rgb, output_mask[q], query_border, snitch_border, grayscale=False)
            # (T, H, W, 3).

            # self.logger.debug(
            #     f'logvis video creation 3: {time.time() - temp_st:.3f}s')  # DEBUG

            # Draw annotated model output at snitch + frontmost + outermost levels.
            vis_allout = visualization.create_model_output_snitch_occl_cont_video(
                seeker_rgb, output_mask[q], query_border, snitch_border, frontmost_border,
                outermost_border, grayscale=True)
            # (T, H, W, 3).

            # self.logger.debug(
            #     f'logvis video creation 4: {time.time() - temp_st:.3f}s')  # DEBUG

            # Mark frames with either no annotation or too high occlusion risk for ytvos at bottom.
            if no_gt_frames is not None and risky_frames is not None:
                vis_snitch = visualization.mark_no_gt_risky_frames_bottom_video(
                    vis_snitch, no_gt_frames, risky_frames)
                vis_allout = visualization.mark_no_gt_risky_frames_bottom_video(
                    vis_allout, no_gt_frames, risky_frames)
                # (T, H, W, 3).

            # self.logger.debug(
            #     f'logvis video creation 5: {time.time() - temp_st:.3f}s')  # DEBUG

            # Denote important VOC flags etc. at top.
            # NOTE: Disabled 11/15 because flags are not evaluated.
            # cur_output_pct = output_occl_pct[q] if output_occl_pct is not None else None
            # cur_target_pct = target_occl_pct[q] if target_occl_pct is not None else None
            # vis_snitch = visualization.append_flags_top_video(
            #     vis_snitch, output_mask[q], target_mask[q], cur_output_pct, cur_target_pct)
            # vis_allout = visualization.append_flags_top_video(
            #     vis_allout, output_mask[q], target_mask[q], cur_output_pct, cur_target_pct)
            # (T, H, W, 3).

            # self.logger.debug(
            #     f'logvis video creation 6: {time.time() - temp_st:.3f}s')  # DEBUG

            # Draw detailed snitch mask per-pixel loss weights for visual debugging.
            if snitch_weights is not None and not('test' in phase):
                vis_slw = visualization.create_snitch_weights_video(seeker_rgb, snitch_weights[q])
                # (T, H, W, 3).

            vis_intgt = None
            if 'test' in phase or ('is_figs' in train_args and train_args.is_figs):
                vis_intgt = visualization.create_model_input_target_video(
                    seeker_rgb, seeker_query_mask[q, 0], target_mask[q], query_border,
                    snitch_border, frontmost_border, outermost_border, grayscale=False)

            vis_extra = []
            if ('test' in phase and test_args.extra_visuals) or \
                    ('is_figs' in train_args and train_args.is_figs):
                
                # Include raw masks mapped directly as RGB channels without any input video.
                vis_extra.append(np.stack(
                    [target_mask[q, 1], target_mask[q, 0], target_mask[q, 2]], axis=-1))
                vis_extra.append(np.stack(
                    [output_mask[q, 1], output_mask[q, 0], output_mask[q, 2]], axis=-1))
                
                # Include temporally concatenated & spatially horizontally concatenated versions of
                # (input) + (output + target) or (input + target) + (output + target).
                vis_allout_pause = np.concatenate([vis_allout[0:1]] * 3 + [vis_allout[1:]], axis=0)
                vis_intgt_pause = np.concatenate([vis_intgt[0:1]] * 3 + [vis_intgt[1:]], axis=0)
                vis_extra.append(np.concatenate([vis_input, vis_allout], axis=0))  # (T, H, W, 3).
                vis_extra.append(np.concatenate([vis_intgt_pause, vis_allout], axis=0))  # (T, H, W, 3).
                vis_extra.append(np.concatenate([vis_input, vis_allout_pause], axis=2))  # (T, H, W, 3).
                vis_extra.append(np.concatenate([vis_intgt_pause, vis_allout_pause], axis=2))  # (T, H, W, 3).

            self.logger.debug(
                f'logvis video creation total: {time.time() - temp_st:.3f}s')  # DEBUG
            temp_st = time.time()  # DEBUG

            file_name_suffix_q = file_name_suffix + f'_q{q}'
            # Easily distinguish all-zero results.
            # file_name_suffix_q += f'_mo{output_mask.mean():.3f}'

            if not('test' in phase):
                wandb_step = epoch
                accumulate_online = 8
                extensions = ['.webm']
            else:
                wandb_step = cur_step
                accumulate_online = 1
                # extensions = ['.gif', '.webm']
                # extensions = ['.webm']
                # extensions = ['.gif']
                extensions = ['.mp4']

            # NOTE: Without apply_async, this part would take by far the most time.
            avoid_wandb = test_args.avoid_wandb if 'test' in phase else train_args.avoid_wandb
            online_name = f'in_p{phase}' if avoid_wandb == 0 else None
            self.save_video(vis_input, step=wandb_step,
                            file_name=f'more/{file_name_suffix_q}_in',
                            online_name=online_name,
                            accumulate_online=accumulate_online,
                            caption=f'{file_name_suffix_q}',
                            extensions=extensions, fps=frame_rate // 2,
                            upscale_factor=2, apply_async=True)
            # if Cmo >= 3:
            #     file_name = f'more/{file_name_suffix_q}_out_sn'
            # else:
            #     file_name = f'{file_name_suffix_q}_out_sn'  # No subfolder because baselines don't have multiple output
            online_name = f'out_p{phase}_sn' if avoid_wandb == 0 else None
            self.save_video(vis_snitch, step=wandb_step,
                            file_name=f'more/{file_name_suffix_q}_out_sn',
                            online_name=online_name,
                            accumulate_online=accumulate_online,
                            caption=f'{file_name_suffix_q}',
                            extensions=extensions, fps=frame_rate // 2,
                            upscale_factor=2, apply_async=True)
            if Cmo >= 3:
                online_name = f'out_p{phase}_oc' if avoid_wandb == 0 else None
                self.save_video(vis_allout, step=wandb_step,
                                file_name=f'{file_name_suffix_q}_out_oc',
                                online_name=online_name,
                                accumulate_online=accumulate_online,
                                caption=f'{file_name_suffix_q}',
                                extensions=extensions, fps=frame_rate // 2,
                                upscale_factor=2, apply_async=True)
            if snitch_weights is not None and not('test' in phase):
                online_name = None
                self.save_video(vis_slw, step=wandb_step,
                                file_name=f'more/{file_name_suffix_q}_slw',
                                online_name=online_name,
                                accumulate_online=accumulate_online,
                                caption=f'{file_name_suffix_q}',
                                extensions=extensions, fps=frame_rate // 2,
                                upscale_factor=2, apply_async=True)

            if vis_intgt is not None:
                online_name = None
                self.save_video(vis_intgt, step=wandb_step,
                                file_name=f'more/{file_name_suffix_q}_tgt',
                                online_name=online_name,
                                accumulate_online=accumulate_online,
                                caption=f'{file_name_suffix_q}',
                                extensions=extensions, fps=frame_rate // 2,
                                upscale_factor=2, apply_async=True)

            if len(vis_extra) != 0:
                for i, vis in enumerate(vis_extra):
                    self.save_video(vis, step=wandb_step,
                                    file_name=f'more/{file_name_suffix_q}_extra{i}',
                                    online_name=online_name,
                                    accumulate_online=accumulate_online,
                                    caption=f'{file_name_suffix_q}',
                                    extensions=extensions, fps=frame_rate // 2,
                                    upscale_factor=2, apply_async=True)


            if features_to_cluster is not None:
                # todo: perform all clustering and shit here!

                # todo: quantify clusters
                # todo: longest temporal segment
                if self.concept_metrics is None:
                    try:
                        self.concept_metrics = {
                            'temporal_support': {cluster_layer[n]: [] for n in range(len(features_to_cluster))},
                            'spatial_support': {cluster_layer[n]: [] for n in range(len(features_to_cluster))},
                        }
                    except:
                        self.concept_metrics = {
                            'temporal_support': {n: [] for n in range(len(features_to_cluster))},
                            'spatial_support': {n: [] for n in range(len(features_to_cluster))},
                        }
                for lay_idx, layer_viz in enumerate(features_to_cluster):

                    # return features of specific shape, then pass to clustering function
                    cost = concepts.cluster_features(features_to_cluster[lay_idx])

                    # resize to original shape
                    cost = torch.nn.functional.interpolate(cost, size=vis_input.shape[-3:-1], mode='bilinear', align_corners=False)

                    layer_num = cluster_layer[lay_idx]
                    # visualize clusters
                    features_to_cluster_assign = cost.argmax(0)
                    for cluster_idx in range(cost.shape[0]):

                        # hard assignment visualization
                        # save each cluster as separate video with alpha value
                        concept_mask = torch.where(features_to_cluster_assign == cluster_idx, 1, 0) # get mask of cluster

                        # log temporal support
                        temporal_support = (concept_mask.sum((1,2))>0).sum().item()
                        self.concept_metrics['temporal_support'][layer_num].append(temporal_support)

                        # log spatial support
                        spatial_support = (concept_mask.sum((1,2))/(concept_mask.shape[-1]*concept_mask.shape[-2])).mean().item()
                        self.concept_metrics['spatial_support'][layer_num].append(spatial_support)

                        concept_mask = np.array(np.repeat(concept_mask.unsqueeze(0),3, axis=0).permute(1,2,3,0)) # repeat along channels

                        vis_concept_assign = visualization.create_concept_mask_video(all_rgb, concept_mask, alpha=0.5) # create video
                        # vis_concept = visualization.create_concept_mask_video(dimmed_rgb, concept_mask, alpha=0.5) # alternate rgb video?
                        online_name = None
                        self.save_video(vis_concept_assign, step=wandb_step,
                                        file_name=f'concepts/Argmax/{file_name_suffix_q}_Lay{layer_num}_Concept{cluster_idx}',
                                        online_name=online_name,
                                        accumulate_online=accumulate_online,
                                        caption=f'{file_name_suffix_q}_Lay{layer_num}_concept{cluster_idx}',
                                        extensions=extensions, fps=frame_rate // 2,
                                        upscale_factor=2, apply_async=True)

                        # soft assignment visualization
                        cluster_heatmap = cost[cluster_idx] # softmax and select cluster
                        cluster_heatmap = cluster_heatmap / cost.max() # normalize based on max value
                        cluster_heatmap = np.array(np.repeat(cluster_heatmap.unsqueeze(0), 3, axis=0).permute(1, 2, 3, 0))  # repeat along channels
                        vis_concept_heatmap = visualization.create_heatmap_video(all_rgb, cluster_heatmap, alpha=0.5) # create video
                        self.save_video(vis_concept_heatmap, step=wandb_step,
                                        file_name=f'concepts/Heatmap/{file_name_suffix_q}_Lay{layer_num}_Concept{cluster_idx}',
                                        online_name=online_name,
                                        accumulate_online=accumulate_online,
                                        caption=f'{file_name_suffix_q}_Lay{layer_num}_concept{cluster_idx}',
                                        extensions=extensions, fps=frame_rate // 2,
                                        upscale_factor=2, apply_async=True)




                # save metrics
                # if file note there, create it
                # if not os.path.exists(f'{self.log_dir}/metrics.json'):
                #     with open(f'{self.log_dir}/concept_metrics.json', 'wb') as f:
                #         json.dump(self.concept_metrics, f)

            self.logger.debug(
                f'logvis video saving: {time.time() - temp_st:.3f}s')  # DEBUG

        return file_name_suffix

    def epoch_finished(self, epoch):
        super().epoch_finished(epoch)
        self.commit_scalars(step=epoch)

    def handle_test_step(self, cur_step, num_steps, data_retval, inference_retval, all_args, cluster_viz=None):
        '''
        :param all_args (dict): train, test, train_dset, test_dset, model.
        '''

        model_retval = inference_retval['model_retval']
        loss_retval = inference_retval['loss_retval']

        file_name_suffix = self.handle_train_step(
            0, 'test', cur_step, cur_step, num_steps, data_retval, model_retval, loss_retval,
            all_args['train'], all_args['test'], cluster_viz)

        return file_name_suffix
