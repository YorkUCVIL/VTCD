
    def handle_train_step_point_track(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                                      data_retval, model_retval, loss_retval, train_args):
        assert train_args.seeker_input_format in ['rgb', 'rgbd', 'xyzrgb', 'xyzrgbd']
        assert train_args.seeker_output_format in ['uv', 'uvd', 'uvo', 'uvdo', 'xyz', 'xyzo']
        assert train_args.seeker_output_type in ['regress', 'gaussian']

        (B, C, Tt, Hf, Wf) = data_retval['kubric_retval']['pv_rgb_tf'].shape
        scene_idx = data_retval['scene_idx'][0].item()

        if model_retval is None or loss_retval is None:
            self.info(f'[Step {cur_step} / {steps_per_epoch}]  ' +
                      f'data_loop_only: {train_args.data_loop_only}')
            return

        total_loss_seeker = loss_retval['total_seeker'].item()
        loss_track = loss_retval['track']
        loss_gauss_reg = loss_retval['gauss_reg']
        loss_occl_flag = loss_retval['occl_flag']
        metric_retval = loss_retval
        eucl_xyz_all = metric_retval['eucl_xyz_all']
        eucl_xyz_vis = metric_retval['eucl_xyz_vis']
        eucl_xyz_occ = metric_retval['eucl_xyz_occ']

        # Print metrics in console.
        self.info(f'[Step {cur_step} / {steps_per_epoch}]  ' +
                  f'total: {total_loss_seeker:.3f}  ' +
                  f'track: {loss_track:.3f}  ' +
                  f'gauss_reg: {loss_gauss_reg:.3f}  ' +
                  f'occl_flag: {loss_occl_flag:.3f}  ' +
                  f'eucl_xyz_all: {eucl_xyz_all:.3f}  ' +
                  f'eucl_xyz_vis: {eucl_xyz_vis:.3f}  ' +
                  f'eucl_xyz_occ: {eucl_xyz_occ:.3f}')

        rgb_idx = train_args.seeker_input_format.index('rgb')  # 0 or 3.

        # Save input, prediction, and ground truth data.
        all_rgb = rearrange(data_retval['kubric_retval']['pv_rgb_tf'][0],
                            'C T H W -> T H W C').detach().cpu().numpy()
        # (T, H, W, 3).
        all_xyz = rearrange(data_retval['kubric_retval']['pv_xyz_tf'][0],
                            'C T H W -> T H W C').detach().cpu().numpy()
        # (T, H, W, 3).
        seeker_rgb = rearrange(model_retval['seeker_input'][0, rgb_idx:rgb_idx + 3],
                               'C T H W -> T H W C').detach().cpu().numpy()
        # (T, H, W, 3).
        # seeker_frames = model_retval['sel_seeker_frames'][0].item()
        dimmed_rgb = (all_rgb + seeker_rgb) / 2.0  # Indicates when input becomes black.
        (H, W) = all_rgb.shape[1:3]

        seeker_query_channels = model_retval['seeker_query_channels'][0].detach().cpu().numpy()
        # (Q, 3-7).
        seeker_query_mask = model_retval['seeker_query_mask'][0].detach().cpu().numpy()
        # (Q, 1, T, H, W).
        output_traject = model_retval['output_traject'][0].detach().cpu().numpy()
        # (Q, T, 2-7).
        target_traject = model_retval['target_traject'][0].detach().cpu().numpy()
        # (Q, T, 6) with (u, v, d, x, y, z).
        target_uv = target_traject[..., 0:2]
        # (Q, T, 2) with (u, v).
        (Q, T) = target_uv.shape[:2]

        # Handle variable and optional output dimensions.
        if 'o' in train_args.seeker_output_format:
            output_occl_flag = output_traject[..., -1]  # (Q, T) with (o).
        else:
            output_occl_flag = np.zeros_like(output_traject[..., -1])  # (Q, T).

        if 'xyz' in train_args.seeker_output_format:
            output_xyz = output_traject[..., 0:3]  # (Q, T, 3) with (x, y, z) or mean thereof.
            out_dims = 3
        elif 'uvd' in train_args.seeker_output_format:
            output_uvd = output_traject[..., 0:3]  # (Q, T, 3) with (u, v, d) or mean thereof.
            output_uv = output_uvd[..., 0:2]  # (Q, T, 2) with (u, v) or mean thereof.
            out_dims = 3
        elif 'uv' in train_args.seeker_output_format:
            output_uv = output_traject[..., 0:2]  # (Q, T, 2) with (u, v) or mean thereof.
            out_dims = 2

        if train_args.seeker_output_type == 'regress':
            output_std = np.zeros_like(output_traject[..., 0:out_dims])  # (Q, T, 2-3).
        elif train_args.seeker_output_type == 'gaussian':
            output_std_raw = output_traject[..., out_dims:out_dims * 2] / 20.0  # (Q, T, 2-3).
            # NOTE: Avoid this manual implementation which is numerically unstable:
            # output_vars = np.log(np.exp(output_std_raw / 20.0) + 1.0)  # Apply softplus.
            output_vars = torch.nn.functional.softplus(torch.tensor(output_std_raw)).numpy()
            output_std = np.sqrt(output_vars)  # (Q, T, 2-3) with sigma_xyz/uvd/uv.

        # If needed, convert 3D (XYZ) output trajectory to 2D (UV) for visualization.
        if 'xyz' in train_args.seeker_output_format:
            camera_K = data_retval['kubric_retval']['camera_K_tf'][0].detach().cpu().numpy()
            # (T, 3, 3).
            camera_R = data_retval['kubric_retval']['camera_R_tf'][0].detach().cpu().numpy()
            # (T, 4, 4).

            all_output_uv = []
            all_output_sensitivity = []

            for t in range(T):
                output_uvd = geometry.project_points_3d_to_2d(
                    output_xyz[:, t], camera_K[t], camera_R[t])  # (Q, 3).
                output_uv = output_uvd[..., 0:2]  # (Q, 2).
                output_sensitivity = geometry.calculate_uv_xyz_sensitivity(
                    output_xyz[:, t], camera_K[t], camera_R[t])  # (Q, 3, 2).

                all_output_uv.append(output_uv)
                all_output_sensitivity.append(output_sensitivity)

            output_uv = np.stack(all_output_uv, axis=1)  # (Q, T, 2).
            output_sensitivity = np.stack(all_output_sensitivity, axis=1)  # (Q, T, 3, 2).

        else:
            # All 2D information is already captured in output_uvd/uv and output_std.
            pass

        # Convert to integer image coordinates for drawing.
        output_uv_scaled = np.clip(output_uv, 0.0, 1.0)
        output_uv_scaled[..., 0] *= W
        output_uv_scaled[..., 1] *= H
        output_uv_scaled = np.floor(output_uv_scaled).astype(np.int32)
        target_uv_scaled = np.clip(target_uv, 0.0, 1.0)
        target_uv_scaled[..., 0] *= W
        target_uv_scaled[..., 1] *= H
        target_uv_scaled = np.floor(target_uv_scaled).astype(np.int32)

        # Prepare canvas & drawing tools.
        cmap = plt.cm.hsv
        z_list = np.arange(Q)
        colors = cmap(z_list / (np.max(z_list) + 1))
        vis = dimmed_rgb.copy()  # (T, H, W, 3).

        for q in range(Q):
            query_time = seeker_query_channels[q, -1]

            for t in range(T):
                # Emphasize frame where query happens by flashing all lines and markers.
                thickness = (2 if t == query_time else 1)

                # Draw lines to mark context within trajectory.
                for u in range(max(t - 2, 0), min(t + 2, T - 1)):
                    cv2.line(vis[t], output_uv_scaled[q, u], output_uv_scaled[q, u + 1],
                             color=colors[q], thickness=thickness)
                    cv2.line(vis[t], target_uv_scaled[q, u], target_uv_scaled[q, u + 1],
                             color=colors[q], thickness=thickness)

                # Clearly indicate target (star) for current frame.
                cv2.drawMarker(vis[t], target_uv_scaled[q, t], markerSize=6,
                               color=colors[q], thickness=thickness,
                               markerType=cv2.MARKER_TILTED_CROSS)

                # Clearly indicate output / mean (circle) for current frame.
                if train_args.seeker_output_type == 'regress':
                    cv2.circle(vis[t], output_uv_scaled[q, t], radius=5,
                               color=colors[q], thickness=thickness)

                elif train_args.seeker_output_type == 'gaussian':
                    center_uv = output_uv_scaled[q, t]

                    if 'xyz' in train_args.seeker_output_format:
                        # If 3D (XYZ world space) uncertainty is predicted, then we have to
                        # transform it and draw an approximate 2D ellipse instead.

                        # NEW:
                        # From geometric reasoning follows that coefficients should be squared,
                        # thereby also making all terms positive.
                        output_sensitivity = np.abs(np.square(output_sensitivity))
                        radius_u = np.sqrt(output_sensitivity[q, t, 0, 0] * output_std[q, t, 0] +
                                           output_sensitivity[q, t, 1, 0] * output_std[q, t, 1] +
                                           output_sensitivity[q, t, 2, 0] * output_std[q, t, 2])
                        radius_v = np.sqrt(output_sensitivity[q, t, 0, 1] * output_std[q, t, 0] +
                                           output_sensitivity[q, t, 1, 1] * output_std[q, t, 1] +
                                           output_sensitivity[q, t, 2, 1] * output_std[q, t, 2])

                    else:
                        # If image/camera space uncertainty is predicted, then draw that directly.
                        radius_u = output_std[q, t, 0]
                        radius_v = output_std[q, t, 1]

                    axis_lengths = [int(np.round(radius_u * W)), int(np.round(radius_v * H))]
                    cv2.ellipse(vis[t], center_uv, axis_lengths, 0, 0, 360, colors[q], thickness)

                # Finally, indicate with a whether the point is thought to be occluded (triangle).
                if output_occl_flag[q, t] > 0.0:
                    cv2.drawMarker(vis[t], output_uv_scaled[q, t], markerSize=6,
                                   color=colors[q], thickness=thickness,
                                   markerType=cv2.MARKER_TRIANGLE_DOWN)

        gallery = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).

        # Save seeker input as 3D point cloud.
        # https://docs.wandb.ai/guides/track/log/media
        all_xyzrgb = np.concatenate([all_xyz, all_rgb], axis=-1)  # (T, H, W, 6).
        all_xyzrgb[..., 3:6] *= 255.0
        pcl_vis_first = all_xyzrgb[0].reshape(-1, 6)  # (H * W, 6).
        pcl_vis_last = all_xyzrgb[-1].reshape(-1, 6)  # (H * W, 6).

        if 'test' not in phase:
            # NOTE: This part takes *by far* the most time!
            # TODO: Optimize somehow?
            self.save_video(gallery, step=epoch,
                            file_name=f'gal_e{epoch}_p{phase}_s{cur_step}_d{scene_idx}',
                            online_name=f'gal_p{phase}', accumulate_online=8,
                            caption=f'e{epoch}_p{phase}_s{cur_step}_d{scene_idx}',
                            extensions=['.gif', '.mp4'], fps=train_args.frame_rate // 2,
                            upscale_factor=2)

            self.save_3d(pcl_vis_first, step=epoch,
                         online_name=f'pcl_0_p{phase}',
                         caption=f'first_e{epoch}_p{phase}_s{cur_step}_d{scene_idx}')
            self.save_3d(pcl_vis_last, step=epoch,
                         online_name=f'pcl_{T - 1}_p{phase}',
                         caption=f'last_e{epoch}_p{phase}_s{cur_step}_d{scene_idx}')

        else:
            # Put seeker loss first so we can conveniently sort files by difficulty.
            self.save_video(gallery, step=cur_step,
                            file_name=f'gal_l{total_loss_seeker:.3f}_s{cur_step}_d{scene_idx}',
                            online_name=f'gal_p{phase}', accumulate_online=1,
                            caption=f'l{total_loss_seeker:.3f}_s{cur_step}_d{scene_idx}',
                            extensions=['.gif', '.mp4'], fps=train_args.frame_rate // 2,
                            upscale_factor=2)

            self.save_3d(pcl_vis_first, step=cur_step,
                         online_name=f'pcl_0_p{phase}',
                         caption=f'first_p{phase}_s{cur_step}_d{scene_idx}')
            self.save_3d(pcl_vis_last, step=cur_step,
                         online_name=f'pcl_{T - 1}_p{phase}',
                         caption=f'last_p{phase}_s{cur_step}_d{scene_idx}')

        if 'test' not in phase:
            pass

        else:
            # Log every step at test time (including loss & metrics).
            self.report_scalar('loss_total_seeker', total_loss_seeker, step=cur_step)
            self.report_scalar('loss_track', loss_track, step=cur_step)
            self.report_scalar('loss_occl_flag', loss_occl_flag, step=cur_step)



# if self.train_args.seeker_output_type == 'heatmap':
#     # Save tracking heatmaps.
#     output_traject_logits = model_retval['output_traject'][0, 0].detach().cpu().numpy()
#     output_traject_probs = 1.0 / (1.0 + np.exp(-output_traject_logits))
#     target_traject = model_retval['target_traject'][0, 0].detach().cpu().numpy()
#     output_heatmap = color_map(output_traject_probs)[..., :3]  # (Tt, Hm, Wm, 3).
#     target_heatmap = color_map(target_traject)[..., :3]  # (Tt, Hm, Wm, 3).
#     output_heatmap = output_heatmap.astype(np.float32)
#     target_heatmap = target_heatmap.astype(np.float32)
#     output_heatmap_up = np.stack([cv2.resize(x, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
#                                 for x in output_heatmap])
#     target_heatmap_up = np.stack([cv2.resize(x, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
#                                 for x in target_heatmap])
#     output_vis = all_rgb * 0.4 + output_heatmap_up * 0.6
#     target_vis = all_rgb * 0.4 + target_heatmap_up * 0.6




# # Save various mean values (visible in overview of runs).
# self.report_scalar(f'{phase}/kubric_mode', kubric_mode, step=epoch, remember=True)
# self.report_scalar(f'{phase}/hider_value', hider_value, step=epoch, remember=True)

# # Save histograms of statistics (more detailed, but only visible per run).
# self.report_scalar(f'{phase}/snitch_action_hist', snitch_action, step=epoch,
#                    remember=True, commit_histogram=True)
# self.report_scalar(f'{phase}/hider_policy_logits_hist', hider_policy_logits, step=epoch,
#                    remember=True, commit_histogram=True)
# self.report_scalar(f'{phase}/hider_policy_probs_hist', hider_policy_probs, step=epoch,
#                    remember=True, commit_histogram=True)
# self.report_scalar(f'{phase}/hider_value_hist', hider_value, step=epoch,
#                    remember=True, commit_histogram=True)



# self.report_scalar('snitch_action', snitch_action, step=cur_step)
# self.report_scalar('kubric_mode', kubric_mode, step=cur_step)
# self.report_scalar('hider_value', hider_value, step=cur_step)
# self.report_histogram('hider_policy_logits_hist', hider_policy_logits, step=cur_step)
# self.report_histogram('hider_policy_probs_hist', hider_policy_probs, step=cur_step)



        # NEW:
        # eucl_uvd_all = metric_retval['eucl_uvd_all']
        # eucl_uvd_vis = metric_retval['eucl_uvd_vis']
        # eucl_uvd_occ = metric_retval['eucl_uvd_occ']
        # eucl_uvd_oof = metric_retval['eucl_uvd_oof']
        # eucl_uvd_mov = metric_retval['eucl_uvd_mov']
        # eucl_xyz_all = metric_retval['eucl_xyz_all']
        # eucl_xyz_vis = metric_retval['eucl_xyz_vis']
        # eucl_xyz_occ = metric_retval['eucl_xyz_occ']
        # eucl_xyz_oof = metric_retval['eucl_xyz_oof']
        # eucl_xyz_mov = metric_retval['eucl_xyz_mov']
        # self.info(f'eucl_xyz_oof: {eucl_xyz_oof:.3f}  ' +
        #           f'eucl_xyz_mov: {eucl_xyz_mov:.3f}  ' +
        #           f'eucl_uvd_all: {eucl_uvd_all:.3f}  ' +
        #           f'eucl_uvd_vis: {eucl_uvd_vis:.3f}  ' +
        #           f'eucl_uvd_occ: {eucl_uvd_occ:.3f}  ' +
        #           f'eucl_uvd_oof: {eucl_uvd_oof:.3f}  ' +
        #           f'eucl_uvd_mov: {eucl_uvd_mov:.3f}')

        # OLD:
        # NOTE: This isn't watertight for B > 1.
        # eucl_all = metric_retval['eucl_all'].mean()
        # eucl_oof = metric_retval['eucl_oof'].mean() \
        #     if metric_retval['count_oof'].sum() != 0 else np.nan
        # eucl_occ = metric_retval['eucl_occ'].mean() \
        #     if metric_retval['count_occ'].sum() != 0 else np.nan
        # eucl_vis = metric_retval['eucl_vis'].mean() \
        #     if metric_retval['count_vis'].sum() != 0 else np.nan
        # eucl_moving = metric_retval['eucl_moving'].mean() \
        #     if metric_retval['count_moving'].sum() != 0 else np.nan

        # self.info(f'eucl_all: {eucl_all:.3f}  ' +
        #           f'eucl_oof: {eucl_oof:.3f}  ' +
        #           f'eucl_occ: {eucl_occ:.3f}  ' +
        #           f'eucl_vis: {eucl_vis:.3f}  ' +
        #           f'eucl_mov: {eucl_moving:.3f}')
