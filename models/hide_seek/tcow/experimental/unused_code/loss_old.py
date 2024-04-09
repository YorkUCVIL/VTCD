


    def get_mask_track_weight_new(self, target_occl_fracs, sel_seeker_frames, query_time):
        '''
        :param target_occl_fracs (B, Q, T, 3) tensor.
        :param sel_seeker_frames (B) tensor: Duration of visibility of every example.
        :param query_time (B) tensor: Query frame index for every example.
        :return pixel_weights (B, Q, T) tensor: floats in [0, inf).
        '''
        (B, Q, T, _) = target_occl_fracs.shape
        pixel_weights = torch.zeros((B, Q, T), dtype=torch.float32, device=target_occl_fracs.device)

        for b in range(B):
            for q in range(Q):
                for t in range(T):

                    if t >= sel_seeker_frames[b]:
                        pixel_weights[b, q, t] += self.train_args.future_weight

                    # We have soft occlusion percentages, so simply scale with desired loss weight.
                    pixel_weights[b, q, t] += target_occl_fracs[b, q, t, 0] * self.train_args.occluded_weight

        # Ensure all zeros (cases where nothing is special) become one.
        pixel_weights = pixel_weights.clip(min=1.0)

        # Query frame is easiest to solve, so reduce its importance.
        for b in range(B):
            for t in range(T):
                if t == query_time[b]:
                    pixel_weights[b, :, t] *= 0.25

        return pixel_weights




    def my_regress_loss(self, output_traject, target_traject, point_weights):
        '''
        :param output_traject (B, Q, T, 2-3) tensor.
        :param target_traject (B, Q, T, 2-3) tensor.
        :param point_weights (B, Q, T) tensor.
        '''
        loss_l1 = self.l1_loss(output_traject, target_traject)
        loss_l2 = self.l2_loss(output_traject, target_traject)
        loss_l1 = loss_l1 * point_weights[..., None]
        loss_l2 = loss_l2 * point_weights[..., None]
        loss_l1 = loss_l1.mean()
        loss_l2 = loss_l2.mean()

        loss_regress = (loss_l1 + loss_l2) / 2.0

        return loss_regress

    def get_point_track_weight(self, target_occl_flag, sel_seeker_frames):
        '''
        :param target_occl_flag (B, Q, T) tensor: 0 / 1 / 2 = visible / occluded / out of frame.
        :param sel_seeker_frames (B) tensor: Duration of visibility of every example.
        :return point_weights (B, Q, T) tensor: floats in [0, inf).
        '''
        (B, Q, T) = target_occl_flag.shape
        point_weights = torch.zeros((B, Q, T), dtype=torch.float32, device=target_occl_flag.device)

        for b in range(B):
            for q in range(Q):
                for t in range(T):

                    if t >= sel_seeker_frames[b]:
                        point_weights[b, q, t] += self.train_args.future_weight

                    if target_occl_flag[b, q, t] == 1:
                        point_weights[b, q, t] += self.train_args.occluded_weight

                    elif target_occl_flag[b, q, t] == 2:
                        point_weights[b, q, t] += self.train_args.out_of_frame_weight

        # Ensure all zeros (cases where nothing is special) become one.
        point_weights = point_weights.clip(min=1.0)

        return point_weights

    def get_point_track_sensibility(self, target_uvdxyz):
        '''
        Gets a per-point mask for where it makes sense to supervise points. Since UVD coordinates
            are not always well-behaved (and potentially unbounded), this acts as a form of
            regularization, especially when we are directly predicting in UV space.
        :param target_uvdxyz (B, Q, T, 6) tensor.
        :return sensible_mask (B, Q, T) tensor.
        '''
        sensible_mask = torch.ones_like(target_uvdxyz[..., 0])

        # Image coordinates are usually in [0, 1] except when out of frame. In that case, only allow
        # deviations up to once the image size.
        sensible_mask[target_uvdxyz[..., 0] < -1.0] = 0.0
        sensible_mask[target_uvdxyz[..., 0] > 2.0] = 0.0
        sensible_mask[target_uvdxyz[..., 1] < -1.0] = 0.0
        sensible_mask[target_uvdxyz[..., 1] > 2.0] = 0.0

        # Ignore points closer than 1 cm to the camera.
        sensible_mask[target_uvdxyz[..., 2] < 0.1] = 0.0

        # Ignore points that are very far away from the camera.
        # NOTE: This should never happen in Kubric, but it acts as a safety check.
        sensible_mask[target_uvdxyz[..., 2] > 200.0] = 0.0

        return sensible_mask

    def get_per_thing_frontmost_relevance(self, occl_fracs, occl_cont_ptrs):
        '''
        :param occl_fracs (B, Q, T, 3) tensor.
        :param occl_cont_ptrs (B, Q, T, 4) tensor.
        '''
        pass

    def my_heatmap_loss(self, output_map_logits, target_map):
        '''
        :param output_map_logits (B, 1, T, H, W) tensor.
        :param target_map (B, 1, T, H, W) tensor.
        '''
        (B, _, T, H, W) = output_map_logits.shape

        if self.train_args.class_balancing:
            pos_mask = (target_map == 1.0)
            neg_mask = (target_map == 0.0)

            if pos_mask.sum() >= T / 6.0:
                output_pos = output_map_logits[pos_mask]
                target_pos = target_map[pos_mask]
                loss_loc_pos = self.bce_or_focal_loss(output_pos, target_pos)
            else:
                loss_loc_pos = 0.0

            if neg_mask.sum() >= T / 6.0:
                output_neg = output_map_logits[neg_mask]
                target_neg = target_map[neg_mask]
                loss_loc_neg = self.bce_or_focal_loss(output_neg, target_neg)
            else:
                loss_loc_neg = 0.0

            loss_loc = (loss_loc_pos + loss_loc_neg) / 2.0

        else:
            loss_loc = self.bce_loss(output_map_logits, target_map)

        return loss_loc

    def my_gaussian_loss(self, output_gaussian, target_traject, point_weights):
        '''
        :param output_gaussian (B, Q, T, 4-6) tensor.
        :param target_traject (B, Q, T, 2-3) tensor.
        :param point_weights (B, Q, T) tensor.
        '''
        out_dims = output_gaussian.shape[-1] // 2
        targets = target_traject.clone()

        # https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        means = output_gaussian[..., 0:out_dims].clone()
        loss_reg_means = self.l2_loss(means, torch.zeros_like(means))

        # Ensure variances are always positive.
        vars = output_gaussian[..., out_dims:out_dims * 2] / 20.0
        loss_reg_vars = self.l2_loss(vars, torch.zeros_like(vars))
        vars = torch.nn.functional.softplus(vars).clone()

        loss_gauss = self.gnll_loss(means, targets, vars) / 100.0
        loss_gauss = loss_gauss * point_weights[..., None]
        loss_gauss = loss_gauss.mean()

        loss_reg = (loss_reg_means + loss_reg_vars) / 2.0

        return (loss_gauss, loss_reg)

    def per_example_point_track(self, data_retval, model_retval):
        (B, C, T, Hf, Wf) = data_retval['kubric_retval']['pv_rgb_tf'].shape
        camera_K = data_retval['kubric_retval']['camera_K_tf']  # (B, T, 3, 3).
        camera_R = data_retval['kubric_retval']['camera_R_tf']  # (B, T, 4, 4).

        # Evaluate entire subbatch for efficiency.
        sel_seeker_frames = model_retval['sel_seeker_frames']  # (B).
        sel_query_inds = model_retval['sel_query_inds']  # (B, Q).
        output_traject = model_retval['output_traject']  # (B, Q, T, 2-7).
        target_uvdxyz = model_retval['target_traject']  # (B, Q, T, 6) with (u, v, d, x, y, z).

        target_occl_flag = data_retval['kubric_retval']['traject_retval_tf']['occl_flag_tf']
        # (B, Q, T).
        target_occl_flag = target_occl_flag[:, sel_query_inds, :]  # (B, B, Q, T).
        target_occl_flag = target_occl_flag.diagonal(dim1=0, dim2=1)  # (Q, T, B).
        target_occl_flag = rearrange(target_occl_flag, 'Q T B -> B Q T')
        target_occl_flag = target_occl_flag.to(output_traject.device)
        # (B, Q, T) with 0 / 1 / 2 = visible / occluded / out of frame.

        loss_track = None
        loss_gauss_reg = None
        loss_occl_flag = None

        if self.train_args.track_lw > 0.0:
            point_weights = self.get_point_track_weight(
                target_occl_flag, sel_seeker_frames)  # (B, Q, T).
            sensible_mask = self.get_point_track_sensibility(target_uvdxyz)  # (B, Q, T).
            used_weights = point_weights * sensible_mask  # (B, Q, T).

            if 'xyz' in self.train_args.seeker_output_format:
                # The loss is calculated in XYZ world space.
                if self.train_args.seeker_output_type == 'regress':
                    loss_track = self.my_regress_loss(
                        output_traject[..., 0:3], target_uvdxyz[..., 3:6], used_weights)
                elif self.train_args.seeker_output_type == 'gaussian':
                    (loss_track, loss_gauss_reg) = self.my_gaussian_loss(
                        output_traject[..., 0:6], target_uvdxyz[..., 3:6], used_weights)

            elif 'uvd' in self.train_args.seeker_output_format:
                # The loss is calculated in UVD camera space.
                if self.train_args.seeker_output_type == 'regress':
                    loss_track = self.my_regress_loss(
                        output_traject[..., 0:3], target_uvdxyz[..., 0:3], used_weights)
                elif self.train_args.seeker_output_type == 'gaussian':
                    (loss_track, loss_gauss_reg) = self.my_gaussian_loss(
                        output_traject[..., 0:6], target_uvdxyz[..., 0:3], used_weights)

            elif 'uv' in self.train_args.seeker_output_format:
                # The loss is calculated in UV image space.
                if self.train_args.seeker_output_type == 'regress':
                    loss_track = self.my_regress_loss(
                        output_traject[..., 0:2], target_uvdxyz[..., 0:2], used_weights)
                elif self.train_args.seeker_output_type == 'gaussian':
                    (loss_track, loss_gauss_reg) = self.my_gaussian_loss(
                        output_traject[..., 0:4], target_uvdxyz[..., 0:2], used_weights)

            if 'uv' in self.train_args.seeker_output_format:
                loss_track *= 3.0  # Image space (UV or UVD) has smaller numerical sensitivity.

        if self.train_args.gauss_reg_lw <= 0.0:
            loss_gauss_reg = 0.0

        if self.train_args.occl_flag_lw > 0.0:
            assert 'o' in self.train_args.seeker_output_format
            loss_occl_flag = self.my_occlusion_flag_loss(output_traject[..., -1], target_occl_flag)

        # Calculate preliminary evaluation metrics in both UVD and XYZ space (we infer one from the
        # other as needed).
        # NOTE: All of the following is just for information and not for backpropagation.
        # NOTE: Some of these values may be NaN due to certain special cases not occurring!

        target_uvd = target_uvdxyz[..., 0:3]  # (B, Q, T, 3).
        target_xyz = target_uvdxyz[..., 3:6]  # (B, Q, T, 3).

        if 'xyz' in self.train_args.seeker_output_format:
            output_xyz = output_traject[..., 0:3]  # (B, Q, T, 3).
            output_uvd = geometry.project_points_3d_to_2d_multi(
                output_xyz, camera_K, camera_R, has_batch=True, has_time=True)  # (B, Q, T, 3).

        elif 'uvd' in self.train_args.seeker_output_format:
            output_uvd = output_traject[..., 0:3]  # (B, Q, T, 3).
            output_xyz = geometry.unproject_points_2d_to_3d_multi(
                output_uvd, camera_K, camera_R, has_batch=True, has_time=True)  # (B, Q, T, 3).

        elif 'uv' in self.train_args.seeker_output_format:
            output_uv = output_traject[..., 0:2]  # (B, Q, T, 2).
            # For world space metrics, assume that the output somehow contains perfect depth.
            output_uvd = torch.cat([output_uv, target_uvd[..., 2:3]], dim=-1)  # (B, Q, T, 3).
            output_xyz = geometry.unproject_points_2d_to_3d_multi(
                output_uvd, camera_K, camera_R, has_batch=True, has_time=True)  # (B, Q, T, 3).

        eucl_uvd_all = torch.norm(output_uvd - target_uvd, p=2, dim=-1).detach()  # (B, Q, T).
        eucl_xyz_all = torch.norm(output_xyz - target_xyz, p=2, dim=-1).detach()  # (B, Q, T).
        moving_mask = ((target_uvdxyz[..., 0, 3:6] - target_uvdxyz[..., -1, 3:6]).sum(dim=-1)
                       >= 0.01)[..., None].expand_as(eucl_uvd_all)  # (B, Q, T).

        metrics_retval = dict()
        metrics_retval['eucl_uvd_vis'] = eucl_uvd_all[target_occl_flag == 0].mean()
        metrics_retval['eucl_uvd_occ'] = eucl_uvd_all[target_occl_flag == 1].mean()
        metrics_retval['eucl_uvd_oof'] = eucl_uvd_all[target_occl_flag == 2].mean()
        metrics_retval['eucl_uvd_mov'] = eucl_uvd_all[moving_mask].mean()
        metrics_retval['eucl_uvd_all'] = eucl_uvd_all.mean()
        metrics_retval['eucl_xyz_vis'] = eucl_xyz_all[target_occl_flag == 0].mean()
        metrics_retval['eucl_xyz_occ'] = eucl_xyz_all[target_occl_flag == 1].mean()
        metrics_retval['eucl_xyz_oof'] = eucl_xyz_all[target_occl_flag == 2].mean()
        metrics_retval['eucl_xyz_mov'] = eucl_xyz_all[moving_mask].mean()
        metrics_retval['eucl_xyz_all'] = eucl_xyz_all.mean()

        # OPTIONAL / TODO:
        # Delete memory-taking stuff before aggregating across GPUs.
        # I observed that detaching saves around 3-4%.

        # Return results.
        loss_retval = dict()
        loss_retval['track'] = loss_track
        loss_retval['gauss_reg'] = loss_gauss_reg
        loss_retval['frontmost'] = None
        loss_retval['occl_flag'] = loss_occl_flag
        loss_retval.update(metrics_retval)  # This is necessary to keep simple loops over items().

        return loss_retval



# if pos_mask.sum() / pos_mask.numel() >= 1e-3:
#     output_pos = output_mask_logits[pos_mask]
#     target_pos = target_mask[pos_mask]
#     loss_mask_pos = self.bce_or_focal_loss(output_pos, target_pos)
# else:
#     loss_mask_pos = 0.0

# if neg_mask.sum() / neg_mask.numel() >= 1e-3:
#     output_neg = output_mask_logits[neg_mask]
#     target_neg = target_mask[neg_mask]
#     loss_mask_neg = self.bce_or_focal_loss(output_neg, target_neg)
# else:
#     loss_mask_neg = 0.0

# loss_mask = (loss_mask_pos + loss_mask_neg) / 2.0
