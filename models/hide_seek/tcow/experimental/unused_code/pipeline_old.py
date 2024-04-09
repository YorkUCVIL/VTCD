
        # for i in range(B):
        #     # i points within subbatch (i.e. resets per GPU);
        #     # j points within entire batch (i.e. resets per iteration);
        #     # k points within dataset (i.e. resets per epoch) and is unique per example;
        #     # l points within entire training process.
        #     j = within_batch_inds[i].item()
        #     k = j + cur_step * self.train_args.batch_size
        #     l = j + total_step * self.train_args.batch_size

        # Lazy instantiation of the controllers to ensure that the entire lifetimes of both
        # PyBullet and Blender are within this thread.
        # NOTE: The Kubric random seed has to be unique (i.e. different) for every (epoch, phase)
        # combination, yet constant for the same given seed.
        # NOTE: For now, we don't consider generating live data at test time, because the evaluation
        # has to remain deterministic in principle
        # if self.kubric_generators[j] is None and 'test' not in self.phase:
        #     kubric_seed_offset = self.train_args.seed
        #     kubric_seed_offset += epoch * 400000
        #     kubric_seed_offset += 200000 if 'val' in self.phase else 0
        #     kubric_seed_offset += 100000 if 'test' in self.phase else 0
        #     self.kubric_generators[j] = data_kubric.MyKubricGenerator(
        #         self.logger, self.phase, kubric_seed_offset, **self.kubric_args)
        # kub_gen = self.kubric_generators[j]

        # Decide whether this step happens in bootstrap (fast) or interactive (slow) mode.
        # if epoch < self.train_args.bootstrap_epochs:

            # # Create stub tensors to allow for seamless batch aggregation.
            # initial_rgb = torch.zeros(B, 3, self.train_args.hider_frames,
            #                           self.train_args.frame_height, self.train_args.frame_width,
            #                           dtype=torch.float32, device=self.device)
            # hider_policy_logits = torch.zeros(B, self.train_args.action_space,
            #                                   dtype=torch.float32, device=self.device)
            # hider_policy_probs = hider_policy_logits.clone()
            # hider_value = torch.zeros(B, 1, dtype=torch.float32, device=self.device)

            # # if self.train_args.bootstrap_dataloader:

            # # Bootstrap clips were provided by the dataloader for efficiency.
            # # NOTE / TODO: Be careful when using kubric_retval_all because the datatypes become
            # # quite different here; everything is a tensor, and strings don't get split!
            # bootstrap_idx = data_retval['scene_idx']  # (B, 1).
            # kubric_retval_all = data_retval['kubric_retval']

            # else:

                # # We have to take care of bootstrap loading ourselves, which is less efficient
                # # because DataParallel does not use multiprocessing.
                # # NOTE: We always force a shuffled access of the bootstrap, to ensure diversity in
                # # logs / visuals.
                # bootstrap_idx = np.random.randint(kub_gen.get_bootstrap_size())
                # kubric_retval_all = kub_gen.load_bootstrap_clip(bootstrap_idx)

                # # TODO how to do this without entire dataset class?
                # kubric_retval_all = self.to_tensor(kubric_retval_all)

        #     snitch_action = kubric_retval_all['snitch_action'].unsqueeze(-1).to(self.device)

        # else:

        #     raise NotImplementedError('TODO retrieve necessary data to update hider policy')
        #     live_idx = data_retval['live_idx']  # (B, 1).
        #     kubric_retval_all = data_retval['kubric_retval']



            # 2D heatmap of snitch.
            target_snitch_batch = []
            for i in range(B):
                target_snitch_traject = kubric_retval['snitch_traject'][i]  # (Tt, 2) with [x, y].
                target_snitch_traject = target_snitch_traject.detach().cpu().numpy()
                target_snitch = my_utils.traject_to_track_map(
                    target_snitch_traject, all_rgb.shape[-2], all_rgb.shape[-1],
                    self.train_args.track_map_stride)  # (1, Tt, Hm, Wm).
                target_snitch = torch.tensor(target_snitch, device=self.device)
                target_snitch_batch.append(target_snitch)
            target_snitch = torch.stack(target_snitch_batch)  # (B, 1, Tt, Hm, Wm).


            # # Get other metadata.
            # kubric_mode = kubric_retval['mode_int'].to(self.device)  # (B, 1).
            # kubric_index = kubric_retval['index'].to(self.device)  # (B, 1).
        
        # model_retval['kubric_mode'] = kubric_mode  # (B), 0 / 1 / 2 = bootstrap / initial / remainder.
        # model_retval['kubric_index'] = kubric_index  # (B), Refers to bootstrap or live seed.
        
        # model_retval['output_snitch'] = output_snitch  # (B, 1, Tt, Hm, Wm).
        # model_retval['target_snitch'] = target_snitch  # (B, 1, Tt, Hm, Wm).
            
        # model_retval['initial_rgb'] = initial_rgb  # (B, 3, Ti, Hf, Wf).
        # model_retval['hider_policy_logits'] = hider_policy_logits  # (B, A).
        # model_retval['hider_policy_probs'] = hider_policy_probs  # (B, A).
        # model_retval['hider_value'] = hider_value  # (B, 1).
        # model_retval['snitch_action'] = snitch_action  # (B, 1).


# FROM forward_kubric:



    if 'point_track' in self.train_args.which_seeker:
        query_uv = kubric_retval['traject_retval_tf']['query_uv_tf']  # (B, Qt, 2).
        query_time = kubric_retval['traject_retval_tf']['query_time']  # (B, Qt).
        query_depth = kubric_retval['traject_retval_tf']['query_depth_tf']  # (B, Qt).
        query_xyz = kubric_retval['traject_retval_tf']['query_xyz_tf']  # (B, Qt, 3).
        target_uvdxyz = kubric_retval['traject_retval_tf']['uvdxyz_tf']  # (B, Qt, T, 6).
        target_complexity = kubric_retval['traject_retval_tf']['complexity_tf']  # (B, Qt, 5).
        # (B, Qt, 3).
        target_desirability = kubric_retval['traject_retval_tf']['desirability_tf']
        Qt = target_uvdxyz.shape[1]  # Total available by dataloader.
        assert Qt == self.train_args.try_queries


    if 'point_track' in self.train_args.which_seeker:
        assert self.train_args.seeker_query_format in [
            'uvt', 'uvdt', 'xyzt', 'uvxyzt', 'uvdxyzt']
        assert self.train_args.seeker_query_type in ['append', 'mask', 'token',
                                                        'append_mask', 'append_token', 'mask_token',
                                                        'append_mask_token']
        assert self.train_args.seeker_output_format in [
            'uv', 'uvd', 'uvo', 'uvdo', 'xyz', 'xyzo']
        assert self.train_args.seeker_output_token in ['direct', 'query']
        assert self.train_args.seeker_output_type in ['regress', 'gaussian']


    if 'point_track' in self.train_args.which_seeker:

        cur_query_uv = query_uv[:, query_idx, :].diagonal(0, 0, 1).T
        # (B, B, 2) => (B, 2).
        qu_idx = torch.clip(torch.floor(cur_query_uv[:, 0] * W).type(torch.int64), 0, W - 1)
        # (B).
        qv_idx = torch.clip(torch.floor(cur_query_uv[:, 1] * H).type(torch.int64), 0, H - 1)
        # (B).
        qt_idx = query_time[:, query_idx].diagonal(0, 0, 1).type(torch.int64)
        # (B, B) => B.
        cur_query_depth = query_depth[:, query_idx].diagonal(0, 0, 1)
        # (B, B) => B.
        cur_query_xyz = query_xyz[:, query_idx, :].diagonal(0, 0, 1).T
        # (B, B, 3) => (B, 3).

        # Torch handles indexing more flexibly so this code is quite different from geometry.py.
        query_target_uvdxyz = target_uvdxyz[:, query_idx, qt_idx, :].diagonal(0, 0, 1).T
        # (B, B, 6) => (B, 6).
        query_target_uv = query_target_uvdxyz[:, 0:2]
        # (B, 2).
        query_target_depth = query_target_uvdxyz[:, 2]
        # (B).
        query_target_xyz = query_target_uvdxyz[:, 3:6]
        # (B, 3).

        # Sanity checks: Compare query info with GT trajectories.
        # NOTE: These values should be almost the same, modulo object_coordinates rounding.
        if (query_target_uv - cur_query_uv).abs().mean() >= 0.01:
            self.logger.warning(f'query_target_uv: {query_target_uv} != '
                                f'cur_query_uv: {cur_query_uv}')
        if (query_target_depth - cur_query_depth).abs().mean() >= 0.2:
            self.logger.warning(f'query_target_depth: {query_target_depth} != '
                                f'cur_query_depth: {cur_query_depth}')
        if (query_target_xyz - cur_query_xyz).abs().mean() >= 0.2:
            self.logger.warning(f'query_target_xyz: {query_target_xyz} != '
                                f'cur_query_xyz: {cur_query_xyz}')

        # query_video_depth = all_depth[:, 0, qt_idx, qv_idx, qu_idx].diagonal(0, 0, 1)
        # # (B, B) => (B).
        # query_video_xyz = all_xyz[:, :, qt_idx, qv_idx, qu_idx].diagonal(0, 0, 2).T
        # # (B, 3, B) => (B, 3).

        # Sanity checks: Compare query info with 3D video data.
        # NOTE: Sometimes huge errors arise due to object_coordinates. This is expected; see
        # geometry.py for an explanation.
        # if (query_video_depth - cur_query_depth).abs().mean() >= 2e-1:
        #     self.logger.warning(f'query_video_depth: {query_video_depth} != '
        #                         f'cur_query_depth: {cur_query_depth}')
        # if (query_video_xyz - cur_query_xyz).abs().mean() >= 2e-1:
        #     self.logger.warning(f'query_video_xyz: {query_video_xyz} != '
        #                         f'cur_query_xyz: {cur_query_xyz}')

        # We always create query channels, even if not for the model, it is still used for
        # visualization.
        if self.train_args.seeker_query_format == 'uvt':
            seeker_query_channels = torch.cat(
                [cur_query_uv,
                    qt_idx[:, None]], dim=-1)  # (B, 3).
        elif self.train_args.seeker_query_format == 'uvdt':
            seeker_query_channels = torch.cat(
                [cur_query_uv,
                    cur_query_depth[:, None], qt_idx[:, None]], dim=-1)  # (B, 4).
        elif self.train_args.seeker_query_format == 'xyzt':
            seeker_query_channels = torch.cat(
                [cur_query_xyz,
                    qt_idx[:, None]], dim=-1)  # (B, 4).
        elif self.train_args.seeker_query_format == 'uvxyzt':
            seeker_query_channels = torch.cat(
                [cur_query_uv,
                    cur_query_xyz, qt_idx[:, None]], dim=-1)  # (B, 6).
        elif self.train_args.seeker_query_format == 'uvdxyzt':
            seeker_query_channels = torch.cat(
                [cur_query_uv, cur_query_depth[:, None],
                    cur_query_xyz, qt_idx[:, None]], dim=-1)  # (B, 7).
        seeker_query_channels = seeker_query_channels.type(torch.float32).to(self.device)

        if 'mask' in self.train_args.seeker_query_type:
            seeker_query_mask = torch.zeros_like(seeker_input[:, 0:1])  # (B, 1, T, Hf, Wf).
            for b in range(B):
                # seeker_query[b, :, qt_idx[b],
                #              qv_idx[b] - 1:qv_idx[b] + 2, qu_idx[b] - 1:qu_idx[b] + 2] = -1.0
                seeker_query_mask[b, :, qt_idx[b],
                                    qv_idx[b], qu_idx[b]] = 1.0
            seeker_query_mask = seeker_query_mask.type(torch.float32).to(self.device)

        else:
            seeker_query_mask = -torch.ones(1, dtype=torch.float32, device=self.device)

        # Prepare ground truth.
        target_traject = target_uvdxyz[:, query_idx, :, 0:6]  # (B, B, T, 6).
        target_traject = target_traject.diagonal(0, 0, 1)  # (T, 6, B).
        target_traject = rearrange(target_traject, 'T C B -> B T C')  # (B, T, 6).
        target_traject = target_traject.to(self.device)

        # Run seeker to recover particle trajectory.
        output_traject = self.networks['seeker'](
            seeker_input, seeker_query_channels, seeker_query_mask)  # (B, T, 2-7).

        all_seeker_query_channels.append(seeker_query_channels)
        all_target_traject_or_mask.append(target_traject)
        all_output_traject_or_mask.append(output_traject)



    if 'point_track' in self.train_args.which_seeker:
        seeker_query_channels = torch.stack(all_seeker_query_channels, dim=1)  # (B, Qs, 3).



    if 'point_track' in self.train_args.which_seeker:
        model_retval['seeker_query_channels'] = seeker_query_channels  # (B, Qs, 3-7).
        model_retval['target_traject'] = target_traject_or_mask  # (B, Qs, T, 6).
        model_retval['output_traject'] = output_traject_or_mask  # (B, Qs, T, 2-7).
