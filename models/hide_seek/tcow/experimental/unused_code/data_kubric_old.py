
class KubricMixedDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class for snitch tracking that controls a variable set of Kubric generators.
    This does not support query-based / 3D / point tracking.
    '''

    def __init__(self, logger, phase, kubric_args, use_data_frac, num_workers, batch_size,
                 live_interval, seed_offset, bootstrap_only):
        '''
        Initializes the dataset and its underlying generators.
        '''
        self.logger = logger
        self.phase = phase
        self.kubric_args = kubric_args
        self.use_data_frac = use_data_frac
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.live_interval = live_interval
        self.seed_offset = seed_offset
        self.bootstrap_only = bootstrap_only

        # This constraint allows for most Kubric generators to not need to instantiate an actual simulator and renderer.
        if not bootstrap_only:
            assert num_workers == batch_size * live_interval, \
                ('This may sound weird until you know how things are implemented, '
                 'but the ratios have to be nice here to avoid requiring every data worker to instantiate Kubric.')

        # Initialize Kubric interfaces.
        # If bootstrap_only, then there are exactly 0 live-capable kub_gens.
        # If not bootstrap_only, then there are exactly L live-capable kub_gens.
        cur_args = copy.deepcopy(kubric_args)
        cur_args['bootstrap_only'] = True
        bootstrap_kub_gen = kubric_live.MyKubricGenerator(
            logger, phase, seed_offset, **cur_args)

        self.kubric_generators = [None for _ in range(num_workers)]
        for i in range(num_workers):
            if bootstrap_only or i < batch_size * (live_interval - 1):
                kub_gen = bootstrap_kub_gen
            else:
                cur_args = copy.deepcopy(kubric_args)
                cur_args['bootstrap_only'] = False
                kub_gen = kubric_live.MyKubricGenerator(
                    logger, phase, seed_offset, **cur_args)
            self.kubric_generators[i] = kub_gen

        # Initialize other variables.
        self.bootstrap_size = kub_gen.get_bootstrap_size()
        self.used_dset_size = int(use_data_frac * self.bootstrap_size)
        self.force_shuffle = (use_data_frac < 1.0 and 'test' not in phase)
        self.set_epoch(-1)
        self.set_source_mode('bootstrap')

    def set_epoch(self, epoch):
        self.logger.info(f'(KubricMixedDataset) Setting epoch to: {epoch}')
        self.epoch = epoch

    def set_source_mode(self, source_mode):
        assert source_mode in ['bootstrap', 'mixed', 'live']
        self.logger.info(f'(KubricMixedDataset) Setting source mode to: {source_mode}')
        self.source_mode = source_mode

    def __len__(self):
        return self.used_dset_size

    def __getitem__(self, index):
        assert self.bootstrap_only or self.epoch >= 0, \
            'I was not correctly initialized by the training loop yet.'

        # Determine whether to gather offline or online data.
        if self.source_mode == 'mixed':
            # We want to synchronize across examples within every batch, such that they are either
            # all bootstrap or all live. Given batch size B and interval L, we will first see
            # B*(L-1) offline examples, then B online examples, then repeat.
            if (index // self.batch_size) % self.live_interval == self.live_interval - 1:
                cur_mode = 'live'
            else:
                cur_mode = 'bootstrap'
        else:
            cur_mode = self.source_mode

        # Some files / scenes may be invalid, so we keep retrying (up to an upper bound).
        scene_idx = -1
        retries = 0
        live_idx = -1
        kub_gen = self.kubric_generators[index % self.num_workers]

        if cur_mode == 'bootstrap':
            while True:
                try:
                    if not(self.force_shuffle) and retries == 0:
                        scene_idx = index
                    else:
                        scene_idx = np.random.randint(self.bootstrap_size)
                    kubric_retval = kub_gen.load_bootstrap_clip(scene_idx)
                    break

                except Exception as e:
                    retries += 1
                    self.logger.warning(f'(KubricMixedDataset) scene_idx: {scene_idx}')
                    self.logger.warning(f'(KubricMixedDataset) {str(e)}')
                    self.logger.warning(f'(KubricMixedDataset) retries: {retries}')
                    if retries >= 12:
                        raise e

        elif cur_mode == 'live':
            raise NotImplementedError()
            # TODO make this work, even if with uniformly random policy
            # TODO later implement actual hider network in this loop

            # Obtain first few frames of a random, unperturbed scene.
            # TODO: ensure these are never randomly augmented for position awareness
            # TODO: consider giving mode 3D information to the hider.
            live_idx = index + self.epoch * self.used_dset_size
            kubric_retval_initial = kub_gen.get_new_clip_initial(live_idx)
            # initial_rgb = kubric_retval_initial['rgb']  # (3, Ti, Hf, Wf).
            # initial_rgb = initial_rgb.unsqueeze(0)  # (B, 3, Ti, Hf, Wf).
            # initial_rgb = initial_rgb.to(self.device)

            # Run hider to calculate optimal starting position of adversarial snitch.
            # Also use random action per clip as reference.
            # (hider_policy_logits, hider_policy_probs, hider_value) = \
            #     self.networks['hider'](initial_rgb)
            # (B, A), (B, A), (B, 1).

            # Sample action according to predicted distribution.
            # policy_dist = torch.distributions.Categorical(logits=hider_policy_logits)
            # snitch_action = policy_dist.sample()  # (B, 1).
            # snitch_x = snitch_action % 4 - 1.5
            # snitch_y = (snitch_action // 4) % 4 - 1.5

            # Update video to insert snitch in chosen, controlled way.
            # kubric_retval = self.kub_gen.get_new_clip_remainder(live_idx, snitch_action.item())

            # TEMP:
            snitch_action = np.random.randint(self.kubric_args.action_space)
            kubric_retval = kub_gen.get_new_clip_remainder(live_idx, snitch_action.item())

        # TODO: Equalize shapes of all arrays within metadata to ensure batch aggregation works.
        # DEBUG / TEMP:
        del kubric_retval['metadata']

        data_retval = dict()
        data_retval['cur_mode'] = cur_mode
        data_retval['dset_idx'] = index
        data_retval['scene_idx'] = scene_idx
        data_retval['retries'] = retries
        data_retval['live_idx'] = live_idx
        data_retval['kubric_retval'] = kubric_retval

        return data_retval




        # Calculate (example) ground truth trajectories (based on NON-augmented data still).
        if self.query_size == 'point':

            # Generate (example) uniformly random queries and corresponding targets.
            (query_time, query_uv) = query_data

            # NOTE: For single scenes, we wish to evaluate spatial query point generalization by using
            # complementary checkerboard patterns in the 2D video. This is not principled, but provides
            # a simple way to avoid trivial overfitting to all possible trajectories.
            if self.single_scene:
                num_cells = 40  # Must be even for flipping to work out.
                query_uv_cats = np.floor(query_uv * num_cells).astype(np.int32)
                # (Q, 2) ints in [0, 19].
                query_uv_cats = (query_uv_cats[..., 0] + query_uv_cats[..., 1]) % 2
                # (Q) binary.
                query_uv_cats = np.broadcast_to(query_uv_cats[:, None], query_uv.shape)
                # (Q, 1) binary.

                # Horizontally flip 50% of the queries -- this ensures they land in the other category
                # iff num_cells is even.
                used_cat = (1 if 'test' in self.phase else 0)
                query_uv[query_uv_cats == used_cat][..., 0] = \
                    1.0 - query_uv[query_uv_cats == used_cat][..., 0]

            traject_retval = geometry.calculate_3d_point_trajectory_frames(
                pv_depth, pv_segm, pv_coords, metadata, query_time, query_uv, frame_inds)




        if self.load_3d:
            # Correct camera parameters to align with spatial changes in both 2D & 3D.
            # NOTE: Camera intrinsics remain unaffected by 3D augmentations.
            camera_K_tf = self.augs_pipeline.apply_augs_2d_intrinsics(camera_K, augs_params)
            # (T, 3, 3).

            # NOTE: Camera extrinsics remain unaffected by 2D augmentations.
            camera_R_tf = self.augs_pipeline.apply_augs_3d_extrinsics(camera_R, augs_params)
            # (T, 4, 4).

            # Apply 3D data transforms / augmentations.
            pv_xyz_tf = self.augs_pipeline.apply_augs_3d_frames(pv_xyz_tf, augs_params)
            # (T, Hf, Wf, 3).

        else:





        if self.query_size == 'point':
            # Correct UV trajectories to align with spatial 2D changes.
            traject_retval_tf = self.augs_pipeline.apply_augs_2d_traject(
                traject_retval, augs_params)

            # Correct XYZ trajectories and other annotations to align with spatial 3D changes.
            traject_retval_tf = self.augs_pipeline.apply_augs_3d_traject(
                traject_retval_tf, augs_params)

            # NOTE: The non-augmented traject_retval will not contain the following information.
            # Calculate per-point occlusion rates.
            traject_retval_tf['occl_flag_tf'] = self._get_occlusion_flags(
                traject_retval_tf['uvdxyz_tf'], pv_depth_tf)
            # (Q, T) with 0 / 1 / 2 = visible / occluded / out of frame.

            # Calculate per-query trajectory complexity (which is purely based on the coordinates
            # and occlusions).
            traject_retval_tf['complexity_tf'] = self._get_point_traject_complexity(
                traject_retval_tf['uvdxyz_tf'], traject_retval_tf['occl_flag_tf'])
            # (Q, 5) with floats in [0, inf).

            # Calculate per-query trajectory desirability (which uses more contextual information).
            traject_retval_tf['desirability_tf'] = self._get_point_traject_desirability(
                traject_retval_tf['complexity_tf'], traject_retval_tf['uvdxyz_tf'],
                traject_retval_tf['query_time'], traject_retval_tf['query_uv_tf'], pv_segm_tf)
            # (Q, 3) with floats in [0, inf).





    def _get_occlusion_flags(self, uvdxyz, depth):
        '''
        :param uvdxyz (Q, T, 6) array.
        :param depth (1, T, H, W) tensor.
        '''
        (Q, T) = uvdxyz.shape[:2]
        (H, W) = depth.shape[-2:]
        flags = np.zeros((Q, T), dtype=np.int32)

        # Mark occluded.
        for t in range(T):
            u_inds = np.clip(np.floor(uvdxyz[:, t, 0] * W).astype(np.int32), 0, W - 1)  # (Q).
            v_inds = np.clip(np.floor(uvdxyz[:, t, 1] * H).astype(np.int32), 0, H - 1)   # (Q).
            uv_hw_inds = np.stack([v_inds, u_inds], axis=0)  # (2, Q).
            uv_hw_inds_flat = np.ravel_multi_index(uv_hw_inds, (H, W))  # (Q).
            depth_sorta_flat = depth.view(T, H * W).numpy()  # (T, H * W).

            video_depth = depth_sorta_flat[t, uv_hw_inds_flat]  # (Q).
            point_depth = uvdxyz[:, t, 2]  # (Q).
            occluded_mask = (video_depth + 0.01 <= point_depth)  # (Q).
            # TODX check correctness, I'm seeing video_depth > point_depth sometimes!

            flags[occluded_mask, t] = 1

        # Mark out of frame (which takes precedence over occlusions).
        flags[uvdxyz[..., 0] < 0.0] = 2
        flags[uvdxyz[..., 0] >= 1.0] = 2
        flags[uvdxyz[..., 1] < 0.0] = 2
        flags[uvdxyz[..., 1] >= 1.0] = 2

        # Remaining = 0 = visible.
        return flags

    def _get_point_traject_complexity(self, uvdxyz, occlusion_flags):
        '''
        :param uvdxyz (Q, T, 6) array.
        :param occlusion_flags (Q, T).
        :return complexity (Q, 5) array of float32.
        '''
        (Q, T) = uvdxyz.shape[:2]
        complexity = np.zeros((Q, 5))

        for q in range(Q):
            # Determine how often the point is invisible (by occlusion or self-occlusion);
            # out-of-frame does not count.
            occluded_fraction = np.mean(occlusion_flags[q] == 1)

            # Measure total variation of normalized UV image space coordinates.
            delta_uv = uvdxyz[q, 1:, 0:2] - uvdxyz[q, :-1, 0:2]
            total_motion_uv = np.linalg.norm(delta_uv, axis=-1).sum()

            # Measure total variation in XYZ world space.
            delta_xyz = uvdxyz[q, 1:, 3:6] - uvdxyz[q, :-1, 3:6]
            total_motion_xyz = np.linalg.norm(delta_xyz, axis=-1).sum()

            # In addition to accumulated movement, we also emphasize acceleration (i.e. non-linear
            # motion) by comparing with a hypothetical linear trajectory from start to end, and
            # summing Euclidean distances. Note that resulting value can sometimes be larger than
            # total_motion, but this is desirable.
            linear_equivalent = np.linspace(uvdxyz[q, 0, 3:6], uvdxyz[q, -1, 3:6], T, axis=-1).T
            nonlinearity_xyz = np.linalg.norm(uvdxyz[q, :, 3:6] - linear_equivalent).sum()

            # Use weighted sum of all metrics, but also remember constituents.
            # NOTE: Ignore total_motion_uv for now, because xyz is more important.
            weighted = occluded_fraction * 3.0 + total_motion_uv + total_motion_xyz + \
                nonlinearity_xyz
            complexity[q, :] = [weighted, occluded_fraction, total_motion_uv, total_motion_xyz,
                                nonlinearity_xyz]

        return complexity

    def _get_point_traject_desirability(self, complexity, uvdxyz, query_time, query_uv, segm):
        '''
        :param complexity (Q, 5) array.
        :param uvdxyz (Q, T, 6) array.
        :param query_time (Q) array.
        :param query_uv (Q, 2) array.
        :param segm (1, T, H, W) tensor: 0 = background.
        :return desirability (Q, 3) array of float32.
        '''
        (Q, T) = uvdxyz.shape[:2]
        (T, H, W) = segm.shape[-3:]
        desirability = np.zeros((Q, 3))

        # Get instance ID for every query point.
        qt_idx = query_time
        qu_idx = np.clip(np.floor(query_uv[:, 0] * W).astype(np.int32), 0, W - 1)
        qv_idx = np.clip(np.floor(query_uv[:, 1] * W).astype(np.int32), 0, H - 1)
        query_video_segm_ids = segm[0, qt_idx, qv_idx, qu_idx].numpy()  # (Q).

        # Calculate overall object frequency / area statistics throughout the input video.
        (segm_ids, id_counts) = segm.unique(return_counts=True)
        id_mean_area_dict = {inst_id.item(): pxl_count.item() / (T * H * W)
                             for (inst_id, pxl_count) in zip(segm_ids, id_counts)}

        for q in range(Q):
            # Simply copy current trajectory complexity over time.
            total_complexity = complexity[q, 0]

            # If this is an object (and not background), then the bias in favor of selecting this
            # query should be correlated to how small this object generally is in image space.
            if query_video_segm_ids[q] == 0:
                object_regard = 0.0
            else:
                mean_area = id_mean_area_dict[query_video_segm_ids[q]]
                object_regard = 1.0 + 1.0 / (mean_area * 5.0 + 0.2)

            # Use weighted sum of all metrics, but also remember constituents.
            weighted = total_complexity + object_regard * 3.0
            desirability[q, :] = [weighted, total_complexity, object_regard]

        return desirability

