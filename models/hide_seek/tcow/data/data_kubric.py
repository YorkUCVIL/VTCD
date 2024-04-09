'''
Data loading and processing logic.
Created by Basile Van Hoorick, Jun 2022.
'''

from models.hide_seek.tcow.__init__ import *

# Library imports.
import json
import PIL  # For depth tiff reading.

# Internal imports.
import models.hide_seek.tcow.data.augs as augs
import models.hide_seek.tcow.data.data_utils as data_utils
import models.hide_seek.tcow.data.data_vis as data_vis
import models.hide_seek.tcow.utils.geometry as geometry
import models.hide_seek.tcow.utils.my_utils as my_utils


def get_num_perturbs_views(scene_dp):
    '''
    X
    '''
    if not os.path.exists(os.path.join(scene_dp, f'frames_p0_v0')):
        # This dataset does not follow the perturbation / viewpoint format and presumably has only
        # one video per scene.
        return (-2, -2)

    num_perturbs = 1
    while os.path.exists(os.path.join(scene_dp, f'frames_p{num_perturbs}_v0')):
        num_perturbs += 1

    num_views = 1
    while os.path.exists(os.path.join(scene_dp, f'frames_p0_v{num_views}')):
        num_views += 1

    return (num_perturbs, num_views)


class KubricQueryDataset(torch.utils.data.Dataset):
    '''
    PyTorch dataset class for point tracking on bootstrap data.
    This does not support live generation.
    '''

    def __init__(self, dset_root, logger, phase, num_frames=20, frame_height=240, frame_width=320,
                 frame_rate=12, frame_stride=1, max_delay=0, use_data_frac=1.0, augs_2d=True,
                 augs_3d=True, num_queries=2, query_time=0.2, query_size='thing', load_3d=False,
                 max_objects=36, load_all_perturbs=False, load_all_views=False,
                 force_perturb_idx=-1, force_view_idx=-1, single_scene=False, fake_data=False,
                 augs_version=3, front_occl_thres=0.95, outer_cont_thres=0.75, reverse_prob=0.0,
                 palindrome_prob=0.0, degrade='none', annot_visible_pxl_only=False):
        '''
        Initializes the dataset and its underlying generators.
        '''
        self.dset_root = dset_root
        self.logger = logger
        self.phase = phase
        self.use_data_frac = use_data_frac

        # Video / clip options.
        self.num_frames_load = num_frames + max_delay
        self.num_frames_clip = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_rate = frame_rate
        self.frame_stride = frame_stride
        self.max_delay = max_delay
        self.augs_2d = augs_2d
        self.augs_3d = augs_3d
        self.num_queries = num_queries
        self.query_time = query_time
        self.query_size = query_size
        self.load_3d = load_3d
        self.max_objects = max_objects
        self.load_all_perturbs = load_all_perturbs
        self.load_all_views = load_all_views
        self.force_perturb_idx = force_perturb_idx
        self.force_view_idx = force_view_idx
        self.single_scene = single_scene
        self.fake_data = fake_data
        self.augs_version = augs_version
        self.front_occl_thres = front_occl_thres
        self.outer_cont_thres = outer_cont_thres
        self.reverse_prob = reverse_prob
        self.palindrome_prob = palindrome_prob
        self.degrade = degrade
        self.annot_visible_pxl_only = annot_visible_pxl_only

        # Change whether to apply random color jittering, flipping, and cropping.
        self.do_random_augs = (('train' in phase or 'val' in phase) and not('noaug' in phase))
        self.to_tensor = torchvision.transforms.ToTensor()

        # Get phase name with respect to file system.
        if 'train' in phase:
            phase_dn = 'train'
        elif 'val' in phase:
            phase_dn = 'val'
        elif 'test' in phase:
            phase_dn = 'test'
        else:
            raise ValueError(phase)

        if not self.single_scene:
            # Get root and phase directories, correcting names first.
            phase_dp = os.path.join(dset_root, phase_dn)
            if not os.path.exists(phase_dp):
                phase_dp = dset_root

            # Get video subdirectories.
            scene_dns = sorted(os.listdir(phase_dp))
            scene_dns = [dn for dn in scene_dns if 'scn' in dn]
            scene_dps = [os.path.join(phase_dp, dn) for dn in scene_dns]
            scene_dps = [dp for dp in scene_dps if os.path.isdir(dp)]

        else:
            self.logger.warning(
                f'({phase}) single_scene is enabled with multiplier {self.use_data_frac:.1f}!')
            phase_dp = None
            scene_dps = [dset_root]

        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        manager = mp.Manager()
        scene_dps = manager.list(scene_dps)

        # Instantiate custom augmentation pipeline.
        self.augs_pipeline = augs.MyAugmentationPipeline(
            self.logger, self.num_frames_load, self.num_frames_clip, self.frame_height,
            self.frame_width, self.frame_stride, self.do_random_augs, self.augs_2d, self.augs_3d,
            self.augs_version, self.reverse_prob, self.palindrome_prob, False)

        # Assign variables.
        num_scenes = len(scene_dps)
        self.logger.info(f'(KubricQueryDataset) ({phase}) Scene count: {num_scenes}')
        self.phase_dn = phase_dn
        self.phase_dp = phase_dp
        self.scene_dps = scene_dps
        self.dset_size = num_scenes
        self.used_dset_size = int(use_data_frac * num_scenes)
        self.logger.info(f'(KubricQueryDataset) ({phase}) Used dataset size: {self.used_dset_size}')
        self.force_shuffle = (use_data_frac < 1.0 and ('train' in phase or 'val' in phase))
        self.follow_retval = None

        # Print remaining warnings depending on circumstances.
        if self.load_all_perturbs:
            self.logger.warning(
                f'(KubricQueryDataset) ({phase}) load_all_perturbs is enabled, which may be slow!')
        if self.load_all_views:
            self.logger.warning(
                f'(KubricQueryDataset) ({phase}) load_all_views is enabled, which may be slow!')

        if self.force_perturb_idx >= 0:
            self.logger.warning(
                f'(KubricQueryDataset) ({phase}) force_perturb_idx is {force_perturb_idx}, which may significantly alter data statistics!')
        if self.force_view_idx >= 0:
            self.logger.warning(
                f'(KubricQueryDataset) ({phase}) force_view_idx is {force_view_idx}, which may significantly alter data statistics!')

        if self.fake_data:
            self.logger.warning(
                f'(KubricQueryDataset) ({phase}) fake_data is enabled, which should be for debugging only!')

        assert self.query_size in ['point', 'thing']
        assert not(self.load_all_perturbs and self.force_perturb_idx >= 0), 'Incompatible options.'
        assert not(self.load_all_views and self.force_view_idx >= 0), 'Incompatible options.'

    def __len__(self):
        return self.used_dset_size

    def __getitem__(self, index):
        '''
        :return data_retval (dict).
        '''
        if self.follow_retval is None:

            # Some files / scenes may be invalid, so we keep retrying (up to an upper bound).
            retries = 0
            scene_idx = -1
            proposed_choice = -1

            while True:
                try:
                    if not(self.force_shuffle) and retries == 0:
                        scene_idx = index % self.dset_size
                    else:
                        scene_idx = np.random.randint(self.dset_size)

                    scene_dp = copy.deepcopy(self.scene_dps[scene_idx])
                    kubric_retval = self._load_example(scene_idx, scene_dp)

                    break  # We are successful if we reach this.

                except Exception as e:
                    retries += 1
                    self.logger.warning(f'(KubricQueryDataset) scene_idx: {scene_idx}')
                    self.logger.warning(f'(KubricQueryDataset) {str(e)}')
                    self.logger.warning(f'(KubricQueryDataset) retries: {retries}')
                    if retries >= 8:
                        raise e

        else:

            # WARNING: This is not supported (and will probably crash) if num_workers > 1!
            # No chance to fail in this case, since we want to precisely reproduce earlier results.
            kubric_retval = self._load_example_from_follow(self.follow_retval)
            retries = -1
            scene_idx = self.follow_retval['scene_idx']
            scene_dp = self.follow_retval['scene_dp']
            self.follow_retval = None  # Disable this special marker for next iteration.

        data_retval = dict()
        data_retval['source_name'] = 'kubric'
        data_retval['dset_idx'] = index
        data_retval['retries'] = retries
        data_retval['scene_idx'] = scene_idx
        data_retval['scene_dp'] = scene_dp
        data_retval['scene_dn'] = str(pathlib.Path(scene_dp).name)

        if isinstance(kubric_retval, list):
            # If we loaded multiple videos of the same scene, this is the one the network should
            # use. The others are just for evaluation and/or visualization. We stash these other
            # videos in another key to avoid confusing downstream users of this dataloader, while
            # also not redundantly copying data.
            proposed_choice = np.random.choice(len(kubric_retval))
            data_retval['proposed_choice'] = proposed_choice
            data_retval['kubric_retval'] = kubric_retval[proposed_choice]
            kubric_retval.pop(proposed_choice)
            data_retval['kubric_retval_others'] = kubric_retval

        else:
            data_retval['kubric_retval'] = kubric_retval

        return data_retval

    def _load_example(self, scene_idx, scene_dp):
        '''
        :return kubric_retvals (dict | list of dict).
        '''
        kubric_retvals = []

        # Generate non-deteministic info beforehand to ensure consistency.
        # NOTE: For mask_track, this is currently the only part where randomness occurs (and is
        # allowed)!
        augs_params = self.augs_pipeline.sample_augs_params()
        query_data = self._sample_query_data(augs_params)

        # Select variation and viewpoint (either random or provided).
        (num_perturbs, num_views) = get_num_perturbs_views(scene_dp)

        if num_perturbs == -2 or num_views == -2:
            # This dataset (scene) does not support multi-perturbation or multi-view videos.
            perturb_inds = [0]
            view_inds = [0]

        else:
            # This dataset (scene) supports and has one or more variations and/or viewpoints.
            if self.load_all_perturbs:
                perturb_inds = np.arange(num_perturbs)
            elif self.force_perturb_idx == -1:
                perturb_inds = [np.random.randint(num_perturbs)]
            else:
                perturb_inds = [self.force_perturb_idx]

            if self.load_all_views:
                view_inds = np.arange(num_views)
            elif self.force_view_idx == -1:
                view_inds = [np.random.randint(num_views)]
            else:
                view_inds = [self.force_view_idx]

        # Actually load the video(s).
        for perturb_idx in perturb_inds:
            for view_idx in view_inds:
                cur_kubric_retval = self._load_example_deterministic_cache_failsafe(
                    scene_idx, scene_dp, perturb_idx, view_idx, augs_params, query_data)
                cur_kubric_retval['num_perturbs'] = num_perturbs
                cur_kubric_retval['num_views'] = num_views
                kubric_retvals.append(cur_kubric_retval)

        if self.load_all_perturbs or self.load_all_views:
            return kubric_retvals
        else:
            return kubric_retvals[0]  # Not a list anymore (for backward compatibility)!

    def _load_example_deterministic_cache_failsafe(self, *args):
        '''
        Calls _load_example_deterministic() but retries once with newly regenerated cache when it
            fails with (potentially outdated) cache first.
        '''
        for retry in range(2):
            try:
                force_renew_cache = (retry >= 1)
                return self._load_example_deterministic(*args, force_renew_cache)
            except Exception as e:
                if retry == 0:
                    if not('[SkipCache]' in str(e)):
                        self.logger.warning(
                            f'(KubricQueryDataset) _load_example_deterministic failed ({str(e)}), '
                            f'setting force_renew_cache...')
                    else:
                        raise e
                elif retry >= 1:
                    raise e

    def _load_example_deterministic(self, scene_idx, scene_dp, perturb_idx, view_idx,
                                    augs_params, query_data, force_renew_cache):
        '''
        Loads an entire scene clip with given (random) augmentations.
        :param scene_idx (int).
        :param scene_dp (str).
        :param perturb_idx (int).
        :param view_idx (int).
        :param augs_params (dict).
        :param query_data (tuple | int).
        :return kubric_retval (dict).
        '''
        # =============================================
        # Part 1: Loading and preprocessing (in numpy).
        # =============================================
        # temp_st = time.time()  # DEBUG
        frame_inds_load = augs_params['frame_inds_load']  # Strictly no randomness involved here.
        newer_than = 1666561000.0  # Last change: bugfixed oc_dag frontmost occlusion 10/23.
        cache_fn = (f'cc_{perturb_idx}_{view_idx}_{frame_inds_load[0]}_'
                    f'{frame_inds_load[1]}_{frame_inds_load[-1]}.p')
        cache_fp = os.path.join(scene_dp, cache_fn)
        if force_renew_cache and os.path.exists(cache_fp):
            os.remove(cache_fp)
        preprocess_retval = my_utils.disk_cached_call(
            self.logger, cache_fp, newer_than, self._load_example_preprocess,
            scene_idx, scene_dp, perturb_idx, view_idx, frame_inds_load)
        # self.logger.debug(f'(KubricQueryDataset) _load_example_preprocess: {time.time() - temp_st:.3f}s')  # DEBUG

        # ===================================================
        # Part 2: Augmentation and postprocessing (in torch).
        # ===================================================
        # temp_st = time.time()  # DEBUG
        kubric_retval = self._load_example_augmentations(
            scene_idx, scene_dp, preprocess_retval, augs_params, query_data)
        # self.logger.debug(f'(KubricQueryDataset) _load_example_augmentations: {time.time() - temp_st:.3f}s')  # DEBUG

        # ============================
        # Part 3: Final sanity checks.
        # ============================
        # temp_st = time.time()  # DEBUG
        self._load_example_verify(kubric_retval, scene_dp)
        # self.logger.debug(f'(KubricQueryDataset) _load_example_verify: {time.time() - temp_st:.3f}s')  # DEBUG

        return kubric_retval

    def _load_example_preprocess(self, scene_idx, scene_dp, perturb_idx, view_idx, frame_inds_load):
        '''
        Data loading part in numpy that has no randomness or augmentations or query dependence.
            NOTE: This method is typically cached, so we should not be afraid of expensive
            calculations here.
        :return preprocess_retval (dict): Partially filled results + intermediate variables.
        '''
        assert self.query_size == 'thing'

        # Get relevant paths.
        scene_dn = str(pathlib.Path(scene_dp).name)
        if os.path.exists(os.path.join(scene_dp, 'frames')):
            # This dataset (scene) does not support multi-perturbation or multi-view videos.
            frames_dp = os.path.join(scene_dp, 'frames')
            metadata_fp = os.path.join(scene_dp, scene_dn + '.json')
        else:
            # This dataset (scene) supports and has one or more variations and/or viewpoints.
            frames_dp = os.path.join(scene_dp, f'frames_p{perturb_idx}_v{view_idx}')
            metadata_fp = os.path.join(scene_dp, scene_dn + f'_p{perturb_idx}_v{view_idx}.json')

        # Load metadata and perform consistency checks.
        with open(metadata_fp, 'r') as f:
            metadata = json.load(f)
        assert metadata['scene']['num_frames'] >= max(frame_inds_load) + 1, \
            f'Not enough frames available on disk versus requested frame_inds_load.'

        # Load camera parameters (intrinsics + extrinsics) for this viewpoint.
        (camera_K, camera_R) = geometry.get_camera_matrices_numpy(metadata)
        camera_K = camera_K[frame_inds_load]  # (Tv, 3, 3).
        camera_R = camera_R[frame_inds_load]  # (Tv, 4, 4).

        # Load all RGB + depth + segmentation + object coordinates video clip frames for this
        # particular (perturbation, viewpoint) pair.
        pv_rgb = []
        pv_depth = []
        pv_segm = []
        pv_coords = []
        pv_xyz = []

        # Loop over all frames.
        for f, t in enumerate(frame_inds_load):
            rgb_fp = os.path.join(frames_dp, f'rgba_{t:05d}.png')
            depth_fp = os.path.join(frames_dp, f'depth_{t:05d}.tiff')
            segm_fp = os.path.join(frames_dp, f'segmentation_{t:05d}.png')
            coords_fp = os.path.join(frames_dp, f'object_coordinates_{t:05d}.png')
            if not os.path.exists(rgb_fp):
                break

            if not(self.fake_data):
                rgb = plt.imread(rgb_fp)[..., 0:3]  # (H, W, 3) floats.
                if self.load_3d:
                    depth = np.array(PIL.Image.open(depth_fp))  # (H, W) floats.
                else:
                    depth = rgb[..., 0] + 1.0  # (H, W) floats.
                segm = plt.imread(segm_fp)[..., 0:3]  # (H, W, 3) floats.
                coords = plt.imread(coords_fp)[..., 0:3]  # (H, W, 3) floats.
            else:
                rgb = np.random.rand(120, 160, 3)
                depth = np.random.rand(120, 160) * 8.0
                segm = np.random.rand(120, 160, 3)
                coords = np.random.rand(120, 160, 3) * 8.0 - 4.0

            depth = depth[..., None]  # (H, W, 1).
            uvd = geometry.uvd_from_depth(depth)
            xyz = geometry.unproject_points_2d_to_3d(uvd, camera_K[f], camera_R[f])

            pv_rgb.append(rgb)
            pv_depth.append(depth)
            pv_segm.append(segm)
            pv_coords.append(coords)
            pv_xyz.append(xyz)

        pv_rgb = np.stack(pv_rgb, axis=0)  # (Tv, Hf, Wf, 3) floats in [0, 1].
        pv_depth = np.stack(pv_depth, axis=0)  # (Tv, Hf, Wf, 1) floats in [0, inf).
        pv_segm = np.stack(pv_segm, axis=0)  # (Tv, Hf, Wf, 3) floats in [0, 1].
        pv_coords = np.stack(pv_coords, axis=0)  # (Tv, Hf, Wf, 3) floats in [0, 1].
        pv_xyz = np.stack(pv_xyz, axis=0)  # (Tv, Hf, Wf, 3) floats in (-inf, inf).

        # Convert segmentation from raw RGB to instance IDs.
        K = metadata['scene']['num_valo_instances']
        if not(self.fake_data):
            pv_segm = data_vis.segm_rgb_to_ids_kubric(pv_segm)  # (Tv, Hf, Wf, 1) ints in [0, inf).
        else:
            pv_segm = np.random.randint(0, K + 1, size=(self.num_frames_load, 120, 160, 1))

        # Load all separated occlusion-invariant instance segmentation masks if needed.
        # pv_div_rgb = []
        # pv_div_depth = []
        pv_div_segm = []

        # Loop over all frames.
        for f, t in enumerate(frame_inds_load):
            per_inst_div_segm = []

            for k in range(K):
                cur_div_segm_fp = os.path.join(
                    frames_dp, f'divided_segmentation_{k:03d}_{t:05d}.png')

                if not(self.fake_data):
                    cur_div_segm = plt.imread(cur_div_segm_fp)[..., :3]  # (H, W, 3) floats.
                else:
                    cur_div_segm = np.random.rand(120, 160, K)

                cur_div_segm = (cur_div_segm.sum(axis=-1) > 0.1).astype(np.uint8)
                # (H, W) ints in [0, 1].
                per_inst_div_segm.append(cur_div_segm)

            div_segm = np.stack(per_inst_div_segm, axis=-1)  # (H, W, K) bytes in [0, 1].
            pv_div_segm.append(div_segm)

        pv_div_segm = np.stack(pv_div_segm, axis=0)  # (Tv, Hf, Wf, K) bytes in [0, 1].

        # Calculate (example) ground truth trajectories (based on NON-augmented data still).
        traject_retval = dict()

        # Almost all data is already available in pv_div_segm, so we let pipeline do the array
        # manipulation work.

        # Calculate occlusion, containment, and collision data.
        # coll_ptrs = data_utils.get_thing_collision_pointers(metadata, frame_inds_load)
        # (K, T).
        occl_fracs = data_utils.get_thing_occl_fracs_numpy(pv_segm, pv_div_segm)
        # (K, Tv, 3).

        # NOTE: This takes roughly as much time as loading pv_div_segm in the first place!
        (occl_cont_dag, relative_order, reconst_pv_segm, reconst_error) = \
            data_utils.get_thing_occl_cont_dag(pv_segm, pv_div_segm, metadata, frame_inds_load)
        # (Tv, K, K, 3), (Tv, K), (Tv, Hf, Wf, 1), float.

        if not(self.fake_data):
            if reconst_error >= 0.02:
                self.logger.warning(f'(KubricQueryDataset) Large reconst_error for pv_segm vs '
                                    f'depth-ordered pv_div_segm: {reconst_error:.3f}.')
            elif reconst_error >= 0.01:
                self.logger.debug(f'(KubricQueryDataset) Large reconst_error for pv_segm vs '
                                  f'depth-ordered pv_div_segm: {reconst_error:.3f}.')

        # Add annotation metadata to traject_retval, useful for evaluation.
        traject_retval['occl_fracs'] = occl_fracs  # (K, Tv, 3).
        traject_retval['occl_cont_dag'] = occl_cont_dag  # (Tv, K, K, 3).

        # Organize & return results gathered so far.
        # There is too much metadata to store in memory, so just save a pointer instead.
        kubric_retval = dict()
        kubric_retval['metadata_fp'] = metadata_fp
        kubric_retval['perturb_idx'] = perturb_idx
        kubric_retval['view_idx'] = view_idx
        kubric_retval['num_valo_instances'] = K
        kubric_retval['frame_inds_load'] = frame_inds_load

        preprocess_retval = dict()
        preprocess_retval['kubric_retval'] = kubric_retval
        preprocess_retval['metadata'] = metadata
        preprocess_retval['camera_K'] = camera_K
        preprocess_retval['camera_R'] = camera_R
        preprocess_retval['traject_retval'] = traject_retval  # Empty if point; partial if thing.
        preprocess_retval['pv_rgb'] = pv_rgb
        preprocess_retval['pv_depth'] = pv_depth
        preprocess_retval['pv_segm'] = pv_segm
        preprocess_retval['pv_coords'] = pv_coords
        preprocess_retval['pv_xyz'] = pv_xyz
        preprocess_retval['pv_div_segm'] = pv_div_segm

        return preprocess_retval

    def _load_example_augmentations(self, scene_idx, scene_dp, preprocess_retval, augs_params,
                                    query_data):
        '''
        Data loading part in torch after reading from disk and preprocessing.
        :return kubric_retval (dict): Completely filled results.
        '''
        assert not(self.load_3d)
        assert self.query_size == 'thing'

        kubric_retval = preprocess_retval['kubric_retval']
        K = kubric_retval['num_valo_instances']
        frame_inds_load = augs_params['frame_inds_load']
        frame_inds_clip = augs_params['frame_inds_clip']
        metadata = preprocess_retval['metadata']
        camera_K = preprocess_retval['camera_K']  # (Tv, 3, 3).
        camera_R = preprocess_retval['camera_R']  # (Tv, 4, 4).
        traject_retval = preprocess_retval['traject_retval']  # Empty if point; partial if thing.
        pv_rgb = preprocess_retval['pv_rgb']
        pv_depth = preprocess_retval['pv_depth']
        pv_segm = preprocess_retval['pv_segm']
        pv_coords = preprocess_retval['pv_coords']
        pv_xyz = preprocess_retval['pv_xyz']
        pv_div_segm = preprocess_retval['pv_div_segm']
        
        query_time = query_data  # Single integer.
        traject_retval['query_time'] = query_time

        # Apply any applicable ablations *before* augmentations / data visualizations.
        if self.degrade == 'smooth':
            # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
            pv_rgb = np.stack([cv2.bilateralFilter(x, 9, 192, 96) for x in pv_rgb])
        elif self.degrade == 'randclr':
            reds = np.random.choice(256, size=K + 1, replace=False)
            greens = np.random.choice(256, size=K + 1, replace=False)
            blues = np.random.choice(256, size=K + 1, replace=False)
            pv_rgb[..., 0] = reds[pv_segm[..., 0]] / 255.0
            pv_rgb[..., 1] = greens[pv_segm[..., 0]] / 255.0
            pv_rgb[..., 2] = blues[pv_segm[..., 0]] / 255.0

        # With small probability, output data with annotations for input & target visualization.
        # NOTE: On my local workstations, always generate all visualizations (= much slower!).
        if scene_idx <= 200 and (np.random.random() < 0.02 or
                                 platform.node() in ['HPOL', 'GPC20']):
            scene_dn = str(pathlib.Path(scene_dp).name)
            dset_name = scene_dn.split('_scn')[0]  # For example kubcon_v10.
            occl_cont_dag = traject_retval['occl_cont_dag']  # (Tv, K, K, 3).
            boxes_3d_vis = geometry.create_boxes_3d_video(
                pv_rgb.shape, frame_inds_load, metadata, camera_K, camera_R)

            data_vis.create_rich_annot_video(
                pv_rgb, pv_div_segm, occl_cont_dag, boxes_3d_vis,
                ['rgb', 'xray_segm'],
                self.logger, self.frame_rate // self.frame_stride,
                file_name=f'dset_{dset_name}/{self.phase_dn}_{scene_dn}_div_segm_xray_oc',
                ignore_if_exist=True, front_occl_thres=self.front_occl_thres,
                outer_cont_thres=self.outer_cont_thres)
            data_vis.create_rich_annot_video(
                pv_rgb, pv_div_segm, occl_cont_dag, boxes_3d_vis,
                ['box3d', 'oc_mark'],
                self.logger, self.frame_rate // self.frame_stride,
                file_name=f'dset_{dset_name}/{self.phase_dn}_{scene_dn}_box3d',
                ignore_if_exist=True, front_occl_thres=self.front_occl_thres,
                outer_cont_thres=self.outer_cont_thres)

        # Convert large numpy arrays to torch tensors, putting channel dimension first.
        pv_rgb_tf = rearrange(torch.tensor(pv_rgb, dtype=torch.float32), 'T H W C -> C T H W')
        pv_depth_tf = rearrange(torch.tensor(pv_depth, dtype=torch.float32), 'T H W C -> C T H W')
        pv_segm_tf = rearrange(torch.tensor(pv_segm, dtype=torch.uint8), 'T H W C -> C T H W')
        pv_coords_tf = rearrange(torch.tensor(pv_coords, dtype=torch.float32), 'T H W C -> C T H W')
        pv_xyz_tf = rearrange(torch.tensor(pv_xyz, dtype=torch.float32), 'T H W C -> C T H W') \
            if self.load_3d else torch.tensor([0], dtype=torch.float32)
        pv_div_segm_tf = rearrange(torch.tensor(pv_div_segm, dtype=torch.uint8), 'T H W K -> K T H W') \
            if self.query_size == 'thing' else torch.tensor([0], dtype=torch.uint8)

        # Apply 2D data transforms / augmentations.
        # NOTE: We must apply the transforms consistently across all modalities, and in particular,
        # resampling should be done carefully for integer / jumpy arrays (i.e. avoid interpolation).
        # NOTE: This involves subsampling videos to clips (Tc < Tv) according to frame_inds_clip!
        modalities = {'rgb': pv_rgb_tf, 'depth': pv_depth_tf, 'segm': pv_segm_tf,
                      'coords': pv_coords_tf, 'xyz': pv_xyz_tf, 'div_segm': pv_div_segm_tf}
        modalities_tf = self.augs_pipeline.apply_augs_2d_frames(modalities, augs_params)
        (pv_rgb_tf, pv_depth_tf, pv_segm_tf, pv_coords_tf, pv_xyz_tf, pv_div_segm_tf) = \
            (modalities_tf['rgb'], modalities_tf['depth'], modalities_tf['segm'],
             modalities_tf['coords'], modalities_tf['xyz'], modalities_tf['div_segm'])

        camera_K_tf = torch.tensor([0], dtype=torch.float32)
        camera_R_tf = torch.tensor([0], dtype=torch.float32)
        pv_xyz_tf = torch.tensor([0], dtype=torch.float32)

        # Tracking inputs and targets are centered around pv_div_segm_tf, so here we simply
        # calculate complexity / desirability for each instance.
        traject_retval_tf = copy.deepcopy(traject_retval)
        # Has occl_fracs, occl_cont_dag, query_time.

        # This step may seem unnecessary, but more objects may have gone out-of-frame after
        # data augmentation, so we need to recalculate the relevant occlusion fractions.
        occl_fracs_tf = data_utils.get_thing_occl_fracs_torch(pv_segm_tf, pv_div_segm_tf)
        # (K, Tc, 3) array of float32 with (f, v, t) in [0, 1].

        # NOTE: Even though we apply clip subsampling, this is still based on non-cropped data, so
        # it could be inaccurate near edges!
        occl_cont_dag_tf = traject_retval['occl_cont_dag'][frame_inds_clip]
        # (Tc, K, K, 3) array of float32.

        # Calculate per-query trajectory desirability (which is based on occlusions and motion).
        # NOTE: This incorporates all temporal and spatial augmentations already.
        desirability_tf = self._get_thing_traject_desirability(
            pv_div_segm_tf, occl_fracs_tf, traject_retval['query_time'])
        # (K, 7) with floats in (-inf, inf).

        # Finally, make all array & tensor sizes uniform such that data can be collated.
        # NOTE: M must be >= max instance count in any Kubric scene.
        (pv_div_segm_tf, _) = data_utils.pad_div_torch(
            pv_div_segm_tf, [0], self.max_objects)
        # (K, Tc, Hf, Wf) => (M, Tc, Hf, Wf).
        (traject_retval_tf['occl_fracs'], _) = data_utils.pad_div_numpy(
            traject_retval_tf['occl_fracs'], [0], self.max_objects)
        # (K, Tv, 3) => (M, Tv, 3). NOTE: Avoid using this because of possible Tv/Tc confusion.
        (traject_retval_tf['occl_fracs_tf'], _) = data_utils.pad_div_numpy(
            occl_fracs_tf, [0], self.max_objects)
        # (K, Tc, 3) => (M, Tc, 3).
        (traject_retval_tf['occl_cont_dag'], _) = data_utils.pad_div_numpy(
            traject_retval_tf['occl_cont_dag'], [1, 2], self.max_objects)
        # (Tv, K, K, 3) => (Tv, M, M, 3). NOTE: Avoid using this because of possible Tv/Tc confusion.
        (traject_retval_tf['occl_cont_dag_tf'], _) = data_utils.pad_div_numpy(
            occl_cont_dag_tf, [1, 2], self.max_objects)
        # (Tc, K, K, 3) => (Tc, M, M, 3).
        (traject_retval_tf['desirability_tf'], _) = data_utils.pad_div_numpy(
            desirability_tf, [0], self.max_objects)
        # (K, 7) => (M, 7).
        pv_inst_count = torch.tensor([K], dtype=torch.int32)  # (1).

        # For bookkeeping, return the direct clip frame index mapping to files on disk.
        frame_inds_direct = frame_inds_load[frame_inds_clip]

        # Include augmented data with results dict.
        # NOTE / WARNING: Transformed camera matrices have NOT been thoroughly vetted yet, so do not
        # trust them for now!
        kubric_retval['augs_params'] = augs_params  # dict.
        kubric_retval['frame_inds_direct'] = frame_inds_direct  # (Tc).
        kubric_retval['camera_K_tf'] = camera_K_tf  # (Tc, 3, 3) or (1).
        kubric_retval['camera_R_tf'] = camera_R_tf  # (Tc, 4, 4) or (1).
        kubric_retval['traject_retval_tf'] = traject_retval_tf  # dict; has occl_cont_dag_tf etc.
        kubric_retval['pv_rgb_tf'] = pv_rgb_tf  # (3, Tc, Hf, Wf).
        kubric_retval['pv_depth_tf'] = pv_depth_tf  # (1, Tc, Hf, Wf).
        kubric_retval['pv_segm_tf'] = pv_segm_tf  # (1, Tc, Hf, Wf).
        kubric_retval['pv_coords_tf'] = pv_coords_tf  # (3, Tc, Hf, Wf).
        kubric_retval['pv_xyz_tf'] = pv_xyz_tf  # (3, Tc, Hf, Wf) or (1).
        kubric_retval['pv_div_segm_tf'] = pv_div_segm_tf  # (M, Tc, Hf, Wf) or (1).
        kubric_retval['pv_inst_count'] = pv_inst_count  # (1).

        return kubric_retval

    def _load_example_verify(self, kubric_retval, scene_dp):
        '''
        :param kubric_retval (dict): Completely filled results.
        :param scene_dp (str).
        '''
        pv_segm_tf = kubric_retval['pv_segm_tf']
        pv_div_segm_tf = kubric_retval['pv_div_segm_tf']
        K = kubric_retval['num_valo_instances']
        desirability_tf = kubric_retval['traject_retval_tf']['desirability_tf']
        # (M, 7) with floats in (-inf, inf).

        # Perform sanity checks due to issues with segmentation indices.
        if self.query_size == 'thing':
            # NOTE: Because of frame subsampling, some formerly VALO instances may now be invisible.
            # This means that the number of unique IDs in pv_segm may be < num_valo_instances, but
            # the highest ID should still never exceed num_valo_instances.
            if pv_segm_tf.max().item() > K:
                raise ValueError(f'K = num_valo_instances: {K} pv_segm_tf: {pv_segm_tf.unique()}')

            # Finally, due to some historical issues, ensure that query and target are consistent.
            for k in range(K):
                num_visible_pxl = (pv_segm_tf[0] == k + 1).sum()
                num_total_pxl = (pv_div_segm_tf[k] == 1).sum()
                overlap = torch.logical_and(pv_segm_tf[0] == k + 1, pv_div_segm_tf[k] == 1).sum()
                if (num_visible_pxl > 0 and num_total_pxl > 0) and \
                        (overlap == 0 or num_visible_pxl >= num_total_pxl * 1.1):
                    raise ValueError(f'Mismatch between pv_segm_tf (query) and pv_div_segm_tf!'
                                     f'scene_dp: {scene_dp} k: {k} K: {K} '
                                     f'num_visible_pxl: {num_visible_pxl} '
                                     f'num_total_pxl: {num_total_pxl} overlap: {overlap}')

        # Now that we have many hard filters in desirability, check we have sufficient amount for
        # pipeline.
        if (desirability_tf[:K, 0] > 0.0).sum() < self.num_queries:
            raise ValueError(f'[SkipCache] Insufficient number of valid queries available! '
                             f'scene_dp: {scene_dp} '
                             f'desirability_tf: {desirability_tf[:K, 0]} '
                             f'num_queries: {self.num_queries}')

    def _sample_query_data(self, augs_params):
        '''
        Creates a set of random query UV + time coordinates in image space for particle tracking.
            This method ensures that we can maintain consistency across perturbations.
        :return query_data (tuple).
        '''
        # For now, we are always grounding the trajectory at a single fixed point in time.
        query_time = np.floor(self.query_time * self.num_frames_load).astype(np.int32)
        query_time = np.ones(self.num_queries, dtype=np.int32) * query_time
        # (Q) ints in [0, T - 1].

        if self.query_size == 'point':

            # Generate (example) uniformly random pixel queries.
            query_uv = np.random.rand(self.num_queries, 2)
            # (Q, 2) floats in [0, 1].

            # Ensure that after flipping and cropping, the query coordinates are still in the image.
            # TODX: Look at augs_params instead, since this is just a cheap substitute.
            query_uv = query_uv * 0.8 + 0.1

            return (query_time, query_uv)

        elif self.query_size == 'thing':

            # Query mask order follows instance IDs (which are sorted by visibility), so Q = K and
            # nothing more must be done here.
            return query_time[0]

    def _get_thing_traject_desirability(self, div_segm, occl_fracs, query_time):
        '''
        NOTE: Some desirability values will be negative, which is a signal for pipeline that they
            should be always skipped.
        :param div_segm (K, Tc, Hf, Wf) tensor of uint8 in [0, 1].
        :param occl_fracs (K, Tc, 3) array of float32 with (f, v, t).
        :param query_time (int).
        :return desirability (K, 7) array of float32.
        '''
        (K, T, H, W) = div_segm.shape
        desirability = np.zeros((K, 7))  # Q = K = number of VALO foreground instances.

        for k in range(K):
            # Determine the average soft occlusion percentage (strictly by other objects) over time;
            # out-of-frame does not count.
            avg_occl_frac = np.mean(occl_fracs[k, :, 0])

            # Measure total variation of visible mask (normalized by its area) over time. This
            # suggests complex motion, rotation, and/or dynamic occlusion patterns.
            # NOTE: Unfortunately, this has a bias towards things with holes in them.
            delta_mask = torch.abs(div_segm[k, 1:] - div_segm[k, :-1]).type(torch.float32)
            delta_mask = (delta_mask != 0).type(torch.float32)
            max_area = div_segm[k].sum(dim=(1, 2)).max().item() / (H * W)
            old_total_var_mask = torch.mean(delta_mask).item() * 100.0
            norm_total_var_mask = torch.mean(delta_mask).item() / (max_area + 1e-6)

            # Ensure we avoid tracking insignificant objects by imposing a soft threshold on the
            # minimum number of visible pixels. The factor implies that if we are below 1% of the
            # image dimension on average, a strong penalty is applied.
            significance_hard = np.mean(occl_fracs[k, :, 1])
            significance_hard = min(significance_hard * 10000.0, 1.0) - 1.0

            # Similarly, ensure that the instance is visible by at least 2% of the image dimension
            # in the first frame, since we are doing supervised tracking.
            init_vis_size_soft = np.mean(occl_fracs[k, query_time, 1])
            init_vis_size_hard = min(init_vis_size_soft * 2500.0, 1.0) - 1.0

            # NEW:
            # Prefer objects that are mostly visible at query time, to avoid tricking the tracker
            # into thinking that we almost always have to segment more than just the given pixels.
            init_vis_rel_soft = 1.0 - np.mean(occl_fracs[k, query_time, 0])

            # Finally, same as the above, but enforce at least 20% visibility with strong penalty.
            init_vis_rel_hard = min(init_vis_rel_soft * 5.0, 1.0) - 1.0

            # Use weighted sum of all metrics, but also remember constituents.
            weighted = avg_occl_frac * 3.0 + norm_total_var_mask * 4.0 + \
                significance_hard * 64.0 + init_vis_size_hard * 256.0 + init_vis_rel_soft * 1.0 + \
                init_vis_rel_hard * 16.0
            desirability[k, :] = [weighted, avg_occl_frac, norm_total_var_mask, significance_hard,
                                  init_vis_size_hard, init_vis_rel_soft, init_vis_rel_hard]

        return desirability

    # def next_follow_data_retval(self, follow_retval):
    #     '''
    #     Imitate the already generated (possibly stripped) data that was returned and used earlier in
    #         time. For example, this could be an iteration of the test set that was stored to disk
    #         and later regenerated with more perturbations and/or views for analysis.
    #     '''
    #     self.follow_retval = follow_retval

    # def _load_example_from_follow(self, follow_retval):
    #     '''
    #     :return kubric_retvals (list of dict).
    #     '''

    #     # TODX look at notebooks etc

    #     raise NotImplementedError()
