'''
Real-world data loading and processing logic.
Created by Basile Van Hoorick, Sep 2022.
'''

from __init__ import *

# Library imports.
import json

# Internal imports.
import augs
import data_utils
import data_vis
import my_utils


def prefetch_video_infos(video_dns, frames_dp, annots_dp, metadata, num_frames, query_time_idx,
                         frame_stride):
    video_infos = dict()
    video_infos['usage_modes'] = []
    video_infos['sample_bias'] = []  # Unnormalized preference for each example.
    video_infos['num_objects'] = []

    # TODX maybe bug here because I'm still hitting files that don't exist??

    for video_dn in video_dns:
        video_frames_dp = os.path.join(frames_dp, video_dn)
        video_annots_dp = os.path.join(annots_dp, video_dn)
        num_objects = len(metadata['videos'][video_dn]['objects'])

        # Calculate usage modes.
        frame_fns = sorted(os.listdir(video_frames_dp))
        annot_fns = sorted(os.listdir(video_annots_dp))
        available_input_inds = sorted([int(fn.split('.')[0]) for fn in frame_fns])
        available_target_inds = sorted([int(fn.split('.')[0]) for fn in annot_fns])
        available_query_inds = copy.deepcopy(available_target_inds)
        usage_modes = data_utils.get_usage_modes(
            available_input_inds, available_query_inds, available_target_inds,
            num_frames, query_time_idx, min_target_frames_covered=2)
        aligned_usage_modes = [um for um in usage_modes if um[1] == frame_stride]

        # Calculate preference based on FPS alignment, target coverage, and query instance
        # diversity.
        # TODX include more factors (which?)
        sample_bias = 1.0
        sample_bias += len(aligned_usage_modes) / 2.0
        sample_bias += sum([um[2] for um in aligned_usage_modes])
        sample_bias += num_objects

        # Update info dictionary.
        video_infos['usage_modes'].append(usage_modes)
        video_infos['sample_bias'].append(sample_bias)
        video_infos['num_objects'].append(num_objects)

    return video_infos


class YoutubeVOSDataset(torch.utils.data.Dataset):
    '''
    X
    '''

    def __init__(self, dset_root, logger, phase, num_frames=20, frame_height=240, frame_width=320,
                 frame_rate=24, frame_stride=2, use_data_frac=1.0, augs_2d=True, query_time=0.2,
                 max_objects=12, augs_version=3):
        '''
        Initializes the dataset.
        '''
        self.dset_root = dset_root
        self.logger = logger
        self.phase = phase
        self.use_data_frac = use_data_frac

        # Final clip options.
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_rate = frame_rate
        self.frame_stride = frame_stride
        self.augs_2d = augs_2d
        self.query_time_val = query_time
        self.query_time_idx = int(np.floor(query_time * num_frames))
        self.max_objects = max_objects
        self.augs_version = augs_version

        # Change whether to apply random color jittering, flipping, and cropping.
        self.do_random_augs = (('train' in phase or 'val' in phase) and not('noaug' in phase))
        self.to_tensor = torchvision.transforms.ToTensor()

        # Get phase name with respect to file system.
        if 'train' in phase:
            phase_dn = 'train'
        elif 'val' in phase:
            phase_dn = 'val'
        else:
            raise ValueError(phase)
        self.frames_dp = os.path.join(dset_root, f'{phase_dn}_all_frames', 'JPEGImages')
        self.annots_dp = os.path.join(dset_root, f'{phase_dn}', 'Annotations')

        # Load global metadata.
        metadata_fp = os.path.join(dset_root, f'{phase_dn}', 'meta.json')
        with open(metadata_fp, 'r') as f:
            self.metadata = json.load(f)

        # Get video subdirectory names.
        video_dns = sorted(os.listdir(self.frames_dp))

        # Instantiate custom augmentation pipeline.
        self.augs_pipeline = augs.MyAugmentationPipeline(
            self.logger, self.num_frames, self.num_frames, self.frame_height, self.frame_width,
                 self.frame_stride, self.do_random_augs, self.augs_2d, False, self.augs_version,
                 0.0, 0.0, False)

        # Prefetch available video to clip subsampling modes, incorporating query times & annotated
        # frames.
        self.logger.info(
            f'(YoutubeVOSDataset) ({phase}) Prefetching input / target frame index info...')
        
        # This call takes about 30 seconds.
        # video_infos = prefetch_video_infos(
        #     video_dns, self.frames_dp, self.annots_dp, self.metadata, self.num_frames,
        #     self.query_time_idx, self.frame_stride)
        newer_than = 0.0  # Last change: N/A.
        cache_fn = f'cc_prefetch_{self.num_frames}_{self.query_time_idx}_{self.frame_stride}.p'
        cache_fp = os.path.join(self.dset_root, 'cache', cache_fn)
        video_infos = my_utils.disk_cached_call(
            self.logger, cache_fp, newer_than, prefetch_video_infos,
            video_dns, self.frames_dp, self.annots_dp, self.metadata, self.num_frames,
            self.query_time_idx, self.frame_stride)
        
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        manager = mp.Manager()
        video_dns = manager.list(video_dns)
        video_infos = manager.dict(video_infos)

        # Assign variables.
        num_videos = len(video_dns)
        self.video_dns = video_dns
        self.video_infos = video_infos
        self.dset_size = num_videos
        self.used_dset_size = int(use_data_frac * num_videos)
        self.force_shuffle = (use_data_frac < 1.0 and ('train' in phase or 'val' in phase))

        self.logger.info(f'(YoutubeVOSDataset) ({phase}) Video count: {num_videos}')
        self.logger.info(
            f'(YoutubeVOSDataset) ({phase}) Used dataset size: {self.used_dset_size}')

    def __len__(self):
        return self.used_dset_size

    def __getitem__(self, index):
        '''
        X
        '''
        # Some files / scenes may be invalid, so we keep retrying (up to an upper bound).
        retries = 0
        video_idx = -1

        while True:
            try:
                if not(self.force_shuffle) and retries == 0:
                    video_idx = index % self.dset_size
                else:
                    # Sample proportionally to precalculated preference.
                    sample_bias = np.array(self.video_infos['sample_bias'])
                    sample_probs = sample_bias / sample_bias.sum()
                    video_idx = np.random.choice(self.dset_size, p=sample_probs)

                data_retval = self._load_example(video_idx)

                break  # We are successful if we reach this.

            except Exception as e:
                retries += 1
                self.logger.warning(f'(YoutubeVOSDataset) video_idx: {video_idx}')
                self.logger.warning(f'(YoutubeVOSDataset) {str(e)}')
                self.logger.warning(f'(YoutubeVOSDataset) retries: {retries}')
                if retries >= 8:
                    raise e

        data_retval['source_name'] = 'ytvos'
        data_retval['dset_idx'] = index
        data_retval['retries'] = retries
        data_retval['scene_idx'] = video_idx

        return data_retval

    def _load_example(self, video_idx):
        '''
        :return data_retval (dict).
        '''
        usage_modes = self.video_infos['usage_modes'][video_idx]
        target_coverages = np.array([um[2] for um in usage_modes])
        sample_probs = target_coverages / target_coverages.sum()

        # Generate non-deteministic info beforehand to ensure consistency.
        # NOTE: For mask_track, this is currently the only part where randomness occurs (and is
        # allowed)!
        usage_mode_idx = np.random.choice(len(usage_modes), p=sample_probs)
        usage_mode = usage_modes[usage_mode_idx]
        (frame_start, frame_stride) = (usage_mode[0], usage_mode[1])
        frame_inds = list(range(
            frame_start, self.num_frames * frame_stride + frame_start, frame_stride))
        augs_params = self.augs_pipeline.sample_augs_params()
        # ^ frame_inds_load and frame_inds_clip are ignored!

        # Actually load the video.
        # data_retval = self._load_example_deterministic_cache_failsafe(
        #     video_idx, usage_mode, frame_inds, augs_params)
        data_retval = self._load_example_deterministic(
            video_idx, usage_mode, frame_inds, augs_params, False)

        return data_retval

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
                            f'(YoutubeVOSDataset) _load_example_deterministic failed ({str(e)}), '
                            f'setting force_renew_cache...')
                    else:
                        raise e
                elif retry >= 1:
                    raise e

    def _load_example_deterministic(self, video_idx, usage_mode, frame_inds, augs_params,
                                    force_renew_cache):
        '''
        Loads an entire scene clip with given (random) augmentations.
        :param video_idx (int).
        :param usage_mode (tuple).
        :param frame_inds (list of int).
        :param augs_params (dict).
        :return data_retval (dict).
        '''
        # =============================================
        # Part 1: Loading and preprocessing (in numpy).
        # =============================================
        temp_st = time.time()  # DEBUG

        # This call takes about 40-70 seconds.
        preprocess_retval = self._load_example_preprocess(video_idx, usage_mode, frame_inds)
        # newer_than = 0.0  # Last change: N/A.
        # video_dn = copy.deepcopy(self.video_dns[video_idx])
        # cache_fn = f'cc_{video_dn}_{frame_inds[0]}_{frame_inds[1]}_{frame_inds[-1]}.p'
        # cache_fp = os.path.join(self.dset_root, 'cache', cache_fn)
        # if force_renew_cache and os.path.exists(cache_fp):
        #     os.remove(cache_fp)
        # preprocess_retval = my_utils.disk_cached_call(
        #     self.logger, cache_fp, newer_than, self._load_example_preprocess,
        #     video_idx, usage_mode, frame_inds)
        
        self.logger.debug(
            f'(YoutubeVOSDataset) _load_example_preprocess: {time.time() - temp_st:.3f}s')  # DEBUG

        # ===================================================
        # Part 2: Augmentation and postprocessing (in torch).
        # ===================================================
        temp_st = time.time()  # DEBUG
        data_retval = self._load_example_augmentations(preprocess_retval, augs_params)
        self.logger.debug(
            f'(YoutubeVOSDataset) _load_example_augmentations: {time.time() - temp_st:.3f}s')  # DEBUG

        # ============================
        # Part 3: Final sanity checks.
        # ============================
        # temp_st = time.time()  # DEBUG
        self._load_example_verify(data_retval)
        # self.logger.debug(
        #     f'(YoutubeVOSDataset) _load_example_verify: {time.time() - temp_st:.3f}s')  # DEBUG

        return data_retval

    def _load_example_preprocess(self, video_idx, usage_mode, frame_inds):
        '''
        Data loading part in numpy that has no randomness or augmentations or query dependence.
            NOTE: This method is typically cached, so we should not be afraid of expensive
            calculations here.
        :return preprocess_retval (dict): Partially filled results + intermediate variables.
        '''
        # Get relevant paths.
        video_dn = copy.deepcopy(self.video_dns[video_idx])
        video_frames_dp = os.path.join(self.frames_dp, video_dn)
        video_annots_dp = os.path.join(self.annots_dp, video_dn)
        inst_count = self.video_infos['num_objects'][video_idx]

        # Load all segmentation frames that exist on disk to maximize available information.
        avail_time_inds = []
        avail_segm = []
        for segm_fn in sorted(os.listdir(video_annots_dp)):
            if segm_fn.lower()[-4:] == '.png':
                time_idx = int(segm_fn.split('.')[0])
                segm_fp = os.path.join(video_annots_dp, segm_fn)
                segm = plt.imread(segm_fp)[..., 0:3]  # (H, W, 3) floats.
                avail_time_inds.append(time_idx)
                avail_segm.append(segm)

        avail_segm = np.stack(avail_segm, axis=0)  # (Tv, Hf, Wf, 3) floats in [0, 1].

        # Convert segmentation from raw RGB to instance IDs.
        (avail_segm, unique_hues) = data_vis.segm_rgb_to_ids_ytvos(avail_segm, None)
        # (T, Hf, Wf, 1) ints in [0, K].
        if avail_segm.max() != inst_count:
            self.logger.warning(f'(YoutubeVOSDataset) Inconsistent instance count? '
                                f'segm {avail_segm.max()} vs. metadata {inst_count}')

        # Judge approximate occlusion risk per frame.
        avail_occl_risk = data_utils.get_approx_occlusion_risk(avail_segm, inst_count)
        # (K, Tv, 2) of float32 with (percentage, increase).

        # Load all RGB + segmentation + video clip frames.
        pv_rgb = []
        pv_segm = []

        # Loop over all frames.
        for f, t in enumerate(frame_inds):
            rgb_fp = os.path.join(video_frames_dp, f'{t:05d}.jpg')
            segm_fp = os.path.join(video_annots_dp, f'{t:05d}.png')
            rgb = plt.imread(rgb_fp)[..., 0:3]  # (H, W, 3) ints.
            rgb = (rgb / 255.0).astype(np.float32)   # (H, W, 3) floats.

            if os.path.exists(segm_fp):
                segm = plt.imread(segm_fp)[..., 0:3]  # (H, W, 3) floats.
            else:
                segm = np.ones_like(rgb) * (-1.0)  # (H, W, 3) floats.

            pv_rgb.append(rgb)
            pv_segm.append(segm)

        pv_rgb = np.stack(pv_rgb, axis=0)  # (Tc, Hf, Wf, 3) floats in [0, 1].
        pv_segm = np.stack(pv_segm, axis=0)  # (Tc, Hf, Wf, 3) floats in [-1, 1].

        # Convert segmentation from raw RGB to instance IDs, which also takes care of non-annotated
        # frames. Make sure to reuse the same hue list as avail_segm to avoid indexing bugs.
        (pv_segm, _) = data_vis.segm_rgb_to_ids_ytvos(pv_segm, unique_hues)
        # (Tc, Hf, Wf, 1) ints in [-1, K].

        # Get subsampled occlusion risk for frames actually used in this clip.
        occl_risk = data_utils.subsample_occlusion_risk(
            avail_occl_risk, avail_time_inds, frame_inds)
        # (K, Tc, 2) of float32 with (percentage, increase).

        # Organize & return results gathered so far.
        data_retval = dict()
        data_retval['scene_dn'] = video_dn
        data_retval['video_frames_dp'] = video_frames_dp
        data_retval['video_annots_dp'] = video_annots_dp
        data_retval['usage_mode'] = usage_mode
        data_retval['frame_inds'] = frame_inds
        data_retval['inst_count'] = inst_count
        data_retval['occl_risk'] = occl_risk

        preprocess_retval = dict()
        preprocess_retval['data_retval'] = data_retval
        preprocess_retval['pv_rgb'] = pv_rgb
        preprocess_retval['pv_segm'] = pv_segm

        return preprocess_retval

    def _load_example_augmentations(self, preprocess_retval, augs_params):
        '''
        Data loading part in torch after reading from disk and preprocessing.
        :return data_retval (dict): Completely filled results.
        '''
        data_retval = preprocess_retval['data_retval']
        pv_rgb = preprocess_retval['pv_rgb']
        pv_segm = preprocess_retval['pv_segm']
        inst_count = data_retval['inst_count']

        # Convert large numpy arrays to torch tensors, putting channel dimension first.
        pv_rgb_tf = rearrange(torch.tensor(pv_rgb, dtype=torch.float32), 'T H W C -> C T H W')
        pv_segm_tf = rearrange(torch.tensor(pv_segm, dtype=torch.int32), 'T H W C -> C T H W')

        # Apply 2D data transforms / augmentations.
        # NOTE: We must apply the transforms consistently across all modalities.
        modalities = {'rgb': pv_rgb_tf, 'segm': pv_segm_tf}
        modalities_tf = self.augs_pipeline.apply_augs_2d_frames(modalities, augs_params)
        (pv_rgb_tf, pv_segm_tf) = (modalities_tf['rgb'], modalities_tf['segm'])

        # Get pixel count per object per frame, such that we can skip all-zero query cases later.
        inst_area = data_utils.get_inst_area(pv_segm_tf, inst_count)  # (K, T).
        # NOTE: & = 0 at frames where no annotation is available.

        # Finally, make all array & tensor sizes uniform such that data can be collated.
        (occl_risk, _) = data_utils.pad_div_numpy(data_retval['occl_risk'], [0], self.max_objects)
        (inst_area, _) = data_utils.pad_div_numpy(inst_area, [0], self.max_objects)

        # Append results to include augmented data.
        data_retval['occl_risk'] = occl_risk  # (M, T, 2) of float32 with (percentage, increase).
        data_retval['inst_area'] = inst_area  # (M, T) of float32 in [0, 1].
        data_retval['query_time'] = self.query_time_idx  # (1).
        data_retval['pv_rgb_tf'] = pv_rgb_tf  # (3, T, Hf, Wf).
        data_retval['pv_segm_tf'] = pv_segm_tf  # (1, T, Hf, Wf).

        return data_retval

    def _load_example_verify(self, data_retval):
        '''
        Data loading part in torch after reading from disk and preprocessing.
        :param data_retval (dict): Completely filled results.
        '''
        # TODX really check for all zero here, and skip those
        pv_segm_tf = data_retval['pv_segm_tf']
        inst_count = data_retval['inst_count']
        inst_area = data_retval['inst_area']
        
        query_areas = inst_area[:inst_count, self.query_time_idx]  #  (K).
        if np.all(query_areas <= 0.005):
            raise ValueError('[SkipCache] All annotated objects are too small (or non-existent) at query time.')
        
        pass
