'''
Live / online / interactive data generation logic.
Created by Basile Van Hoorick, Jul 2022.
'''

from __init__ import *

# Library imports.
import json

# Internal imports.
import data_utils
import kubric_sim


class MyKubricGenerator:
    '''
    Provides interface between interactive Kubric control or bootstrap data loading and model
    training functionality. Implements the actual behavior of the action space. While this class
    allows for offline loading, most of its functionality is directed toward online generation.
    '''

    def __init__(self, logger, phase, seed_offset, bootstrap_path='', num_frames_initial=2,
                 num_frames_total=12, frame_height=256, frame_width=320, frame_rate=6,
                 render_samples_per_pixel=32, avg_static_objects=8, avg_dynamic_objects=4,
                 action_space=16, augs_2d=True, bootstrap_only=False, verbosity=1):
        '''
        :param logger (MyLogger).
        :param bootstrap_path (str): Optional path to warmup dataset exported from Kubric.
        :param num_frames (int): Video clip length.
        :param frame_height, frame_width (int): Size of returned clips.
        '''
        # Define color and final resize transforms.
        to_tensor = torchvision.transforms.ToTensor()
        pre_transform = torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)
        post_transform = torchvision.transforms.Resize((frame_height, frame_width))

        # Video / clip options.
        self.num_frames_initial = num_frames_initial
        self.num_frames_total = num_frames_total
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_rate = frame_rate
        self.render_samples_per_pixel = render_samples_per_pixel
        self.avg_static_objects = avg_static_objects
        self.avg_dynamic_objects = avg_dynamic_objects
        self.action_space = action_space
        self.augs_2d = augs_2d

        # Other variables.
        self.logger = logger
        self.bootstrap_path = bootstrap_path
        self.snitch_grid_size = int(round(np.sqrt(action_space)))  # 16 => 4, 25 => 5.
        self.snitch_max_extent = (self.snitch_grid_size - 1.0) / 2.0  # 16 => 1.5, 25 => 2.0.
        self.snitch_grid_step = 1.5
        self.snitch_size_meters = 0.45
        self.bootstrap_only = bootstrap_only
        self.verbosity = verbosity
        self.to_tensor = to_tensor
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        # Setup Kubric controller.
        if not bootstrap_only:
            self.kubric_controller = kubric_sim.MyKubricSimulatorRenderer(
                logger, frame_width=frame_width, frame_height=frame_height,
                num_frames=num_frames_total, frame_rate=frame_rate,
                render_samples_per_pixel=render_samples_per_pixel)

        self.set_phase(phase)
        self.set_seed_offset(seed_offset)

    def set_phase(self, phase):
        '''
        Updates which part of the bootstrap dataset to use.
        :param phase (str): Which stage of training or testing we are in.
            NOTE: Suffixes will be removed (e.g. so val_aug becomes val).
        '''
        # Correct directory names.
        if 'train' in phase:
            phase = 'train'
        elif 'val' in phase:
            phase = 'val'
        elif 'test' in phase:
            phase = 'test'
        else:
            raise ValueError(phase)

        # Get root and phase directories.
        phase_dp = os.path.join(self.bootstrap_path, phase)
        if not os.path.exists(phase_dp):
            phase_dp = self.bootstrap_path

        scene_dns = sorted(os.listdir(phase_dp))
        scene_dps = [os.path.join(phase_dp, dn) for dn in scene_dns]
        scene_dps = [dp for dp in scene_dps if os.path.isdir(dp)]
        scene_dps = [dp for dp in scene_dps if 'scn' in dp]
        scene_count = len(scene_dps)
        if self.verbosity >= 1:
            self.logger.info(f'({phase}) Bootstrap scene (video clip) count: {scene_count}')

        self.phase = phase
        self.scene_dps = scene_dps
        self.dset_size = scene_count

        # Change whether to apply random color jittering, flipping, and cropping.
        self.do_random_augs = (('train' in phase or 'val' in phase) and not('noaug' in phase))

    def set_seed_offset(self, seed_offset):
        '''
        Updates the seed offset for the interactively generated video clips.
        '''
        self.seed_offset = seed_offset

    def get_new_clip_initial(self, index):
        '''
        Simulates and renders the first few (uncontrolled) frames of a video clip.
        :param index (int): Scene index / unique substep within this epoch.
        :param policy (?) array: Hider actions to take.
        :return data_stack_initial.
        '''
        if self.bootstrap_only:
            raise RuntimeError('Can only read bootstrap clips in this state.')
        
        self.last_index = index

        self.kubric_controller.prepare_next_scene('train', self.seed_offset + index)

        self.kubric_controller.insert_static_objects(
            min_count=round(self.avg_static_objects * 0.75),
            max_count=round(self.avg_static_objects * 1.25),
            force_containers=1, force_carriers=1)

        self.kubric_controller.simulate_frames(-50, 0)

        self.kubric_controller.reset_objects_velocity_friction_restitution()

        self.kubric_controller.insert_dynamic_objects(
            min_count=round(self.avg_dynamic_objects * 0.75),
            max_count=round(self.avg_dynamic_objects * 1.25))

        self.kubric_controller.simulate_frames(0, self.num_frames_initial)

        (data_stack_initial, _) = \
            self.kubric_controller.render_frames(0, self.num_frames_initial - 1)

        # Get RGB video clip frames.
        raw_frames = data_stack_initial['rgba'][..., 0:3]

        # Apply data transforms / augmentations.
        raw_frames = self.to_tensor(raw_frames)
        raw_frames = rearrange(raw_frames, '?')
        (resize_frames, horz_flip, crop_rect) = self.apply_transforms(raw_frames)
        resize_frames = rearrange(resize_frames, 'T C H W -> C T H W')

        # Remember current transform parameters to ensure the remaining frames are consistent.
        self.last_horz_flip = horz_flip
        self.last_crop_rect = crop_rect

        # Organize & return results.
        kubric_retval = dict()
        kubric_retval['mode'] = 'kubric_initial'
        kubric_retval['mode_int'] = 1
        kubric_retval['index'] = index
        kubric_retval['scene_dp'] = 'live'
        kubric_retval['snitch_action'] = -1
        kubric_retval['snitch_insert_xy'] = None
        kubric_retval['snitch_traject'] = None
        kubric_retval['metadata'] = None
        kubric_retval['horz_flip'] = horz_flip
        kubric_retval['crop_rect'] = torch.tensor(crop_rect, dtype=torch.float32)
        kubric_retval['rgb'] = resize_frames

        return kubric_retval

    def get_new_clip_remainder(self, index, snitch_action):
        '''
        Simulates and renders the steered frames of a video clip.
        :param index (int): Unique substep within this epoch.
        :param snitch_action (?) array: Hider actions to take.
        :return (data_stack_total, metadata).
        '''
        if self.bootstrap_only:
            raise RuntimeError('Can only read bootstrap clips in this state.')
        
        assert index == self.last_index

        (snitch_x, snitch_y) = self.snitch_action_to_insert_xy(snitch_action)

        self.kubric_controller.insert_snitch(
            at_x=snitch_x, at_y=snitch_y, size_meters=self.snitch_size_meters)

        self.kubric_controller.simulate_frames(self.num_frames_initial, self.num_frames_total)

        (data_stack_total, _) = \
            self.kubric_controller.render_frames(self.num_frames_initial, self.num_frames_total - 1)

        # Get RGB video clip frames.
        raw_frames = data_stack_total['rgba'][..., 0:3]

        # Apply exactly the same data transforms / augmentations as in the initial frames.
        raw_frames = self.to_tensor(raw_frames)
        raw_frames = rearrange(raw_frames, '?')
        (resize_frames, horz_flip, crop_rect) = self.apply_transforms(
            raw_frames, horz_flip=self.last_horz_flip, crop_rect=self.last_crop_rect)
        resize_frames = rearrange(resize_frames, 'T C H W -> C T H W')

        # Get resulting snitch track.
        (metadata, _) = self.kubric_controller.get_metadata()
        snitch_traject = self.get_snitch_traject(metadata)

        # Correct ground truth to align with used augmentations.
        snitch_traject = self.apply_transforms_traject(snitch_traject, horz_flip, crop_rect)

        # Organize & return results.
        kubric_retval = dict()
        kubric_retval['mode'] = 'kubric_remainder'
        kubric_retval['mode_int'] = 2
        kubric_retval['index'] = index
        kubric_retval['scene_dp'] = 'live'
        kubric_retval['snitch_action'] = snitch_action
        kubric_retval['snitch_insert_xy'] = torch.tensor([snitch_x, snitch_y], dtype=torch.float32)
        kubric_retval['snitch_traject'] = torch.tensor(snitch_traject, dtype=torch.float32)
        kubric_retval['metadata'] = metadata
        kubric_retval['horz_flip'] = horz_flip
        kubric_retval['crop_rect'] = torch.tensor(crop_rect, dtype=torch.float32)
        kubric_retval['rgb'] = resize_frames

        return kubric_retval

    def export_all_data(self, output_dp):
        '''
        Writes all generated frames of the latest scene to the specified folder.
        '''
        self.kubric_controller.write_all_data(output_dp)

    def get_bootstrap_size(self):
        '''
        :return size (int): Number of clips in bootstrap dataset.
        '''
        return self.dset_size

    def load_bootstrap_clip(self, index):
        '''
        NOTE: Unlike other stateful functionality in this class, this method can safely be executed
            by multiple data loader processes in parallel.
        '''
        # Get relevant paths.
        src_dp = self.scene_dps[index]
        src_dn = str(pathlib.Path(src_dp).name)
        src_video_fp = os.path.join(src_dp, src_dn + '_clean.mp4')
        src_metadata_fp = os.path.join(src_dp, src_dn + '.json')

        # Load RGB video clip frames and metadata in JSON format.
        raw_frames = np.array(imageio.mimread(src_video_fp, memtest='256MB'))  # (Tt, Hf, Wf, 3).
        raw_frames = (raw_frames / 255.0).astype(np.float32)
        with open(src_metadata_fp, 'r') as f:
            metadata = json.load(f)

        # NOTE: This is temporary due to wrong stored metadata in kubcon_v1/v2.
        # NOTE: The physics / simulator still assumes 24 so we should leave it as such.
        # if '_v1_' in src_dn:
        #     metadata['scene']['frame_rate'] = 12
        # if '_v2_' in src_dn:
        #     metadata['scene']['frame_rate'] = 20

        # Apply data transforms / augmentations.
        raw_frames = torch.tensor(raw_frames, dtype=torch.float32)
        raw_frames = rearrange(raw_frames, 'T H W C -> T C H W')
        (resize_frames, horz_flip, crop_rect) = self.apply_transforms(raw_frames)
        resize_frames = rearrange(resize_frames, 'T C H W -> C T H W')

        # Get snitch taken action & resulting track.
        if 'insert_snitch_args' in metadata['scene']:
            snitch_x = metadata['scene']['insert_snitch_args']['at'][0]
            snitch_y = metadata['scene']['insert_snitch_args']['at'][1]
        else:
            (snitch_x, snitch_y) = (0.0, 0.0)  # For kubcon_v1 only.
        snitch_action = self.snitch_insert_xy_to_action(snitch_x, snitch_y)
        snitch_traject = self.get_snitch_traject(metadata)

        # Correct ground truth to align with used augmentations.
        snitch_traject = self.apply_transforms_traject(snitch_traject, horz_flip, crop_rect)

        # We want to be robust to potential mismatches between bootstrap and live time-related
        # settings, in particular clip length and frame rate, by skipping higher-resolution frames.
        # This correction step helps align the distributions for the most part.
        # frame_rate_factor = int(round(metadata['scene']['frame_rate'] / self.frame_rate))
        # NOTE: Actually, we can't change this because the hider always sees the first Ti frames
        # and the snitch is always inserted at frame Ti, regardless of frame rates.
        frame_rate_factor = 1
        resize_frames = resize_frames[:, ::frame_rate_factor]
        resize_frames = resize_frames[:, :self.num_frames_total]
        snitch_traject = snitch_traject[::frame_rate_factor]
        snitch_traject = snitch_traject[:self.num_frames_total]
        assert resize_frames.shape[1] == self.num_frames_total
        assert snitch_traject.shape[0] == self.num_frames_total

        # Organize & return results.
        kubric_retval = dict()
        kubric_retval['mode'] = 'bootstrap'
        kubric_retval['mode_int'] = 0
        kubric_retval['index'] = index
        kubric_retval['scene_dp'] = src_dp
        kubric_retval['snitch_action'] = snitch_action
        kubric_retval['snitch_insert_xy'] = torch.tensor([snitch_x, snitch_y], dtype=torch.float32)
        kubric_retval['snitch_traject'] = torch.tensor(snitch_traject, dtype=torch.float32)  # (Tt, 2).
        kubric_retval['metadata'] = metadata
        kubric_retval['horz_flip'] = horz_flip
        kubric_retval['crop_rect'] = torch.tensor(crop_rect, dtype=torch.float32)
        kubric_retval['rgb'] = resize_frames  # (3, Tt, Hf, Wf).

        return kubric_retval

    def apply_transforms(self, raw_frames, horz_flip=None, crop_rect=None):
        '''
        :param raw_frames (T, 3, H, W) tensor with float32 values in [0, 1].
        :return (resize_frames, horz_flip, crop_rect).
            resize_frames (T, 3, H, W) tensor.
            horz_flip (bool).
            crop_rect (tuple of float): (y1, y2, x1, x2).
        '''
        (T, C, H, W) = raw_frames.shape
        distort_frames = raw_frames

        if self.do_random_augs:
            # Apply color perturbation (train time only).
            # NOTE: The 3 trailing dimensions have to be (1 or 3, H, W) for this to work correctly.
            if np.random.rand() < 0.9:
                distort_frames = self.pre_transform(distort_frames)

        if self.do_random_augs and self.augs_2d:
            # Apply random horizontal flip (train time only).
            if horz_flip is None:
                horz_flip = (np.random.rand() < 0.5)
            if horz_flip:
                distort_frames = torch.flip(distort_frames, dims=[-1])

            # Apply crops (train time only).
            if crop_rect is None:
                crop_y1 = np.random.rand() * 0.1
                crop_y2 = np.random.rand() * 0.1 + 0.9
                crop_x1 = np.random.rand() * 0.1
                crop_x2 = np.random.rand() * 0.1 + 0.9
                crop_rect = (crop_y1, crop_y2, crop_x1, crop_x2)
            else:
                (crop_y1, crop_y2, crop_x1, crop_x2) = crop_rect
            if np.all(np.array(crop_rect) >= 0.0):
                crop_frames = distort_frames[..., int(crop_y1 * H):int(crop_y2 * H),
                                            int(crop_x1 * W):int(crop_x2 * W)]
                distort_frames = crop_frames

        else:
            # Either random augs or spatial augs specifically are turned off.
            # Indicate that no changes occur, but cannot be None because of tensors.
            horz_flip = False
            (crop_y1, crop_y2, crop_x1, crop_x2) = (-1, -1, -1, -1)
            crop_rect = (crop_y1, crop_y2, crop_x1, crop_x2)
            
        # Resize to final size (always).
        resize_frames = self.post_transform(distort_frames)

        return (resize_frames, horz_flip, crop_rect)

    def apply_transforms_traject(self, traject_noaug, horz_flip, crop_rect):
        '''
        :param traject_noaug (Tt, 2) array with (x, y).
        :param horz_flip (bool).
        :param crop_rect (tuple of float): (y1, y2, x1, x2).
        :return traject_aug (Tt, 2) array with (x, y).
        '''
        traject_aug = traject_noaug.copy()
        
        # Replicate horizontal flip.
        if horz_flip:
            traject_aug[:, 0] = 1.0 - traject_aug[:, 0]

        # Counter crop by "uncropping" the 2D positions.
        if crop_rect is not None and np.all(np.array(crop_rect) >= 0.0):
            traject_aug[:, 0] = (traject_aug[:, 0] - crop_rect[2]) / (crop_rect[3] - crop_rect[2])
            traject_aug[:, 1] = (traject_aug[:, 1] - crop_rect[0]) / (crop_rect[1] - crop_rect[0])
        
        return traject_aug

    def snitch_action_to_insert_xy(self, action):
        '''
        X
        '''
        action_x = action % self.snitch_grid_size
        action_y = (action // self.snitch_grid_size) % self.snitch_grid_size
        insert_x = (action_x - self.snitch_max_extent) * self.snitch_grid_step
        insert_y = (action_y - self.snitch_max_extent) * self.snitch_grid_step
        return (insert_x, insert_y)

    def snitch_insert_xy_to_action(self, insert_x, insert_y):
        '''
        X
        '''
        action_x = int(round(insert_x / self.snitch_grid_step + self.snitch_max_extent))
        action_y = int(round(insert_y / self.snitch_grid_step + self.snitch_max_extent))
        action = action_x + action_y * self.snitch_grid_size
        return action

    def get_snitch_traject(self, kubric_metadata):
        '''
        :return traject (Tt, 2) array with (x, y).
        '''
        snitch_metadata = [x for x in kubric_metadata['instances']
                           if 'is_snitch' in x and x['is_snitch']][0]
        traject = np.array(snitch_metadata['image_positions'], dtype=np.float32)
        return traject
