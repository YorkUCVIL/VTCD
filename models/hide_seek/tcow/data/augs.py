'''
Data augmentation / transform logic.
Created by Basile Van Hoorick, Aug 2022.
'''

from models.hide_seek.tcow.__init__ import *

# Internal imports.
import models.hide_seek.tcow.utils.geometry as geometry


class MyAugmentationPipeline:

    def __init__(self, logger, num_frames_load, num_frames_clip, frame_height, frame_width,
                 frame_stride, do_random_augs, augs_2d, augs_3d, augs_version, reverse_prob,
                 palindrome_prob, center_crop):
        '''
        Initializes the data augmentation pipeline.
        '''
        self.logger = logger
        self.num_frames_load = num_frames_load
        self.num_frames_clip = num_frames_clip
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_stride = frame_stride
        self.do_random_augs = do_random_augs
        self.augs_2d = augs_2d
        self.augs_3d = augs_3d
        self.augs_version = augs_version
        self.reverse_prob = reverse_prob
        self.palindrome_prob = palindrome_prob
        self.center_crop = center_crop

        # Define color and resize transforms. Crop is handled at runtime.
        # If augs_2d, we apply random color jittering, crops, and flips in __getitem__().
        # If augs_3d, we also apply random noise and yaw rotations.
        if augs_version >= 2:
            self.color_transform = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        elif augs_version == 1:
            self.color_transform = torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)

        if augs_version >= 2:
            self.blur_transform = torchvision.transforms.GaussianBlur(5, sigma=(0.1, 3.5))
        elif augs_version == 1:
            self.blur_transform = None

        if augs_version >= 4:
            self.grayscale_transform = torchvision.transforms.Grayscale(num_output_channels=3)
        elif augs_version <= 3:
            self.grayscale_transform = None

        # NOTE: For depth, segmentation, object coordinates, and world coordinates, it is crucial
        # that we avoid introducing unrealistic values that interpolate between big jumps in the
        # source arrays. Therefore, NEAREST seems to be the best option for all except RGB itself.
        # NOTE: In practice, we apply smooth to all except segmentation, which is nearest.
        if augs_version >= 2:
            self.post_resize_smooth = torchvision.transforms.Resize(
                (frame_height, frame_width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        elif augs_version == 1:
            self.post_resize_smooth = torchvision.transforms.Resize(
                (frame_height, frame_width),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.post_resize_nearest = torchvision.transforms.Resize(
            (frame_height, frame_width),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)

    def sample_augs_params(self):
        '''
        Creates a set of random numbers to use for 2D and/or 3D data augmentation. This method
            ensures that we can maintain consistency across modalities and perturbations.
        :return augs_params (dict).
            If a particular value is None, it means that that augmentation is disabled and no
            transformation should be applied. Otherwise, the value indicates the parameters with
            which to apply the transformation (which could still end up leaving some examples
            unchanged, for example if a flip is sampled as False).
        '''
        # Address temporal things first.
        # Offset index is typically in [0, max_delay], but is fixed to max_delay // 2 at test time.
        # However, if palindrome is True and/or frame_stride_factor > 1, then the range of offset
        # values changes depending on the configuration, so we must simultaneously infer lists of
        # actual frame indices to load and return for simplicity.
        # NOTE: If not kubric, then max_delay = 0 so offset = 0, and all other things are disabled.
        palindrome = False
        reverse = False
        frame_stride_factor = 1
        offset = (self.num_frames_load - self.num_frames_clip) // 2
        
        # NOTE: While frame_inds_load refers to file names, frame_inds_clip refers to positions
        # within frame_inds_load!
        frame_inds_load = list(range(0, self.num_frames_load * self.frame_stride, self.frame_stride))
        frame_inds_clip = list(range(0, self.num_frames_clip))
        
        if self.do_random_augs:

            palindrome = (np.random.rand() < self.palindrome_prob)
            if palindrome:
                reverse = (np.random.rand() < 0.35)
                frame_stride_factor = (2 if np.random.rand() < 0.35 else 1)
            else:
                reverse = (np.random.rand() < self.reverse_prob)
                frame_stride_factor = 1

            if palindrome:
                frame_inds_clip = frame_inds_clip + frame_inds_clip[::-1][1:]
            if reverse:
                frame_inds_clip = frame_inds_clip[::-1]
            if frame_stride_factor > 1:
                frame_inds_clip = frame_inds_clip[::frame_stride_factor]
            
            # Determine offset, now that the number of available frames may have changed.
            num_frames_avail = len(frame_inds_clip)
            assert num_frames_avail >= self.num_frames_clip
            offset = np.random.randint(0, num_frames_avail - self.num_frames_clip + 1)
            frame_inds_clip = frame_inds_clip[offset:offset + self.num_frames_clip]
        
        # Create dictionary with info.
        augs_params = dict()
        augs_params['palindrome'] = palindrome
        augs_params['reverse'] = reverse
        augs_params['frame_stride_factor'] = frame_stride_factor
        augs_params['offset'] = offset
        augs_params['frame_inds_load'] = np.array(frame_inds_load)
        augs_params['frame_inds_clip'] = np.array(frame_inds_clip)
        
        # Next, address other (mostly color & spatial) augmentations. Initialize with default values
        # that correspond to identity (data loader collate does not support None!).
        color_jitter = False
        rgb_blur = False
        rgb_grayscale = False
        horz_flip = False
        crop_rect = -np.ones(4)
        scene_yaw_deg = 0.0
        scene_trans_3d = np.array([0.0, 0.0, 0.0])

        if self.do_random_augs:
            color_jitter = (np.random.rand() < 0.9)
            
            if self.augs_version >= 5:
                rgb_blur = (np.random.rand() < 0.2)
            elif self.augs_version >= 2:
                rgb_blur = (np.random.rand() < 0.3)
            else:
                rgb_blur = False

            if self.augs_version >= 5:
                rgb_grayscale = (np.random.rand() < 0.05)
            elif self.augs_version >= 4:
                rgb_grayscale = (np.random.rand() < 0.1)
            else:
                rgb_grayscale = False

            if self.augs_2d:
                horz_flip = (np.random.rand() < 0.5)

                if self.augs_version <= 2:
                    crop_y1 = np.random.rand() * 0.1
                    crop_y2 = np.random.rand() * 0.1 + 0.9
                    crop_x1 = np.random.rand() * 0.1
                    crop_x2 = np.random.rand() * 0.1 + 0.9

                elif self.augs_version >= 3:
                    crop_y1 = np.random.rand() * 0.2
                    crop_y2 = np.random.rand() * 0.2 + 0.8
                    crop_x1 = np.random.rand() * 0.2
                    crop_x2 = np.random.rand() * 0.2 + 0.8

                crop_rect = np.array([crop_y1, crop_y2, crop_x1, crop_x2])

            # NOTE: These 3D augmentations are only to increase XYZ coordinate diversity. Because
            # the camera extrinsics are updated accordingly, all videos will look exactly the same.
            if self.augs_3d:
                scene_yaw_deg = np.random.uniform(0.0, 360.0)
                scene_trans_3d = np.random.uniform([-2.0, -2.0, -1.0], [2.0, 2.0, 1.0])
                # DEBUG / TEMP:
                # scene_yaw_deg = np.random.uniform(-20.0, 20.0)
                # scene_trans_3d = np.random.uniform([0.0, 0.0, -1.0], [0.0, 0.0, 1.0])

        # Update dictionary.
        augs_params['color_jitter'] = color_jitter
        augs_params['rgb_blur'] = rgb_blur
        augs_params['rgb_grayscale'] = rgb_grayscale
        augs_params['horz_flip'] = horz_flip
        augs_params['crop_rect'] = crop_rect
        augs_params['scene_yaw_deg'] = scene_yaw_deg
        augs_params['scene_trans_3d'] = scene_trans_3d

        return augs_params

    def apply_augs_2d_frames(self, modalities_noaug, augs_params):
        '''
        :param modalities_noaug (dict): Maps frame types (rgb / segm / ...) to original
            (1/3/K, Tv, H, W) tensors.
        :param augs_params (dict): Maps transform names to values (which could be None).
        :return modalities_aug (dict): Maps frame types to augmented (1/3/K, Tc, H, W) tensors.
        '''
        modalities_aug = dict()

        for modality, raw_frames_untrim in modalities_noaug.items():

            # In some cases, some arrays explicitly do not exist (e.g. no xyz or div_segm).
            if len(raw_frames_untrim.shape) < 4:
                modalities_aug[modality] = raw_frames_untrim.clone()
                continue

            # Address temporal things first (always, but test time values are fixed).
            frame_inds_clip = augs_params['frame_inds_clip']
            assert len(frame_inds_clip) == self.num_frames_clip
            raw_frames = raw_frames_untrim[:, frame_inds_clip, :, :]
            assert raw_frames.shape[1] == self.num_frames_clip, \
                f'raw_frames: {raw_frames.shape}  num_frames_clip: {self.num_frames_clip}'
            
            (C, T, H, W) = raw_frames.shape
            assert ((C > 3) == ('div' in modality))
            distort_frames = rearrange(raw_frames, 'C T H W -> T C H W')

            # Apply center crop if needed (test time only).
            if self.center_crop:
                current_ar = W / H
                desired_ar = self.frame_width / self.frame_height
                if current_ar > desired_ar:
                    crop_tf = torchvision.transforms.CenterCrop((H, int(H * desired_ar)))
                    distort_frames = crop_tf(distort_frames)
                elif current_ar < desired_ar:
                    crop_tf = torchvision.transforms.CenterCrop((int(W / desired_ar), W))
                    distort_frames = crop_tf(distort_frames)

            # Apply color perturbation (train time only).
            # NOTE: The trailing dimensions have to be (1/3, H, W) for this to work correctly.
            if 'rgb' in modality:
                if augs_params['color_jitter']:
                    distort_frames = self.color_transform(distort_frames)
                if augs_params['rgb_blur']:
                    distort_frames = self.blur_transform(distort_frames)
                if augs_params['rgb_grayscale']:
                    distort_frames = self.grayscale_transform(distort_frames)

            # Apply random horizontal flip (train time only).
            if augs_params['horz_flip']:
                distort_frames = torch.flip(distort_frames, dims=[-1])

            # Apply crops (train time only).
            # NOTE: These values always pertain to coordinates within post-flip images.
            crop_rect = augs_params['crop_rect']
            if crop_rect is not None and np.all(np.array(crop_rect) >= 0.0):
                (crop_y1, crop_y2, crop_x1, crop_x2) = crop_rect
                crop_frames = distort_frames[..., int(crop_y1 * H):int(crop_y2 * H),
                                             int(crop_x1 * W):int(crop_x2 * W)]
                distort_frames = crop_frames

            # Resize to final size (always).
            if 'segm' in modality or 'mask' in modality:
                # Segmentation masks have integer values.
                resize_frames = self.post_resize_nearest(distort_frames)
            else:
                # RGB, depth, object coordinates.
                resize_frames = self.post_resize_smooth(distort_frames)

            resize_frames = rearrange(resize_frames, 'T C H W -> C T H W')
            modalities_aug[modality] = resize_frames

        return modalities_aug

    def apply_augs_2d_traject(self, traject_retval_noaug, augs_params):
        '''
        :param traject_retval_noaug (dict).
        :param augs_params (dict): Maps transform names to values (which could be None).
        :return traject_retval_aug (dict).
        '''
        raise NotImplementedError('frame_inds_clip')

        traject_retval_aug = copy.deepcopy(traject_retval_noaug)

        # Loop over all expected (*, 2+) arrays that have values (u, v, *).
        for uv_key in ['query_uv', 'uvdxyz', 'uvdxyz_static']:

            traject_retval_aug[uv_key + '_tf'] = copy.deepcopy(traject_retval_noaug[uv_key])

            # Replicate horizontal flip.
            if augs_params['horz_flip']:
                traject_retval_aug[uv_key + '_tf'][..., 0] = \
                    1.0 - traject_retval_aug[uv_key + '_tf'][..., 0]

            # Counter crop by "uncropping" the 2D positions.
            crop_rect = augs_params['crop_rect']
            if crop_rect is not None and np.all(np.array(crop_rect) >= 0.0):
                (crop_y1, crop_y2, crop_x1, crop_x2) = crop_rect
                traject_retval_aug[uv_key + '_tf'][..., 0] = (
                    traject_retval_aug[uv_key + '_tf'][..., 0] - crop_x1) / (crop_x2 - crop_x1)
                traject_retval_aug[uv_key + '_tf'][..., 1] = (
                    traject_retval_aug[uv_key + '_tf'][..., 1] - crop_y1) / (crop_y2 - crop_y1)

            # Correct datatype to original.
            traject_retval_aug[uv_key + '_tf'] = \
                traject_retval_aug[uv_key + '_tf'].astype(traject_retval_noaug[uv_key].dtype)

        return traject_retval_aug

    def apply_augs_2d_intrinsics(self, camera_K_noaug, augs_params):
        '''
        :param camera_K_noaug (T, 3, 3) array.
        :param augs_params (dict): Maps transform names to values (which could be None).
        '''
        raise NotImplementedError('frame_inds_clip')

        camera_K_aug = copy.deepcopy(camera_K_noaug)

        # TODX this has not been thoroughly vetted yet!

        # Replicate horizontal flip.
        if augs_params['horz_flip']:
            camera_K_aug[:, 0, 0] *= -1.0  # From positive to negative sign.

        crop_rect = augs_params['crop_rect']
        if crop_rect is not None and np.all(np.array(crop_rect) >= 0.0):
            (crop_y1, crop_y2, crop_x1, crop_x2) = crop_rect

            # Update normalized focal lengths -- these always increase when cropping.
            # https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html
            # https://ksimek.github.io/2013/08/13/intrinsic/
            camera_K_aug[:, 0, 0] /= (crop_x2 - crop_x1)
            camera_K_aug[:, 1, 1] /= (crop_y2 - crop_y1)

            # Update normalized principal points -- these have to correspond to the pinhole.
            assert np.allclose(camera_K_noaug[:, 0, 2], -0.5)
            assert np.allclose(camera_K_noaug[:, 1, 2], -0.5)
            camera_K_aug[:, 0, 2] = -(0.5 - crop_x1) / (crop_x2 - crop_x1)
            camera_K_aug[:, 1, 2] = -(0.5 - crop_y1) / (crop_y2 - crop_y1)

        camera_K_aug = camera_K_aug.astype(camera_K_noaug.dtype)
        return camera_K_aug

    def apply_augs_3d_frames(self, pv_xyz_noaug, augs_params):
        '''
        :param pv_xyz_tf (3, T, H, W) tensor.
        :param augs_params (dict): Maps transform names to values (which could be None).
        :return pv_xyz_tf (3, T, H, W) tensor.
        '''
        raise NotImplementedError('frame_inds_clip')

        pv_xyz_aug = copy.deepcopy(pv_xyz_noaug)

        # First, apply rotation in XY plane in world coordinates around the scene center.
        if augs_params['scene_yaw_deg'] is not None:
            cos_yaw = np.cos(augs_params['scene_yaw_deg'] * np.pi / 180.0)
            sin_yaw = np.sin(augs_params['scene_yaw_deg'] * np.pi / 180.0)
            pv_xyz_aug[0] = cos_yaw * pv_xyz_noaug[0] - sin_yaw * pv_xyz_noaug[1]
            pv_xyz_aug[1] = sin_yaw * pv_xyz_noaug[0] + cos_yaw * pv_xyz_noaug[1]

        # Then, apply translation noise.
        if augs_params['scene_trans_3d'] is not None:
            pv_xyz_aug += augs_params['scene_trans_3d'][:, None, None, None]

        pv_xyz_aug = pv_xyz_aug.type(pv_xyz_noaug.dtype)
        return pv_xyz_aug

    def apply_augs_3d_traject(self, traject_retval_noaug, augs_params):
        '''
        :param traject_retval_noaug (dict).
        :param augs_params (dict): Maps transform names to values (which could be None).
        :return traject_retval_aug (dict).
        '''
        raise NotImplementedError('frame_inds_clip')

        traject_retval_aug = copy.deepcopy(traject_retval_noaug)

        # Loop over all expected (*, 3+) arrays that have values (*, x, y, z).
        # NOTE: We are explicitly ignoring inst_bboxes_3d here.
        for xyz_key in ['query_xyz', 'uvdxyz', 'uvdxyz_static']:

            traject_retval_aug[xyz_key + '_tf'] = copy.deepcopy(traject_retval_noaug[xyz_key])

            # First, apply rotation in XY plane in world coordinates around the scene center.
            if augs_params['scene_yaw_deg'] is not None:
                cos_yaw = np.cos(augs_params['scene_yaw_deg'] * np.pi / 180.0)
                sin_yaw = np.sin(augs_params['scene_yaw_deg'] * np.pi / 180.0)
                traject_retval_aug[xyz_key + '_tf'][..., -3] = \
                    cos_yaw * traject_retval_noaug[xyz_key][..., -3] - \
                    sin_yaw * traject_retval_noaug[xyz_key][..., -2]
                traject_retval_aug[xyz_key + '_tf'][..., -2] = \
                    sin_yaw * traject_retval_noaug[xyz_key][..., -3] + \
                    cos_yaw * traject_retval_noaug[xyz_key][..., -2]

            # Then, apply translation noise.
            if augs_params['scene_trans_3d'] is not None:
                traject_retval_aug[xyz_key + '_tf'][..., -3] += augs_params['scene_trans_3d'][0]
                traject_retval_aug[xyz_key + '_tf'][..., -2] += augs_params['scene_trans_3d'][1]
                traject_retval_aug[xyz_key + '_tf'][..., -1] += augs_params['scene_trans_3d'][2]

            # Correct datatype to original.
            traject_retval_aug[xyz_key] = \
                traject_retval_aug[xyz_key + '_tf'].astype(traject_retval_noaug[xyz_key].dtype)

        return traject_retval_aug

    def apply_augs_3d_extrinsics(self, camera_R_noaug, augs_params):
        '''
        :param camera_R_noaug (T, 4, 4) tensor.
        :param augs_params (dict): Maps transform names to values (which could be None).
        '''
        raise NotImplementedError('frame_inds_clip')

        # camera_R_aug = copy.deepcopy(camera_R_noaug)

        # TODX this has not been thoroughly vetted yet!

        rotation_matrix = np.eye(4)
        if augs_params['scene_yaw_deg'] is not None:
            cos_yaw = np.cos(augs_params['scene_yaw_deg'] * np.pi / 180.0)
            sin_yaw = np.sin(augs_params['scene_yaw_deg'] * np.pi / 180.0)
            rotation_matrix[..., 0:2, 0:2] = [[cos_yaw, -sin_yaw],
                                              [sin_yaw, cos_yaw]]

        translation_matrix = np.eye(4)
        if augs_params['scene_trans_3d'] is not None:
            translation_matrix[..., 0:3, 3] = augs_params['scene_trans_3d']

        camera_R_aug = np.matmul(np.matmul(translation_matrix, rotation_matrix), camera_R_noaug)

        camera_R_aug = camera_R_aug.astype(camera_R_noaug.dtype)
        return camera_R_aug
