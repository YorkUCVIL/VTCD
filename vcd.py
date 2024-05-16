
import psutil
import gc
import json
from decord import VideoReader, cpu
from PIL import Image
from sklearn.metrics.cluster import silhouette_score
from threadpoolctl import threadpool_limits
from utilities.clustering import cluster_features, cluster_dataset
import models.hide_seek.tcow as tcow
from models.hide_seek.tcow.data.data_vis import *
from models.hide_seek.tcow.data.data_utils import *
from models.hide_seek.tcow.eval.metrics import calculate_metrics_mask_track, calculate_weighted_averages

def random_color():
    """Generate a random color"""
    return [random.randint(0, 255) for _ in range(3)]

class VideoConceptDiscovery(object):
    """Discovering video concepts.
    """

    def __init__(self, args, model,):
        self.args = args
        self.model = model

        # initialize transforms depending on the model
        self.initialize_transforms()

        # load dataset (optionally can be cached)
        if args.dataset == 'kubric':
            if not 'timesformer' in self.args.model:
                cached_file_path = os.path.join(self.args.kubric_path, 'val', '{}Frames_Max{}.pkl'.format(model.num_frames, args.max_num_videos))
            else:
                cached_file_path = os.path.join(self.args.kubric_path, 'val', 'Max{}.pkl'.format(self.args.max_num_videos))
            self.cached_file_path = cached_file_path
            if os.path.exists(cached_file_path) and not self.args.force_reload_videos:
                try:
                    with open(cached_file_path, 'rb') as f:
                        self.dataset = pickle.load(f)
                except:
                    print('Failed to load cached file, reloading videos...')
                    self.dataset = self.load_kubric_videos()
            else:
                self.dataset = self.load_kubric_videos()
        elif args.dataset == 'ssv2':
            if 'timesformer' in args.model:
                cached_file_path = os.path.join(self.args.ssv2_path, '{}_Max{}_{}_{}.pkl'.format(self.args.target_class, self.args.max_num_videos, 'train' if self.args.use_train else 'val', 'tcow')).replace(' ', '_')
            else:
                cached_file_path = os.path.join(self.args.ssv2_path, '{}_Max{}_{}.pkl'.format(self.args.target_class, self.args.max_num_videos, 'train' if self.args.use_train else 'val')).replace(' ', '_')
            print(cached_file_path)
            self.cached_file_path = cached_file_path
            if os.path.exists(cached_file_path) and not self.args.force_reload_videos:
                try:
                    with open(cached_file_path, 'rb') as f:
                        self.dataset = pickle.load(f)
                except:
                    print('Failed to load cached file, reloading videos...')
                    self.dataset = self.load_ssv2_videos()
            else:
                self.dataset = self.load_ssv2_videos()
                # save pkl file
                with open(cached_file_path, 'wb') as f:
                    pickle.dump(self.dataset, f)
        elif 'davis16' in args.dataset:
            cached_file_path = os.path.join(self.args.davis16_path, 'Max{}.pkl'.format(self.args.max_num_videos)).replace(' ', '_')
            print(cached_file_path)
            self.cached_file_path = cached_file_path
            if os.path.exists(cached_file_path) and not self.args.force_reload_videos:
                try:
                    with open(cached_file_path, 'rb') as f:
                        self.dataset = pickle.load(f)
                except:
                    print('Failed to load cached file, reloading videos...')
                    self.dataset = self.load_davis16_videos()
            else:
                self.dataset = self.load_davis16_videos()
                # save pkl file
                with open(cached_file_path, 'wb') as f:
                    pickle.dump(self.dataset, f)
        else:
            raise NotImplementedError

        # save pkl file
        if args.dataset_cache:
            with open(cached_file_path, 'wb') as f:
                pickle.dump(self.dataset, f)

    def initialize_transforms(self):
        if 'vidmae' in self.args.model:
            self.frame_width = self.model.default_cfg['input_size'][1]
            self.frame_height = self.model.default_cfg['input_size'][2]
            self.normalize = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        elif 'intern' in self.args.model:
            self.frame_width = 224
            self.frame_height = 224
        elif 'timesformer' in self.args.model:
            self.frame_width = 320
            self.frame_height = 240

        # resize transformers for segmentation masks
        self.post_resize_smooth = torchvision.transforms.Resize(
            (self.frame_height, self.frame_width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.post_resize_nearest = torchvision.transforms.Resize(
            (self.frame_height, self.frame_width),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        # resize transformers for video
        self.pre_upsample = torch.nn.Upsample(
            size=(int(self.model.num_frames*self.args.temporal_resize_factor), int(self.frame_height*self.args.spatial_resize_factor), int(self.frame_width*self.args.spatial_resize_factor)),
            mode='trilinear', align_corners=True)
        self.post_upsample = torch.nn.Upsample(
            size=(int(self.model.num_frames), int(self.frame_height), int(self.frame_width)),
            mode='nearest')

        # set up multiclass setting
        if len(self.args.target_class_idxs) > 1:
            self.multiclass = True
            self.args.target_class = '_'.join([str(x) for x in sorted(self.args.target_class_idxs)])
        else:
            self.multiclass = False

    def load_davis16_videos(self, sampling_rate=2, num_frames=16):
        # get video names
        videos = glob.glob(os.path.join(self.args.davis16_path, 'JPEGImages/480p/*'))
        dataset = []
        self.video_names = []
        self.labels = []
        self.seeker_query_labels = []

        train_cls_list = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 'paragliding', 'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf', 'swing', 'tennis', 'train']

        try:
            sampling_rate = self.model.sampling_rate
            num_frames = self.model.num_frames
        except:
            pass

        for vid_num, frame_path in enumerate(videos):
            # get label
            vid_name = frame_path.split('/')[-1]
            if 'val' in self.args.dataset:
                if vid_name.split('_')[0] in train_cls_list:
                    continue

            frame_paths = sorted(glob.glob(os.path.join(frame_path, '*.jpg')))
            frames = [torch.tensor(plt.imread(frame)) for frame in frame_paths]
            labels = [torch.tensor(plt.imread(frame.replace('JPEGImages/480p', 'Annotations/480p').replace('jpg', 'png'))) for frame in frame_paths]


            # sample frames every self.model.args.sampling_rate frames
            if not self.args.process_full_video:
                frames = frames[::sampling_rate]
                labels = labels[::sampling_rate]
            if len(frames) < num_frames:
                continue
            if not self.args.process_full_video:
                frames = frames[:num_frames]
                labels = labels[:num_frames]

            rgb_video = torch.stack(frames).permute(3, 0, 1, 2) / 255.0
            rgb_video = self.post_resize_smooth(rgb_video)

            # select only first channel if there are more than one
            for label_num, label in enumerate(labels):
                if len(label.shape) > 2:
                    label = label[:,:,0]
                    # replace label
                    labels[label_num] = label
            labels = torch.stack(labels)
            labels = self.post_resize_nearest(labels)
            dataset.append(rgb_video)
            self.labels.append(labels)
            # stack 29 frames of zeros after the first frame
            zeros = torch.zeros((num_frames-1, labels.shape[-2] , labels.shape[-1]))
            query_label = torch.cat([labels[0].unsqueeze(0), zeros], dim=0).unsqueeze(0)
            self.seeker_query_labels.append(query_label)
            self.video_names.append(vid_name)
            if len(dataset) == self.args.max_num_videos:
                break

        if not self.args.process_full_video:
            dataset = torch.stack(dataset, dim=0) # n x c x t x h x w
        return dataset

    def load_ssv2_videos(self):
        # get class names
        label_path = os.path.join(self.args.ssv2_path, 'something-something-v2-labels.json')

        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        idx_to_label = {v: k for k, v in label_dict.items()}
        if self.multiclass:
            cls_idx = [x for i, x in enumerate(label_dict) if i in self.args.target_class_idxs]
        else:
            cls_idx = label_dict[self.args.target_class]
            self.target_label_id = int(cls_idx)

        # open validation file
        if self.args.use_train:
            data_path = os.path.join(self.args.ssv2_path, 'something-something-v2-train.json')
        else:
            data_path = os.path.join(self.args.ssv2_path, 'something-something-v2-validation.json')
        with open(data_path, 'r') as f:
            data_dict = json.load(f)

        # get videos for target class
        video_ids = []
        video_labels = []
        if self.multiclass:
            for idx in self.args.target_class_idxs:
                target_class = idx_to_label[str(idx)]
                video_ids += [x['id'] for x in data_dict if x['template'].replace('[something]', 'something').replace('[something in it]', 'something in it') == target_class]
                video_labels += [idx for x in data_dict if x['template'].replace('[something]', 'something').replace('[something in it]', 'something in it') == target_class]
        else:
            video_ids = [x['id'] for x in data_dict if x['template'].replace('[something]', 'something') == self.args.target_class]
            video_labels = [cls_idx for x in data_dict if x['template'].replace('[something]', 'something') == self.args.target_class]
        videos = [('{}/20bn-something-something-v2/{}.webm'.format(self.args.ssv2_path, video_ids[x]), video_labels[x]) for x in range(len(video_ids))]

        random.shuffle(videos)
        dataset = []
        self.list_video_ids = []
        save_frames = []
        self.seeker_query_labels = []
        self.labels = []
        for vid_num, data in enumerate(videos):
            video = data[0]
            label = data[1]

            try:
                vr = VideoReader(video, num_threads=1, ctx=cpu(0),width = self.frame_width, height = self.frame_height)
            except:
                continue
            frames = []
            for i in range(len(vr)):
                frame = vr[i]
                try:
                    frames.append(torch.tensor(frame.asnumpy()))
                except:
                    frames.append(torch.tensor(frame))

            # sample frames every self.model.args.sampling_rate frames
            frames = frames[::self.model.sampling_rate]
            if len(frames) < self.model.num_frames:
                continue
            frames = frames[:self.model.num_frames]

            try:
                if 'timesformer' in self.args.model:
                    # open up labels
                    label_path = os.path.join('ssv2_labels', 'first_frame_labels', self.args.target_class, data[0].split('/')[-1].split('.')[0],'0_gt.png').replace(' ', '_')
                    label = np.array(Image.open(label_path))[:,:,:3]
                    label_binary = torch.tensor(np.where(label.sum(2)==765, 0, 1)).unsqueeze(0).unsqueeze(0)
                    zeros = torch.zeros((1, self.model.num_frames-1, label_binary.shape[-2] , label_binary.shape[-1]))
                    label_binary = torch.cat([label_binary, zeros], dim=1).unsqueeze(0)
                    self.seeker_query_labels.append(label_binary)
            except:
                print('no label found for {}'.format(data[0]))

            save_frames.append(frames)

            rgb_video = torch.stack(frames).permute(3, 0, 1, 2)/255.0
            dataset.append(rgb_video)
            self.labels.append(label)
            self.list_video_ids.append(int(data[0].split('/')[-1].split('.')[0]))
            if len(dataset) == self.args.max_num_videos:
                break

        dataset = torch.stack(dataset, dim=0) # n x c x t x h x w
        return dataset

    def load_kubric_videos(self, perturb_idx=0, view_idx=0, frame_inds_load=36, frame_inds_clip=30, stride=1):

        frame_inds_load = list(range(0, frame_inds_load * stride, stride))

        frame_inds_clip = list(range(0, frame_inds_clip * stride, stride))
        # load kubric videos
        video_paths = [path for path in glob.glob(os.path.join(self.args.kubric_path, 'val/*')) if 'pkl' not in path]
        random.shuffle(video_paths)
        video_paths = video_paths[:self.args.max_num_videos]
        if self.args.max_num_videos == 2:
            # video_paths = ['/data/kubcon_v10/val/kubcon_v10_scn03743', '/data/kubcon_v10/val/kubcon_v10_scn03760']
            # video_paths = ['/data/kubcon_v10/val/kubcon_v10_scn03693', '/data/kubcon_v10/val/kubcon_v10_scn03717']
            video_paths = ['/data/kubcon_v10/val/kubcon_v10_scn03725', '/data/kubcon_v10/val/kubcon_v10_scn03781']
        if self.args.max_num_videos == 1:
            video_paths = ['/data/kubcon_v10/val/kubcon_v10_scn03760']

        full_pv_rgb_tf = []
        full_pv_segm_tf = []
        full_pv_depth_tf = []
        full_pv_flow_tf = []
        full_pv_div_segm_tf = []
        full_pv_inst_count = []
        full_traject_retval_tf = []
        full_scene_dp = []
        for vid_num, path in enumerate(video_paths):
            # only load max_num_videos
            if vid_num == self.args.max_num_videos:
                break
            scene_dn = path.split('/')[-1]
            cache_fp = path
            frames_dp = os.path.join(path, f'frames_p{perturb_idx}_v{view_idx}')
            metadata_fp = os.path.join(path, scene_dn + f'_p{perturb_idx}_v{view_idx}.json')
            with open(metadata_fp, 'r') as f:
                metadata = json.load(f)
            data_ranges_fp = os.path.join(frames_dp, 'data_ranges.json')
            with open(data_ranges_fp, 'r') as f:
                data_ranges = json.load(f)
            K = metadata['scene']['num_valo_instances']
            pv_rgb = []
            pv_segm = []
            pv_depth = []
            pv_flow = []
            for k, t in enumerate(frame_inds_load):
                rgb_fp = os.path.join(frames_dp, f'rgba_{t:05d}.png')
                segm_fp = os.path.join(frames_dp, f'segmentation_{t:05d}.png')
                depth_fp = os.path.join(frames_dp, f'depth_{t:05d}.tiff')
                flow_fp = os.path.join(frames_dp, f'forward_flow_{t:05d}.png')
                rgb = plt.imread(rgb_fp)[..., 0:3]  # (H, W, 3) floats.
                segm = plt.imread(segm_fp)[..., 0:3]  # (H, W, 3) floats.
                depthm = imageio.imread(depth_fp, format="tiff")  # (H, W, 1) floats.
                flowm = plt.imread(flow_fp)[..., 0:2]  # (H, W, 2) floats.
                pv_rgb.append(rgb)
                pv_segm.append(segm)
                pv_depth.append(depthm)
                pv_flow.append(flowm)

            pv_rgb = np.stack(pv_rgb, axis=0)  # (Tv, Hf, Wf, 3) floats in [0, 1].
            pv_segm = np.stack(pv_segm, axis=0)
            pv_depth = np.stack(pv_depth, axis=0)
            pv_flow = np.stack(pv_flow, axis=0)

            flow_min = data_ranges['forward_flow']['min']
            flow_max = data_ranges['forward_flow']['max']

            pv_flow = (pv_flow * (flow_max - flow_min)) + flow_min

            pv_segm = segm_rgb_to_ids_kubric(pv_segm)  # (Tv, Hf, Wf, 1) ints in [0, inf).

            pv_div_segm = []
            for f, t in enumerate(frame_inds_load):
                per_inst_div_segm = []

                for k in range(K):
                    cur_div_segm_fp = os.path.join(
                        frames_dp, f'divided_segmentation_{k:03d}_{t:05d}.png')

                    cur_div_segm = plt.imread(cur_div_segm_fp)[..., :3]  # (H, W, 3) floats.

                    cur_div_segm = (cur_div_segm.sum(axis=-1) > 0.1).astype(np.uint8)
                    # (H, W) ints in [0, 1].
                    per_inst_div_segm.append(cur_div_segm)

                div_segm = np.stack(per_inst_div_segm, axis=-1)  # (H, W, K) bytes in [0, 1].
                pv_div_segm.append(div_segm)

            pv_div_segm = np.stack(pv_div_segm, axis=0)

            pv_rgb_tf = rearrange(torch.tensor(pv_rgb, dtype=torch.float32), 'T H W C -> C T H W')
            pv_segm_tf = rearrange(torch.tensor(pv_segm, dtype=torch.uint8), 'T H W C -> C T H W')
            try:
                pv_depth_tf = rearrange(torch.tensor(pv_depth, dtype=torch.uint8), 'T H W C -> C T H W')
            except:
                pv_depth_tf = rearrange(torch.tensor(pv_depth, dtype=torch.uint8).unsqueeze(-1), 'T H W C -> C T H W')
            pv_flow_tf = rearrange(torch.tensor(pv_flow, dtype=torch.float32), 'T H W C -> C T H W')
            pv_div_segm_tf = rearrange(torch.tensor(pv_div_segm, dtype=torch.uint8), 'T H W K -> K T H W')


            traject_retval = dict()
            occl_fracs = get_thing_occl_fracs_numpy(pv_segm, pv_div_segm)
            (occl_cont_dag, relative_order, reconst_pv_segm, reconst_error) = \
                get_thing_occl_cont_dag(pv_segm, pv_div_segm, metadata, frame_inds_load)
            # Add annotation metadata to traject_retval, useful for evaluation.
            traject_retval['occl_fracs'] = occl_fracs  # (K, Tv, 3).
            traject_retval['occl_cont_dag'] = occl_cont_dag  # (Tv, K, K, 3).
            traject_retval['query_time'] = 0

            modalities_noaug = {'rgb': pv_rgb_tf, 'segm': pv_segm_tf, 'div_segm': pv_div_segm_tf,
                                'depth': pv_depth_tf, 'flow': pv_flow_tf}

            modalities_aug = dict()
            for modality, raw_frames_untrim in modalities_noaug.items():
                raw_frames = raw_frames_untrim[:, frame_inds_clip, :, :]
                distort_frames = rearrange(raw_frames, 'C T H W -> T C H W')
                if 'segm' in modality or 'mask' in modality:
                    # Segmentation masks have integer values.
                    resize_frames = self.post_resize_nearest(distort_frames)
                else:
                    # RGB, depth, object coordinates.
                    resize_frames = self.post_resize_smooth(distort_frames)

                resize_frames = rearrange(resize_frames, 'T C H W -> C T H W')
                modalities_aug[modality] = resize_frames

            # modalities_noaug = {'rgb': pv_rgb_tf, 'segm': pv_segm_tf, 'div_segm': pv_div_segm_tf}
            pv_rgb_tf, pv_segm_tf, pv_div_segm_tf, pv_depth_tf, pv_flow_tf = modalities_aug['rgb'], modalities_aug['segm'], modalities_aug['div_segm'], modalities_aug['depth'], modalities_aug['flow']

            traject_retval_tf = copy.deepcopy(traject_retval)

            occl_fracs_tf = get_thing_occl_fracs_torch(pv_segm_tf, pv_div_segm_tf)
            occl_cont_dag_tf = traject_retval['occl_cont_dag'][frame_inds_clip]

            desirability_tf = self._get_thing_traject_desirability(pv_div_segm_tf, occl_fracs_tf, traject_retval['query_time'])

            (traject_retval_tf['occl_fracs'], _) = pad_div_numpy(
                traject_retval_tf['occl_fracs'], [0], max_size=36)
            # (K, Tv, 3) => (M, Tv, 3). NOTE: Avoid using this because of possible Tv/Tc confusion.
            (traject_retval_tf['occl_fracs_tf'], _) = pad_div_numpy(
                occl_fracs_tf, [0], max_size=36)
            # (K, Tc, 3) => (M, Tc, 3).
            (traject_retval_tf['occl_cont_dag'], _) = pad_div_numpy(
                traject_retval_tf['occl_cont_dag'], [1, 2], max_size=36)
            # (Tv, K, K, 3) => (Tv, M, M, 3). NOTE: Avoid using this because of possible Tv/Tc confusion.
            (traject_retval_tf['occl_cont_dag_tf'], _) = pad_div_numpy(
                occl_cont_dag_tf, [1, 2], max_size=36)
            # (Tc, K, K, 3) => (Tc, M, M, 3).
            (traject_retval_tf['desirability_tf'], _) = pad_div_numpy(
                desirability_tf, [0], max_size=36)

            (pv_div_segm_tf, _) = tcow.data.data_utils.pad_div_torch(
                pv_div_segm_tf, [0], max_size=36)
            pv_inst_count = torch.tensor([K], dtype=torch.int32)

            full_pv_rgb_tf.append(pv_rgb_tf)
            full_pv_segm_tf.append(pv_segm_tf)
            full_pv_div_segm_tf.append(pv_div_segm_tf)
            full_pv_depth_tf.append(pv_depth_tf)
            full_pv_flow_tf.append(pv_flow_tf)
            full_pv_inst_count.append(pv_inst_count)
            full_traject_retval_tf.append(traject_retval_tf)
            full_scene_dp.append([path])
        # if 'timesformer' in self.args.model:
        #     start_frame = 0
        #     end_frame = 30
        # else:
        #     start_frame = 7
        #     end_frame = 7 + self.model.num_frames
        concept_retval = dict()
        # concept_retval['augs_params'] = augs_params  # dict.
        # concept_retval['frame_inds_direct'] = frame_inds_direct  # (Tc).
        # concept_retval['camera_K_tf'] = camera_K_tf  # (Tc, 3, 3) or (1).
        # concept_retval['camera_R_tf'] = camera_R_tf  # (Tc, 4, 4) or (1).
        concept_retval['traject_retval_tf'] = full_traject_retval_tf  # dict; has occl_cont_dag_tf etc.
        concept_retval['pv_rgb_tf'] = torch.stack(full_pv_rgb_tf, dim=0) # (n, 3, Tc, Hf, Wf).
        # concept_retval['pv_depth_tf'] = pv_depth_tf  # (1, Tc, Hf, Wf).
        concept_retval['pv_segm_tf'] = torch.stack(full_pv_segm_tf, dim=0)  # (n, 1, Tc, Hf, Wf).
        concept_retval['pv_depth_tf'] = torch.stack(full_pv_depth_tf, dim=0)  # (n, 1, Tc, Hf, Wf).
        concept_retval['pv_flow_tf'] = torch.stack(full_pv_flow_tf, dim=0)  # (n, 1, Tc, Hf, Wf).
        # concept_retval['pv_coords_tf'] = pv_coords_tf  # (3, Tc, Hf, Wf).
        # concept_retval['pv_xyz_tf'] = pv_xyz_tf  # (3, Tc, Hf, Wf) or (1).
        concept_retval['pv_div_segm_tf'] = torch.stack(full_pv_div_segm_tf, dim=0)  # (n, M, Tc, Hf, Wf) or (1).
        concept_retval['pv_inst_count'] = torch.stack(full_pv_inst_count, dim=0)  # (1).
        concept_retval['full_scene_dp'] = full_scene_dp  # (1).
        concept_retval['seeker_query_mask'] = []
        return concept_retval

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

    def tcow_timesformer_forward(self, vid_idx, keep_all=False):
        # hard coded stuff
        qt_idx = 0
        b = 0
        B = 1
        Qs = 1
        seeker_input = self.dataset['pv_rgb_tf'][vid_idx].unsqueeze(0).cuda()
        all_segm = self.dataset['pv_segm_tf'][vid_idx].unsqueeze(0).cuda()
        all_div_segm = self.dataset['pv_div_segm_tf'][vid_idx].unsqueeze(0).cuda()
        inst_count = self.dataset['pv_inst_count'][vid_idx].unsqueeze(0)
        target_desirability = torch.tensor(self.dataset['traject_retval_tf'][vid_idx]['desirability_tf']).unsqueeze(0)
        occl_fracs = torch.tensor(self.dataset['traject_retval_tf'][vid_idx]['occl_fracs_tf']).unsqueeze(0)
        occl_cont_dag = torch.tensor(self.dataset['traject_retval_tf'][vid_idx]['occl_cont_dag_tf']).unsqueeze(0)
        scene_dp = self.dataset['full_scene_dp'][vid_idx]

        # Sample either random or biased queries.
        sel_query_inds = tcow.utils.my_utils.sample_query_inds(
            B, Qs, inst_count, target_desirability, self.model.train_args, 'cuda', 'test')
        query_idx = sel_query_inds[:, 0]
        seeker_query_mask = torch.zeros_like(all_segm, dtype=torch.uint8)  # (B, 1, T, Hf, Wf).
        seeker_query_mask[b, 0, qt_idx] = (all_segm[b, 0, qt_idx] == query_idx[b] + 1)

        # Prepare query mask and ground truths.
        (seeker_query_mask, snitch_occl_by_ptr, full_occl_cont_id, target_mask,
         target_flags) = tcow.data.data_utils.fill_kubric_query_target_mask_flags(
            all_segm, all_div_segm, query_idx, qt_idx, occl_fracs, occl_cont_dag, scene_dp,
            None, self.model.train_args, 'cuda', 'test')

        del full_occl_cont_id, all_segm, all_div_segm, occl_cont_dag, scene_dp

        # add things to dataset for logging if they were missed, this works with cacheing and non-cached datasets
        if keep_all:
            try:
                self.dataset['seeker_query_mask'].append(seeker_query_mask.cpu())
                self.dataset['target_mask'].append(target_mask.cpu())
            except:
                self.dataset['seeker_query_mask'] = [seeker_query_mask.cpu()]
                self.dataset['target_mask'] = [target_mask.cpu()]


        # forward pass:
        (output_mask, output_flags, features) = self.model(seeker_input, seeker_query_mask)

        # debug - visualize the output of the model
        # t = 13
        # plt.imshow(seeker_input[0][:, t].permute(1, 2, 0).cpu());plt.show()
        # plt.imshow(output_mask[0].sigmoid()[0][t].cpu());plt.show()
        try:
            if self.args.save_prediction:
                # seeker_input -> T H W C
                # output_mask -> B T H W
                # make output mask between 0 and 1
                output_mask = output_mask.sigmoid()
                (Qs, Cmt) = target_mask.shape[:2]
                query_border = self.draw_segm_borders(np.array(seeker_query_mask.detach().cpu()[0, 0][..., None]), fill_white=False)
                snitch_border = self.draw_segm_borders(np.array(target_mask.detach().cpu()[0, 0][..., None]), fill_white=False) if Cmt >= 1 else np.zeros_like(output_mask[0, 0], dtype=np.bool)
                vis_snitch = self.create_model_output_snitch_video(np.array(seeker_input[0].detach().cpu().permute(1,2,3,0)), np.array(output_mask[0].detach().cpu()), query_border, snitch_border,
                                        grayscale=False)
                # save video
                prediction_save_dir = os.path.join(self.args.save_dir, 'predictions')
                if not os.path.exists(prediction_save_dir):
                    os.makedirs(prediction_save_dir, exist_ok=True)
                save_file = os.path.join(prediction_save_dir, f'{vid_idx}.mp4')
                self.save_video(vis_snitch, save_file, fps=6)

                # save frames
                prediction_frame_save_dir = os.path.join(self.args.save_dir, 'prediction_frames')
                if not os.path.exists(prediction_frame_save_dir):
                    os.makedirs(prediction_frame_save_dir, exist_ok=True)
                prediction_frame_video_save_dir = os.path.join(prediction_frame_save_dir, f'{vid_idx}')
                if not os.path.exists(prediction_frame_video_save_dir):
                    os.makedirs(prediction_frame_video_save_dir, exist_ok=True)
                for frame_idx in range(vis_snitch.shape[0]):
                    img = Image.fromarray((vis_snitch[frame_idx] * 255).astype(np.uint8))
                    img.save(os.path.join(prediction_frame_video_save_dir, 'frame_{}.png'.format(frame_idx)))
        except:
            pass


        model_retval = {}
        # all_target_flags.append(target_flags)  # (B, T, 3).
        # target_flags = torch.stack([target_flags], dim=1)  # (B, Qs, T, 3).
        model_retval['target_flags'] = torch.stack([target_flags], dim=1).cuda()  # (B, Qs, T, 3).

        # snitch_occl_by_ptr = torch.stack([snitch_occl_by_ptr], dim=1)  # (B, Qs, 1, T, Hf, Wf).
        model_retval['snitch_occl_by_ptr'] = torch.stack([snitch_occl_by_ptr], dim=1).cuda()

        cur_occl_fracs = occl_fracs[:, query_idx, :, :].diagonal(0, 0, 1)
        cur_occl_fracs = rearrange(cur_occl_fracs, 'T V B -> B T V')  # (B, T, 3).
        sel_occl_fracs = torch.stack([cur_occl_fracs], dim=1)  # (B, Qs, T, 3).
        model_retval['sel_occl_fracs'] = sel_occl_fracs.cuda()  # (B, Qs, T, 3).

        return output_mask, output_flags, target_mask, features, model_retval

    def get_target_mask(self, vid_idx):
        # hard coded stuff
        qt_idx = 0
        b = 0
        B = 1
        Qs = 1
        # seeker_input = self.dataset['pv_rgb_tf'][vid_idx].unsqueeze(0).cuda()
        all_segm = self.dataset['pv_segm_tf'][vid_idx].unsqueeze(0).cuda()
        all_div_segm = self.dataset['pv_div_segm_tf'][vid_idx].unsqueeze(0).cuda()
        inst_count = self.dataset['pv_inst_count'][vid_idx].unsqueeze(0)
        target_desirability = torch.tensor(self.dataset['traject_retval_tf'][vid_idx]['desirability_tf']).unsqueeze(
            0)
        occl_fracs = torch.tensor(self.dataset['traject_retval_tf'][vid_idx]['occl_fracs_tf']).unsqueeze(0)
        occl_cont_dag = torch.tensor(self.dataset['traject_retval_tf'][vid_idx]['occl_cont_dag_tf']).unsqueeze(0)
        scene_dp = self.dataset['full_scene_dp'][vid_idx]

        # Sample either random or biased queries.
        sel_query_inds = tcow.utils.my_utils.sample_query_inds(
            B, Qs, inst_count, target_desirability, None, 'cuda', 'test')
        query_idx = sel_query_inds[:, 0]
        seeker_query_mask = torch.zeros_like(all_segm, dtype=torch.uint8)  # (B, 1, T, Hf, Wf).
        seeker_query_mask[b, 0, qt_idx] = (all_segm[b, 0, qt_idx] == query_idx[b] + 1)

        # Prepare query mask and ground truths.
        (seeker_query_mask, snitch_occl_by_ptr, full_occl_cont_id, target_mask,
         target_flags) = tcow.data.data_utils.fill_kubric_query_target_mask_flags(
            all_segm, all_div_segm, query_idx, qt_idx, occl_fracs, occl_cont_dag, scene_dp,
            None, self.model.train_args, 'cuda', 'test')

        del full_occl_cont_id, all_segm, all_div_segm, occl_cont_dag, scene_dp

        # add things to dataset for logging if they were missed, this works with cacheing and non-cached datasets
        try:
            self.dataset['seeker_query_mask'].append(seeker_query_mask.cpu())
            self.dataset['target_mask'].append(target_mask.cpu())
        except:
            self.dataset['seeker_query_mask'] = [seeker_query_mask.cpu()]
            self.dataset['target_mask'] = [target_mask.cpu()]

    def get_layer_activations(self, num_videos):
        '''
        Get layer activations for the given number of videos
        :param num_videos:
        :return:
        '''

        self.outputs = []
        preds = []
        results = {'mean_snitch_iou': [],
                   'mean_occl_mask_iou': [],
                   'mean_cont_mask_iou': [],
                   }
        with torch.no_grad():
            if 'pre' in self.args.model:
                # initialize mask
                mask = torch.zeros((1, 1568)).cuda().type(torch.bool)

            self.layer_activations = {k: [] for k in self.args.cluster_layer}
            for vid_idx in range(num_videos):
                if 'timesformer' in self.args.model:
                    if 'kubric' in self.args.dataset:
                        output_mask, output_flags, target_mask, features, model_retval = self.tcow_timesformer_forward(vid_idx, keep_all=True)


                        model_retval = {
                            'output_mask': output_mask.unsqueeze(0),
                            'target_mask': target_mask.unsqueeze(0)
                        }
                        metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                  source_name='kubric')
                        # put all values to cpu
                        metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                        metrics_retval = calculate_weighted_averages(metrics_retvals=[metrics_retval])
                        metrics_retval = {k: float(v) for k, v in metrics_retval.items()}
                        for metric in results.keys():
                            if metrics_retval[metric] != -1:
                                results[metric].append(metrics_retval[metric])

                        # debug metrics
                        # self.outputs.append(output_mask.cpu())
                        # output_mask_binary = (output_mask > 0.0).bool()
                        # target_mask_binary = (target_mask > 0.5).bool()  # (B, Q?, 1/3, T, Hf, Wf).
                        #
                        # # snitch iou
                        # snitch_output_mask = output_mask_binary[:, 0, :, :, :]
                        # snitch_target_mask = target_mask_binary[:, 0, :, :, :]
                        # snitch_iou = (snitch_output_mask & snitch_target_mask).sum() / (snitch_output_mask | snitch_target_mask).sum()
                        # results['mean_snitch_iou'].append(snitch_iou.item())
                        #
                        # # occl mask iou
                        # occl_output_mask = output_mask_binary[:, 1, :, :, :]
                        # occl_target_mask = target_mask_binary[:, 1, :, :, :]
                        # occl_iou = (occl_output_mask & occl_target_mask).sum() / (occl_output_mask | occl_target_mask).sum()
                        # results['mean_occl_mask_iou'].append(occl_iou.item())
                        #
                        # # cont mask iou
                        # cont_output_mask = output_mask_binary[:, 2, :, :, :]
                        # cont_target_mask = target_mask_binary[:, 2, :, :, :]
                        # cont_iou = (cont_output_mask & cont_target_mask).sum() / (cont_output_mask | cont_target_mask).sum()
                        # results['mean_cont_mask_iou'].append(cont_iou.item())

                    else:
                        if 'davis' not in self.args.dataset:
                            video = self.post_resize_smooth(self.dataset[vid_idx]).unsqueeze(0).cuda()
                            seeker_query_mask = self.post_resize_nearest(self.seeker_query_labels[vid_idx].squeeze(0)).unsqueeze(0).cuda()
                        else:
                            video = self.dataset[vid_idx].unsqueeze(0).cuda()
                            seeker_query_mask = self.seeker_query_labels[vid_idx].unsqueeze(0).cuda()
                        (output_mask, output_flags, features) = self.model(video, seeker_query_mask)
                        output_mask_binary = (output_mask > 0.0).bool().cpu()
                        self.outputs.append(output_mask_binary)
                elif 'vidmae' in self.args.model:
                    # preprocess video
                    if self.args.dataset == 'kubric':
                        video = self.dataset['pv_rgb_tf'][vid_idx].permute(1,0,2,3)
                        self.get_target_mask(vid_idx)
                        start_frame = int((30-self.model.num_frames)/2)
                        video = video[start_frame:start_frame+self.model.num_frames]
                    else:
                        video = self.dataset[vid_idx].permute(1,0,2,3)
                    video = torch.stack([self.normalize(vid) for vid in video], dim=0).permute(1,0,2,3)
                    video = video.unsqueeze(0).cuda()
                    if 'pre' in self.args.model:
                        _, features = self.model(video, mask)
                    else:
                        _, features = self.model(video)
                elif 'intern' in self.args.model:
                    video = self.dataset[vid_idx]
                    video = self.model.transform(video).unsqueeze(0).cuda()
                    _, features = self.model.encode_video(video)
                    # video_features = self.model.encode_video(video)
                    # debug -> working!
                    # text_cand = [self.args.target_class, "an airplane is flying", "a dog is chasing a ball"]
                    # text = self.model.tokenize(text_cand).cuda()
                    # text_features = self.model.encode_text(text)
                    # video_features = torch.nn.functional.normalize(video_features, dim=1)
                    # text_features = torch.nn.functional.normalize(text_features, dim=1)
                    # t = self.model.logit_scale.exp()
                    # probs = (video_features @ text_features.T * t).softmax(dim=-1).cpu().numpy()
                    #
                    # print("Label probs: ")  # [[9.5619422e-01 4.3805469e-02 2.0393253e-07]]
                    # for t, p in zip(text_cand, probs[0]):
                    #     print("{:30s}: {:.4f}".format(t, p))
                    # print()
                else:
                    raise NotImplementedError

                # features.shape = num_layers x channels x num_heads x time x height x width
                for layer_idx, layer in enumerate(self.args.cluster_layer):
                    self.layer_activations[layer].append(features[layer_idx])

        if len(preds) > 0:
            # print layers
            print('Layers perturbed: ', self.args.cluster_layer)
            # print accuracy of predictions
            print(f'Classification accuracy: {np.mean(np.array(preds) == np.array(self.labels))}')
            exit()

    def get_layer_activations_full_video(self, num_videos):
        if self.args.process_full_video:
            self.chunk_overlap_sizes = []
        self.outputs = []
        preds = []
        results = {'mean_snitch_iou': [],
                   'mean_occl_mask_iou': [],
                   'mean_cont_mask_iou': [],
                   }
        with torch.no_grad():
            if 'pre' in self.args.model:
                # initialize mask
                mask = torch.zeros((1, 1568)).cuda().type(torch.bool)

            self.layer_activations = {k: [] for k in self.args.cluster_layer}
            for vid_idx in range(num_videos):
                if 'timesformer' in self.args.model:
                    if 'kubric' in self.args.dataset:
                        output_mask, output_flags, target_mask, features, model_retval = self.tcow_timesformer_forward(
                            vid_idx, keep_all=True)
                        self.outputs.append(output_mask.cpu())

                        model_retval = {
                            'output_mask': output_mask.unsqueeze(0),
                            'target_mask': target_mask.unsqueeze(0)
                        }
                        metrics_retval = calculate_metrics_mask_track(data_retval=None, model_retval=model_retval,
                                                                      source_name='kubric')
                        # put all values to cpu
                        metrics_retval = {k: v.cpu() for k, v in metrics_retval.items()}
                        metrics_retval = calculate_weighted_averages(metrics_retvals=[metrics_retval])
                        metrics_retval = {k: float(v) for k, v in metrics_retval.items()}
                        for metric in results.keys():
                            if metrics_retval[metric] != -1:
                                results[metric].append(metrics_retval[metric])

                        # shitty metrics
                        # output_mask_binary = (output_mask > 0.0).bool()
                        # target_mask_binary = (target_mask > 0.5).bool()  # (B, Q?, 1/3, T, Hf, Wf).
                        #
                        # # snitch iou
                        # snitch_output_mask = output_mask_binary[:, 0, :, :, :]
                        # snitch_target_mask = target_mask_binary[:, 0, :, :, :]
                        # snitch_iou = (snitch_output_mask & snitch_target_mask).sum() / (snitch_output_mask | snitch_target_mask).sum()
                        # results['mean_snitch_iou'].append(snitch_iou.item())
                        #
                        # # occl mask iou
                        # occl_output_mask = output_mask_binary[:, 1, :, :, :]
                        # occl_target_mask = target_mask_binary[:, 1, :, :, :]
                        # occl_iou = (occl_output_mask & occl_target_mask).sum() / (occl_output_mask | occl_target_mask).sum()
                        # results['mean_occl_mask_iou'].append(occl_iou.item())
                        #
                        # # cont mask iou
                        # cont_output_mask = output_mask_binary[:, 2, :, :, :]
                        # cont_target_mask = target_mask_binary[:, 2, :, :, :]
                        # cont_iou = (cont_output_mask & cont_target_mask).sum() / (cont_output_mask | cont_target_mask).sum()
                        # results['mean_cont_mask_iou'].append(cont_iou.item())

                    else:
                        if 'davis' not in self.args.dataset:
                            video = self.post_resize_smooth(self.dataset[vid_idx]).unsqueeze(0).cuda()
                            seeker_query_mask = self.post_resize_nearest(
                                self.seeker_query_labels[vid_idx].squeeze(0)).unsqueeze(0).cuda()
                        else:
                            video = self.dataset[vid_idx].unsqueeze(0).cuda()

                        if self.args.process_full_video:
                            # divide up frames into self.model.num_frames chunks
                            video_chunks = []
                            for i in range(0, video.shape[2], self.model.num_frames):
                                if i == 0:
                                    video_chunks.append(video[:, :, i:i + self.model.num_frames])
                                else:
                                    # grab the last frame of the previous prediction
                                    video_chunks.append(video[:, :, i - 1:i - 1 + self.model.num_frames])
                            # record remainder
                            self.chunk_overlap_sizes.append(
                                video.shape[2] % self.model.num_frames)  # don't really need to record this in advance...
                            # remove last chunk if it is not the right size
                            if video_chunks[-1].shape[2] != self.model.num_frames:
                                # replace with the last self.model.num_frames frame
                                video_chunks[-1] = video[:,:, -self.model.num_frames:]
                            # process each chunk
                            for video_chunk_idx, video_chunk in enumerate(video_chunks):
                                video_chunk = video_chunk.cuda()
                                if video_chunk_idx == 0:
                                    seeker_query_mask = self.seeker_query_labels[vid_idx].unsqueeze(0).cuda()
                                elif video_chunk_idx == len(video_chunks) - 1:
                                    # for last prediction, need to grab the query 30 frames from the end of the video
                                    seeker_query_mask[:,0,0] = final_predicted_mask[:,0,-(30-self.chunk_overlap_sizes[vid_idx])]
                                else:
                                    # grab the last frame of the previous prediction
                                    seeker_query_mask[:,0] = output_mask_binary[:,0].float()
                                (output_mask, output_flags, features) = self.model(video_chunk, seeker_query_mask)
                                output_mask_binary = (output_mask > 0.0).bool().cpu()
                                # if first chunk, initialize combined feature
                                if video_chunk_idx == 0:
                                    combined_feature = features
                                    final_predicted_mask = output_mask_binary
                                # check if last feature
                                elif video_chunk_idx == len(video_chunks) - 1:
                                    feature_chunk_overlap = int(self.chunk_overlap_sizes[vid_idx])
                                    # slice and concat video to remove overlap
                                    combined_feature = [torch.cat([combined_feature[layer_chunk_idx],features[layer_chunk_idx][:, :, :,:feature_chunk_overlap]], dim=3) for layer_chunk_idx in range(len(features))]
                                    if feature_chunk_overlap == 0:
                                        final_predicted_mask = torch.cat([final_predicted_mask, output_mask_binary], dim=2)
                                    else:
                                        final_predicted_mask = torch.cat([final_predicted_mask, output_mask_binary[:,:,:feature_chunk_overlap]], dim=2)
                                else:
                                    final_predicted_mask = torch.cat([final_predicted_mask, output_mask_binary], dim=2)
                                    # concat along time dimension
                                    for layer_chunk_idx in range(len(features)):
                                        combined_feature[layer_chunk_idx] = torch.cat(
                                            [combined_feature[layer_chunk_idx], features[layer_chunk_idx]], dim=3)
                            features = combined_feature
                        else:
                            seeker_query_mask = self.seeker_query_labels[vid_idx].unsqueeze(0).cuda()
                            (output_mask, output_flags, features) = self.model(video, seeker_query_mask)
                            final_predicted_mask = (output_mask > 0.0).bool().cpu()
                        self.outputs.append(final_predicted_mask)
                elif 'vidmae' in self.args.model:
                    # preprocess video
                    video = self.dataset[vid_idx].permute(1, 0, 2, 3)
                    video = torch.stack([self.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                    if self.args.process_full_video:
                        # divide up frames into self.model.num_frames chunks
                        video_chunks = list(torch.split(video, self.model.num_frames, dim=1))
                        # record remainder
                        self.chunk_overlap_sizes.append(video.shape[1] % self.model.num_frames) # don't really need to record this in advance...
                        # remove last chunk if it is not the right size
                        if video_chunks[-1].shape[1] != self.model.num_frames:
                            # replace with the last self.model.num_frames frames
                            video_chunks[-1] = video[:, -self.model.num_frames:]
                        # process each chunk
                        for video_chunk_idx, video_chunk in enumerate(video_chunks):
                            video_chunk = video_chunk.unsqueeze(0).cuda()
                            if 'pre' in self.args.model:
                                _, features = self.model(video_chunk, mask)
                            else:
                                _, features = self.model(video_chunk)
                            # if first chunk, initialize combined feature
                            if video_chunk_idx == 0:
                                combined_feature = features
                            # check if last feature
                            elif video_chunk_idx == len(video_chunks) - 1:
                                feature_chunk_overlap = int(self.chunk_overlap_sizes[vid_idx]/2)
                                # slice and concat video to remove overlap
                                combined_feature = [torch.cat([combined_feature[layer_chunk_idx], features[layer_chunk_idx][:,:,:,:feature_chunk_overlap]], dim=3) for layer_chunk_idx in range(len(features))]
                            else:
                                # concat along time dimension
                                for layer_chunk_idx in range(len(features)):
                                    combined_feature[layer_chunk_idx] = torch.cat([combined_feature[layer_chunk_idx], features[layer_chunk_idx]], dim=3)
                        features = combined_feature
                    else:
                        video = video.unsqueeze(0).cuda()
                        if 'pre' in self.args.model:
                            _, features = self.model(video, mask)
                        else:
                            _, features = self.model(video)
                            # preds.append(_.argmax().item())
                elif 'svt' in self.args.model:
                    video = self.dataset[vid_idx].permute(1, 0, 2, 3)
                    video = torch.stack([self.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    _, features = self.model(video)
                elif 'mme' in self.args.model:
                    video = self.dataset[vid_idx].permute(1, 0, 2, 3)
                    video = torch.stack([self.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    if 'pre' in self.args.model:
                        _, features = self.model(video, mask)
                    else:
                        _, features = self.model(video)
                elif 'tf_og' in self.args.model:
                    video = self.dataset[vid_idx].permute(1, 0, 2, 3)
                    video = torch.stack([self.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                    video = video.unsqueeze(0).cuda()
                    _, features = self.model(video)
                elif 'intern' in self.args.model:
                    video = self.dataset[vid_idx]
                    video = self.model.transform(video)
                    if self.args.process_full_video:
                        # divide up frames into self.model.num_frames chunks
                        video_chunks = list(torch.split(video, self.model.num_frames, dim=1))
                        # record remainder
                        self.chunk_overlap_sizes.append(
                            video.shape[1] % self.model.num_frames)  # don't really need to record this in advance...
                        # remove last chunk if it is not the right size
                        if video_chunks[-1].shape[1] != self.model.num_frames:
                            # replace with the last self.model.num_frames frames
                            video_chunks[-1] = video[:, -self.model.num_frames:]
                        # process each chunk
                        for video_chunk_idx, video_chunk in enumerate(video_chunks):
                            video_chunk = video_chunk.unsqueeze(0).cuda()
                            _, features = self.model.encode_video(video_chunk)
                            # if first chunk, initialize combined feature
                            if video_chunk_idx == 0:
                                combined_feature = features
                            # check if last feature
                            elif video_chunk_idx == len(video_chunks) - 1:
                                feature_chunk_overlap = int(self.chunk_overlap_sizes[vid_idx])
                                # slice and concat video to remove overlap
                                combined_feature = [torch.cat([combined_feature[layer_chunk_idx],
                                                               features[layer_chunk_idx][:, :, :,
                                                               :feature_chunk_overlap]], dim=3) for layer_chunk_idx in
                                                    range(len(features))]
                            else:
                                # concat along time dimension
                                for layer_chunk_idx in range(len(features)):
                                    combined_feature[layer_chunk_idx] = torch.cat(
                                        [combined_feature[layer_chunk_idx], features[layer_chunk_idx]], dim=3)
                        features = combined_feature
                    else:
                        _, features = self.model.encode_video(video.unsqueeze(0).cuda())
                    # video_features = self.model.encode_video(video)
                    # debug -> working!
                    # text_cand = [self.args.target_class, "an airplane is flying", "a dog is chasing a ball"]
                    # text = self.model.tokenize(text_cand).cuda()
                    # text_features = self.model.encode_text(text)
                    # video_features = torch.nn.functional.normalize(video_features, dim=1)
                    # text_features = torch.nn.functional.normalize(text_features, dim=1)
                    # t = self.model.logit_scale.exp()
                    # probs = (video_features @ text_features.T * t).softmax(dim=-1).cpu().numpy()
                    #
                    # print("Label probs: ")  # [[9.5619422e-01 4.3805469e-02 2.0393253e-07]]
                    # for t, p in zip(text_cand, probs[0]):
                    #     print("{:30s}: {:.4f}".format(t, p))
                    # print()
                elif 'jepa' in self.args.model:
                    # video = self.dataset[vid_idx].permute(1,0,2,3)
                    # video = torch.stack([self.normalize(vid) for vid in video], dim=0).permute(1,0,2,3)
                    # video = video.unsqueeze(0).cuda()
                    # _, features = self.model(video)

                    # preprocess video
                    video = self.dataset[vid_idx].permute(1, 0, 2, 3)
                    video = torch.stack([self.normalize(vid) for vid in video], dim=0).permute(1, 0, 2, 3)
                    if self.args.process_full_video:
                        # divide up frames into self.model.num_frames chunks
                        video_chunks = list(torch.split(video, self.model.num_frames, dim=1))
                        # record remainder
                        self.chunk_overlap_sizes.append(
                            video.shape[1] % self.model.num_frames)  # don't really need to record this in advance...
                        # remove last chunk if it is not the right size
                        if video_chunks[-1].shape[1] != self.model.num_frames:
                            # replace with the last self.model.num_frames frames
                            video_chunks[-1] = video[:, -self.model.num_frames:]
                        # process each chunk
                        for video_chunk_idx, video_chunk in enumerate(video_chunks):
                            video_chunk = video_chunk.unsqueeze(0).cuda()
                            _, features = self.model(video_chunk)
                            # if first chunk, initialize combined feature
                            if video_chunk_idx == 0:
                                combined_feature = features
                            # check if last feature
                            elif video_chunk_idx == len(video_chunks) - 1:
                                feature_chunk_overlap = int(self.chunk_overlap_sizes[vid_idx] / 2)
                                # slice and concat video to remove overlap
                                combined_feature = [torch.cat([combined_feature[layer_chunk_idx],
                                                               features[layer_chunk_idx][:, :, :,
                                                               :feature_chunk_overlap]], dim=3) for layer_chunk_idx in
                                                    range(len(features))]
                            else:
                                # concat along time dimension
                                for layer_chunk_idx in range(len(features)):
                                    combined_feature[layer_chunk_idx] = torch.cat(
                                        [combined_feature[layer_chunk_idx], features[layer_chunk_idx]], dim=3)
                        features = combined_feature
                    else:
                        video = video.unsqueeze(0).cuda()
                        _, features = self.model(video)

                else:
                    raise NotImplementedError

                for layer_idx, layer in enumerate(self.args.cluster_layer):
                    self.layer_activations[layer].append(features[layer_idx])  # b x c x t x h x w

        # from evaluation.vtcd_vos_eval import compute_iou
        # miou = []
        # for video_idx in range(num_videos):
        #     out_mask = self.outputs[video_idx][0,0]
        #     label = self.labels[video_idx]
        #     # compute iou
        #     iou = compute_iou(out_mask, label)
        #     miou.append(iou)
        # print('Mean iou: ', np.mean(miou))

        if len(preds) > 0:
            # print layers
            print('Layers perturbed: ', self.args.cluster_layer)
            # print accuracy of predictions
            print(f'Classification accuracy: {np.mean(np.array(preds) == np.array(self.labels))}')
            exit()

    def intra_video_clustering(self):
        """
        Clusters the features of all videos into tubelets.
        """


        if self.args.dataset == 'kubric':
            num_videos = self.dataset['pv_rgb_tf'].shape[0]
        else:
            num_videos = len(self.dataset)

        # get video features
        if self.args.process_full_video:
            self.get_layer_activations_full_video(num_videos)
        else:
            self.get_layer_activations(num_videos)

        if not self.args.intra_inter_cluster and 'attn' in self.args.cluster_subject:
            print('Are you sure you want to cluster the attention maps across videos??')

        # cluster features
        self.num_intra_clusters = {k: {head: [] for head in self.args.attn_head} for k in self.args.cluster_layer}
        self.segment_dataset = {k: {head: [] for head in self.args.attn_head} for k in self.args.cluster_layer}
        self.segment_dataset_pooled = {k: {head: [] for head in self.args.attn_head} for k in self.args.cluster_layer}
        self.segment_dataset_mask = {k: {head: [] for head in self.args.attn_head} for k in self.args.cluster_layer}
        self.segment_id_assign = {k: {head: [] for head in self.args.attn_head} for k in self.args.cluster_layer} # note: these are the same for each layer

        for layer_idx, (layer, features) in enumerate(self.layer_activations.items()):
            if isinstance(self.args.intra_elbow_threshold, list):
                try:
                    elbow_value = float(self.args.intra_elbow_threshold[layer_idx])
                except:
                    elbow_value = float(self.args.intra_elbow_threshold[0])
            else:
                elbow_value = self.args.intra_elbow_threshold
            if self.args.save_intr_concept_videos_all_k:
                print('saving intra concept video for layer {}'.format(layer))
            no_mask_cluster = 0
            total_num_clusters = 0
            print('Clustering layer {}'.format(layer))
            for vid_idx in tqdm.tqdm(range(num_videos)):
                # check memory usage
                if psutil.virtual_memory().percent > 90:
                    print('Memory usage too high, clearing memory...')
                    gc.collect()
                    torch.cuda.empty_cache()
                    exit()
                for head_idx, head in enumerate(self.args.attn_head):
                    feature = features[vid_idx][:,:,head]
                    B, C, T, H, W = feature.shape
                    with threadpool_limits(limits=self.args.max_num_workers, user_api='openmp'):
                        if self.args.save_intr_concept_videos_all_k:
                            _, num_clusters, _ = cluster_features(feature,
                                                               elbow='dino',
                                                               max_num_clusters=10,
                                                               elbow_threshold=elbow_value,
                                                               layer=layer, verbose=False)
                            for k in range(2, 10):
                                cost, _, _ = cluster_features(feature,
                                                          elbow='faiss_k',
                                                          max_num_clusters=k,
                                                          elbow_threshold=elbow_value,
                                                          layer=layer, verbose=False)
                                cost = cost.squeeze(0)
                                # save video
                                if 'vidmae' in self.args.model:
                                    cost_resized = torch.nn.functional.interpolate(cost.unsqueeze(0), size=(
                                        self.model.args.num_frames, self.frame_width, self.frame_height),
                                                                                   mode='trilinear').squeeze(0)
                                else:
                                    cost_resized = self.post_resize_smooth(cost)


                                self.save_intra_tubelets(video_idx=vid_idx,cost=cost_resized,layer=layer,head=head)

                        else:
                            if self.args.intra_cluster_method == 'slic' or self.args.intra_cluster_method == 'random':
                                if not self.args.process_full_video:
                                    feature = self.pre_upsample(feature)
                                else:
                                    # define a function to upsample for the entire video length
                                    feature = torch.nn.Upsample(size=(self.dataset[vid_idx].shape[1], int(self.dataset[vid_idx].shape[2]*self.args.spatial_resize_factor), int(self.dataset[vid_idx].shape[3]*self.args.spatial_resize_factor)))(feature)
                            elif self.args.intra_cluster_method == 'crop':
                                feature = self.post_resize_smooth(feature[0]).unsqueeze(0)
                            all_out, num_clusters, centroids = cluster_features(feature,
                                                    elbow=self.args.intra_cluster_method,
                                                    max_num_clusters=self.args.intra_max_cluster,
                                                    elbow_threshold=elbow_value,
                                                    layer=layer, verbose=False,
                                                    sample_interval=self.args.sample_interval,
                                                    n_segments=self.args.n_segments,
                                                    slic_compactness=self.args.slic_compactness,
                                                    spacing=self.args.slic_spacing)
                    num_total_segments = 0
                    for resolution_idx in range(len(all_out)):
                        curr_out = all_out[resolution_idx]
                        curr_out = curr_out.squeeze(0) if (self.args.intra_cluster_method not in ['slic', 'crop', 'random'])  else curr_out
                        self.num_intra_clusters[layer][head].append(num_clusters)
                        if 'vidmae' in self.args.model:
                            try:
                                curr_out_resized = torch.nn.functional.interpolate(curr_out.unsqueeze(0), size=(self.model.num_frames, self.frame_width, self.frame_height), mode='trilinear').squeeze(0)
                            except:
                                if self.args.intra_cluster_method == 'crop':
                                    curr_out_resized = torch.nn.Upsample(size=(self.dataset.shape[2], self.dataset.shape[3], self.dataset.shape[4]))(curr_out.type(torch.uint8).unsqueeze(0)).squeeze(0)
                                elif self.args.dataset == 'kubric':
                                    curr_out_resized = torch.nn.Upsample(size=(self.dataset['pv_rgb_tf'].shape[2], self.dataset['pv_rgb_tf'].shape[3], self.dataset['pv_rgb_tf'].shape[4]))(curr_out.type(torch.uint8).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                                else:
                                    if not self.args.process_full_video:
                                        curr_out_resized = torch.nn.Upsample(size=(self.dataset.shape[2], self.dataset.shape[3], self.dataset.shape[4]))(curr_out.type(torch.uint8).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                                    else:
                                        curr_out_resized = torch.nn.Upsample(size=(self.dataset[vid_idx].shape[1], int(
                                            self.dataset[vid_idx].shape[2]), int(
                                            self.dataset[vid_idx].shape[3])))(curr_out.type(torch.uint8).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

                        else:
                            if self.args.intra_cluster_method == 'slic' or self.args.intra_cluster_method == 'random':
                                if not self.args.process_full_video:
                                    curr_out_resized = self.post_upsample(curr_out.unsqueeze(0).unsqueeze(0).type(torch.uint8)).squeeze()
                                else:
                                    # define a function to upsample for the entire video length
                                    curr_out_resized = torch.nn.Upsample(size=(self.dataset[vid_idx].shape[1], int(
                                        self.dataset[vid_idx].shape[2]), int(
                                        self.dataset[vid_idx].shape[3])))(
                                        curr_out.type(torch.uint8).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                            else:
                                curr_out_resized = self.post_resize_smooth(curr_out)


                        if self.args.save_intra_segments:
                            self.save_intra_tubelets(video_idx=vid_idx,cost=curr_out_resized,layer=layer,head=head,dir_name='intra_concepts')
                        if self.args.save_intra_indv_segments:
                            self.save_intra_tubelets_indv(video_idx=vid_idx,cost=curr_out_resized,layer=layer,head=head,dir_name='indv_intra_concepts')
                        if self.args.intra_cluster_method not in ['slic', 'crop', 'random']:
                            if self.args.intra_cluster_method == 'cnmf':
                                assign = curr_out.argmax(0)
                                assign_resized = curr_out_resized.argmax(0)
                            else:
                                assign = curr_out.argmin(0)
                                assign_resized = curr_out_resized.argmin(0)
                            clusters_range = range(curr_out.shape[0])
                        else:
                            assign = curr_out
                            assign_resized = curr_out_resized
                            clusters_range = curr_out.unique()
                        for cluster_idx in clusters_range:
                            num_total_segments += 1
                            total_num_clusters += 1
                            if 'attn' in self.args.cluster_subject and not 'mae' in self.args.model:
                                self.save_attn_centroids(centroids, layer, vid_idx)

                            masked_feature = torch.where(assign == cluster_idx, feature, 0).squeeze(0)
                            mask_small = torch.where(assign == cluster_idx, 1, 0).squeeze(0)
                            mask = torch.where(assign_resized == cluster_idx, 1, 0).squeeze(0)

                            # would need to count number of non-zero elements in the mask, sum the masked feature, then divide by the number of non-zero elements

                            # don't add mask if less than 0.1% of the mask is non-zero
                            if mask_small.sum() < 0.001 * mask_small.numel():
                                no_mask_cluster += 1
                                continue


                            if self.args.pool_non_zero_features:
                                segment_vector = (masked_feature.sum((1, 2, 3)) / mask_small.sum()).squeeze()
                            else:
                                segment_vector = masked_feature.mean((1,2,3)).squeeze()
                            self.segment_dataset_pooled[layer][head].append(segment_vector)
                            self.segment_id_assign[layer][head].append(vid_idx)

                            # self.segment_dataset_mask[layer][head].append(mask)
                            mask_dir = os.path.join(self.args.vcd_dir, 'masks', f'layer_{layer}', f'head_{head}', f'video_{vid_idx}')

                            if not os.path.exists(mask_dir):
                                os.makedirs(mask_dir)
                            # save mask
                            mask_path = os.path.join(mask_dir, f'segment_{num_total_segments}.npy')
                            # np.save(mask_path, mask.numpy())
                            np.save(mask_path, mask.numpy().astype(bool))
                            self.segment_dataset_mask[layer][head].append(mask_path)

        # stack the features (might not work without pooling)
        for layer in self.args.cluster_layer:
            for head in self.args.attn_head:
                self.segment_dataset_pooled[layer][head] = torch.stack(self.segment_dataset_pooled[layer][head], dim=0)
                self.segment_id_assign[layer][head] = np.stack(self.segment_id_assign[layer][head])

        # remove self.layer_activations
        delattr(self, 'layer_activations')

    def inter_video_clustering(self):
        """
        Clusters the features of multiple videos into different concept clusters.
        """
        # get video features
        self.dic = {layer: {} for layer in self.args.cluster_layer}
        for layer_idx, layer in enumerate(self.args.cluster_layer):
            for head_idx, head in enumerate(self.args.attn_head):
                bn_dic = {}
                if not self.args.intra_inter_cluster:
                    concept_number, bn_dic['concepts'] = 0, []
                    labels = np.array(self.labels[layer])
                    centers = self.centroids[layer]
                    for i in range(labels.max() + 1):
                        label_idxs = np.where(labels == i)[0]
                        if label_idxs.shape[0] == 0:
                            continue
                        spread = len(set(self.segment_id_assign[layer][label_idxs])) / len(label_idxs)
                        concept_number += 1
                        concept = 'concept_{}'.format(concept_number)
                        bn_dic['concepts'].append(concept)
                        bn_dic[concept] = {
                            'video_mask': self.segment_dataset_mask[layer][label_idxs],
                            # binary mask of videos in this cluster
                            'video_numbers': self.segment_id_assign[layer][label_idxs],
                            # original video numbers that clusters belong to
                            'spread': spread,
                            'center': centers[i],
                        }
                        layer_sillhouette_score = 0
                else:
                    if isinstance(self.args.inter_elbow_threshold, list):
                        try:
                            elbow_value = float(self.args.inter_elbow_threshold[layer_idx])
                        except:
                            elbow_value = float(self.args.inter_elbow_threshold[0])
                    else:
                        elbow_value = self.args.inter_elbow_threshold
                    # get dataset of features and assignment
                    features = self.segment_dataset_pooled[layer][head]

                    # cluster features across dataset of videos
                    with threadpool_limits(limits=self.args.max_num_workers, user_api='openmp'):
                        labels, cost, centers, mdl_weights = cluster_dataset(features,
                                                                elbow=self.args.inter_cluster_method,
                                                                max_num_clusters=self.args.inter_max_cluster,
                                                                elbow_threshold=elbow_value,
                                                                layer=layer,
                                                                verbose=False)


                    concept_number, bn_dic['concepts'] = 0, []
                    for i in range(labels.max() + 1):
                        label_idxs = np.where(labels == i)[0]
                        concept_costs = cost[label_idxs]
                        concept_idxs = label_idxs[np.argsort(concept_costs)]

                        # metrics of clusters
                        spread = len(set(self.segment_id_assign[layer][head][concept_idxs])) / len(label_idxs) # fraction of videos that have this concept

                        # can filter by a  minimum size, but we want to see all clusters
                        concept_number += 1
                        concept = 'concept_{}'.format(concept_number)
                        bn_dic['concepts'].append(concept)


                        bn_dic[concept] = {
                            'video_mask': [self.segment_dataset_mask[layer][head][x] for x in concept_idxs], # binary mask of videos in this cluster
                            'video_numbers': self.segment_id_assign[layer][head][concept_idxs], # original video numbers that clusters belong to
                            'spread': spread,
                            'center': centers[i],
                        }
                    if self.args.inter_cluster_method == 'dino': # need to normalize if we dino-ed
                        features = np.array(features)
                    layer_sillhouette_score = silhouette_score(features,labels)  # how well separated the clusters are
                bn_dic['silhouette'] = layer_sillhouette_score
                if not mdl_weights is None:
                    cnmf = {
                        'W': mdl_weights[0],
                        'G': mdl_weights[1],
                        'H': mdl_weights[2],
                    }
                else:
                    cnmf = None
                bn_dic['cnmf'] = cnmf
                self.dic[layer][head] = bn_dic

    def save_intra_tubelets_indv(self, video_idx, cost, layer, head, dir_name='intra_concepts_all_k',extra_frames=10):

        extensions = ['.mp4']

        concept_dir = os.path.join(self.args.save_dir, dir_name)

        if not os.path.exists(concept_dir):
            os.makedirs(concept_dir, exist_ok=True)
        # create concept directory
        layer_dir = os.path.join(concept_dir, 'layer_{}'.format(layer))
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir, exist_ok=True)

        if self.args.dataset == 'kubric':
            rgb_video = self.dataset['pv_rgb_tf'][video_idx]
        else:
            rgb_video = self.dataset[video_idx]
        rgb_video = rgb_video.permute(1, 2, 3, 0)
        # argmax cost to get cluster assignment


        if self.args.intra_cluster_method == 'cnmf':
            assign = cost.argmax(0)
        elif self.args.intra_cluster_method in ['slic', 'crop', 'random']:
            assign = cost
        else:
            assign = cost.argmin(0)
        num_clusters = np.unique(assign) if self.args.intra_cluster_method in ['slic', 'crop', 'random'] else range(cost.shape[0])

        # place full labelled video at top left
        if self.args.dataset == 'kubric':
            target_mask = self.dataset['target_mask'][video_idx].detach().cpu().numpy()[0, 0]
            target_border = self.draw_segm_borders(target_mask[..., None], fill_white=False)
            vis_concept_assign = self.create_model_input_video(np.array(rgb_video), target_mask, target_border, extra_frames=extra_frames,target=True)
            # need to account for new break point



        videos_to_save = [vis_concept_assign]
        for cluster_idx in num_clusters:
            # draw concept mask
            mask = torch.where(assign == cluster_idx, 1, 0).squeeze(0)
            mask = np.array(np.repeat(mask.unsqueeze(0), 3, axis=0).permute(1, 2, 3, 0))  # repeat along channels
            vis_concept_assign = self.create_concept_mask_video(rgb_video, mask, alpha=0.7)

            # draw query mask
            if self.args.dataset == 'kubric':
                # draw query mask
                seeker_query_mask = self.dataset['seeker_query_mask'][video_idx].detach().cpu().numpy()[0,0]
                query_border = self.draw_segm_borders(seeker_query_mask[..., None], fill_white=False)
                vis_concept_assign = self.create_model_input_video(vis_concept_assign, seeker_query_mask, query_border, extra_frames=extra_frames)
            # insert video into canvas
            videos_to_save.append(vis_concept_assign)

        for seg_id, video in enumerate(videos_to_save):
            layer_head_dir = os.path.join(layer_dir, 'Vid{}'.format(video_idx))
            file_name = os.path.join(layer_head_dir, 'Seg{}'.format(seg_id))
            # canvas: (T, H, W, C)
            self.save_video(frames=video,
                                file_name=file_name,
                                extensions=extensions, fps=6,
                                upscale_factor=1)

    def save_intra_tubelets(self, video_idx, cost, layer, head, dir_name='intra_concepts_all_k',extra_frames=0, draw_mask_border=True):

        extensions = ['.mp4']

        # if layer == 0:
        #     return
        # if layer == 11:
        #     return

        concept_dir = os.path.join(self.args.save_dir, dir_name)
        if not os.path.exists(concept_dir):
            os.makedirs(concept_dir, exist_ok=True)


        concept_frame_dir = os.path.join(self.args.save_dir, dir_name + '_frames')
        if not os.path.exists(concept_frame_dir):
            os.makedirs(concept_frame_dir, exist_ok=True)

        concept_single_dir = os.path.join(self.args.save_dir, dir_name + '_single')
        if not os.path.exists(concept_single_dir):
            os.makedirs(concept_single_dir, exist_ok=True)

        # create concept directory
        layer_dir = os.path.join(concept_dir, 'layer_{}'.format(layer))
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir, exist_ok=True)

        layer_frame_dir = os.path.join(concept_frame_dir, 'layer_{}'.format(layer))
        if not os.path.exists(layer_frame_dir):
            os.makedirs(layer_frame_dir, exist_ok=True)

        layer_single_dir = os.path.join(concept_single_dir, 'layer_{}'.format(layer))
        if not os.path.exists(layer_single_dir):
            os.makedirs(layer_single_dir, exist_ok=True)

        head_frame_dir = os.path.join(layer_frame_dir, 'head_{}'.format(head))
        if not os.path.exists(head_frame_dir):
            os.makedirs(head_frame_dir, exist_ok=True)
        video_head_frame_dir = os.path.join(head_frame_dir, 'Vid{}'.format(video_idx))
        if not os.path.exists(video_head_frame_dir):
            os.makedirs(video_head_frame_dir, exist_ok=True)

        head_single_dir = os.path.join(layer_single_dir, 'head_{}'.format(head))
        if not os.path.exists(head_single_dir):
            os.makedirs(head_single_dir, exist_ok=True)

        if self.args.dataset == 'kubric':
            rgb_video = self.dataset['pv_rgb_tf'][video_idx]
            if 'timesformer' in self.args.model:
                canvas = np.zeros((rgb_video.shape[1]+extra_frames, rgb_video.shape[2] * 4, rgb_video.shape[3] * 4, 3))
            else:
                canvas = np.zeros((rgb_video.shape[1], rgb_video.shape[2] * 4, rgb_video.shape[3] * 4, 3))
        else:
            rgb_video = self.dataset[video_idx]
            canvas = np.zeros((rgb_video.shape[1], rgb_video.shape[2] * 4, rgb_video.shape[3] * 4, 3))
        rgb_video = rgb_video.permute(1, 2, 3, 0)
        # argmax cost to get cluster assignment

        if self.args.intra_cluster_method == 'cnmf':
            assign = cost.argmax(0)
        elif self.args.intra_cluster_method in ['slic', 'crop', 'random']:
            assign = cost
        else:
            assign = cost.argmin(0)
        num_clusters = np.unique(assign) if self.args.intra_cluster_method in ['slic', 'crop', 'random'] else range(cost.shape[0])

        # place full labelled video at top left
        if self.args.dataset == 'kubric' and 'timesformer' in self.args.model:
            target_mask = self.dataset['target_mask'][video_idx].detach().cpu().numpy()[0, 0]
            target_border = self.draw_segm_borders(target_mask[..., None], fill_white=False)
            vis_concept_assign = self.create_model_input_video(np.array(rgb_video), target_mask, target_border, extra_frames=extra_frames,target=True)
            canvas[:vis_concept_assign.shape[0], :vis_concept_assign.shape[1], :vis_concept_assign.shape[2]] = vis_concept_assign

            # need to account for new break point
            break_idx = 15
        else:
            break_idx = 16

        for cluster_idx in num_clusters:
            # stop if we have 9 videos
            if cluster_idx < break_idx:
                # draw concept mask
                single_mask = torch.where(assign == cluster_idx, 1, 0).squeeze(0)
                mask = np.array(np.repeat(single_mask.unsqueeze(0), 3, axis=0).permute(1, 2, 3, 0))  # repeat along channels
                vis_concept_assign = self.create_concept_mask_video(rgb_video, mask, alpha=0.5, blend_white=True)

                # draw query mask
                if self.args.dataset == 'kubric' and 'timesformer' in self.args.model:
                    # draw query mask
                    seeker_query_mask = self.dataset['seeker_query_mask'][video_idx].detach().cpu().numpy()[0,0]
                    # don't draw border for query mask
                    # query_border = self.draw_segm_borders(seeker_query_mask[..., None], fill_white=False)
                    # vis_concept_assign = self.create_model_input_video(vis_concept_assign, seeker_query_mask, query_border, extra_frames=extra_frames)
                    if draw_mask_border:
                        concept_mask = np.array(single_mask.float())
                        mask_border = self.draw_segm_borders(concept_mask[..., None], fill_white=False)
                        vis_concept_assign = self.create_model_input_video(vis_concept_assign, concept_mask, mask_border,extra_frames=extra_frames, target=True, color='orange')

                    row = (cluster_idx+1) // 4
                    col = (cluster_idx+1) % 4
                else:
                    row = cluster_idx // 4
                    col = cluster_idx % 4

                # insert video into canvas
                canvas[:, row * rgb_video.shape[1]:(row + 1) * rgb_video.shape[1],col * rgb_video.shape[2]:(col + 1) * rgb_video.shape[2],:] = vis_concept_assign

            # save frames
            for frame_idx in range(vis_concept_assign.shape[0]):
                img = Image.fromarray((vis_concept_assign[frame_idx] * 255).astype(np.uint8))
                img.save(os.path.join(video_head_frame_dir, 'cluster{}_frame{}.png'.format(cluster_idx,frame_idx)))

            single_file_path = os.path.join(head_single_dir, 'cluster{}'.format(cluster_idx))
            self.save_video(frames=vis_concept_assign,
                            file_name=single_file_path,
                            extensions=extensions, fps=6,
                            upscale_factor=1)
        # resize vide
        layer_head_dir = os.path.join(layer_dir, 'head_{}'.format(head))
        file_name = os.path.join(layer_head_dir, 'Vid{}_K{}'.format(video_idx, len(num_clusters) if self.args.intra_cluster_method not in ['slic', 'crop', 'random'] else cost.shape[0]))



        self.save_video(frames=canvas,
                            file_name=file_name,
                            extensions=extensions, fps=6,
                            upscale_factor=1)

    def save_concepts(self, args, extra_frames=0, mode='random', collapse_masks=False, size=3, mask_video_with_concepts=True, draw_mask_border=True):

        extensions = ['.mp4']

        # concept_heatmap_dir = os.path.join(self.args.save_dir, 'concepts', 'heatmaps')
        concept_dir = os.path.join(self.args.save_dir, 'concepts')
        single_vid_concept_dir = os.path.join(self.args.save_dir, 'single_vid_concepts')
        concept_frames_dir = os.path.join(self.args.save_dir, 'concept_frames')
        # if not os.path.exists(concept_heatmap_dir):
        #     os.makedirs(concept_heatmap_dir, exist_ok=True)
        if not os.path.exists(concept_dir):
            os.makedirs(concept_dir, exist_ok=True)

        if not os.path.exists(single_vid_concept_dir):
            os.makedirs(single_vid_concept_dir, exist_ok=True)

        if not os.path.exists(concept_frames_dir):
            os.makedirs(concept_frames_dir, exist_ok=True)

        for layer in reversed(self.args.cluster_layer):
            print('Saving concepts for layer {}'.format(layer))
            for head in self.args.attn_head:

                # create concept directory
                layer_head_dir = os.path.join(concept_dir, 'layer_{}/head_{}'.format(layer, head))
                if not os.path.exists(layer_head_dir):
                    os.makedirs(layer_head_dir, exist_ok=True)

                # for each concept, save the video in order of closest to center
                for concept in self.dic[layer][head]['concepts']:
                    # concept_layer_dir = os.path.join(layer_dir, concept)
                    # if not os.path.exists(concept_layer_dir):
                    #     os.makedirs(concept_layer_dir, exist_ok=True)
                    # target_mask = self.dataset['target_mask']
                    # grab rgb videos and masks in concept

                    if len(args.single_save_layer_head) > 0:
                        layer = args.single_save_layer_head[0]
                        head = args.single_save_layer_head[1]
                        concept = 'concept_{}'.format(args.single_save_layer_head[2])
                        print('Saving single concept: layer {}, head {}, concept {}'.format(layer, head, concept))

                        single_post_resize_nearest = torchvision.transforms.Resize(
                            (224, 224),
                            interpolation=torchvision.transforms.InterpolationMode.NEAREST)

                        single_vid_layer_concept_head_dir = os.path.join(single_vid_concept_dir, 'layer_{}/head_{}'.format(layer, head))
                        if not os.path.exists(single_vid_layer_concept_head_dir):
                            os.makedirs(single_vid_layer_concept_head_dir, exist_ok=True)

                    if self.args.dataset == 'kubric':
                        concept_rgb_videos = self.dataset['pv_rgb_tf'][self.dic[layer][head][concept]['video_numbers']]
                        # concept_query_masks = torch.stack(self.dataset['seeker_query_mask'])[self.dic[layer][head][concept]['video_numbers']]
                        if 'timesformer' in self.args.model:
                            target_masks = torch.stack(self.dataset['target_mask'])[self.dic[layer][head][concept]['video_numbers']]
                            canvas = np.zeros((concept_rgb_videos.shape[2] + extra_frames, concept_rgb_videos.shape[3] * size, concept_rgb_videos.shape[4] * size, 3))
                        else:
                            canvas = np.zeros((concept_rgb_videos.shape[2], concept_rgb_videos.shape[3] * size,concept_rgb_videos.shape[4] * size, 3))
                    else:
                        if not self.args.process_full_video:
                            concept_rgb_videos = self.dataset[self.dic[layer][head][concept]['video_numbers']]
                            canvas = np.zeros((concept_rgb_videos.shape[2], concept_rgb_videos.shape[3] * size,
                                               concept_rgb_videos.shape[4] * size, 3))
                        else:
                            video_ids = self.dic[layer][head][concept]['video_numbers']
                            concept_rgb_videos = [self.dataset[vid_id] for vid_id in video_ids]
                            # get maximum number of frames
                            max_frames = max([vid.shape[1] for vid in concept_rgb_videos])
                            canvas = np.zeros((max_frames, concept_rgb_videos[0].shape[2] * size, concept_rgb_videos[0].shape[3] * size, 3))

                    if mode == 'closest' and 'cnmf' in self.args.inter_cluster_method:
                        concept_masks = self.dic[layer][head][concept]['video_mask']
                        concept_idx = (int(concept.split('_')[-1])-1)
                        G = self.dic[layer][head]['cnmf']['G']
                        assign_matrix = G[G.argmax(1) == concept_idx, concept_idx]
                        # H = self.dic[layer][head]['cnmf']['H']
                        # assign_matrix = H[concept_idx, H.argmax(0) == concept_idx]
                        vid_nums = np.argsort(assign_matrix)[::-1]
                    else:
                        # vid_nums = range(len(self.dic[layer][head][concept]['video_numbers']))
                        # join masks if we have more than one from the same video
                        if collapse_masks:
                            concept_masks = []
                            for vid_id in np.unique(self.dic[layer][head][concept]['video_numbers']):
                                vid_paths = [path for i, path in enumerate(self.dic[layer][head][concept]['video_mask']) if self.dic[layer][head][concept]['video_numbers'][i] == vid_id]
                                concept_masks.append(vid_paths)
                            vid_nums = range(len(concept_masks))
                        else:
                            vid_nums = range(len(self.dic[layer][head][concept]['video_numbers']))
                            concept_masks = self.dic[layer][head][concept]['video_mask']


                    # [6 18  4 34 40 31 43 10 11 38 29 15 30 37 41 26  7 19  0  1  8 32 25  3, 28 33 23 22 27  5 17 39 16 20 13  9 44  2 35 24 42 21 14 12 36]
                    # save each video in concept
                    for num_segment, i in enumerate(vid_nums):

                        # replace self.dic[layer][head][concept]['video_numbers'] with a list of video numbers based on 1) distance from center 2) random_order

                        if not len(args.single_save_layer_head) > 0:
                            if num_segment == size**2:
                                break
                        rgb_video = concept_rgb_videos[i].permute(1,2,3,0)
                        mask_path = concept_masks[i]
                        # load mask from path
                        if isinstance(mask_path, list):
                            # combine all masks into one
                            mask = torch.tensor(np.stack([np.load(path) for path in mask_path], axis=0).sum(axis=0))
                        else:
                            mask = torch.tensor(np.load(mask_path))
                        try:
                            concept_mask = self.post_resize_nearest(mask)
                        except:
                            concept_mask= self.post_resize_nearest(mask.unsqueeze(0))

                        mask = np.array(np.repeat(concept_mask.unsqueeze(0), 3, axis=0).permute(1, 2, 3, 0))  # repeat along channels
                        if mask_video_with_concepts:
                            vis_concept_assign = self.create_concept_mask_video(rgb_video, mask, alpha=0.5, blend_white=True)
                        else:
                            vis_concept_assign = rgb_video

                        if self.args.dataset == 'kubric' and 'timesformer' in self.args.model:
                            # draw query mask
                            # new
                            target_mask = target_masks[i].detach().cpu().numpy()[0, 0]
                            target_border = self.draw_segm_borders(target_mask[..., None], fill_white=False)
                            vis_concept_assign = self.create_model_input_video(vis_concept_assign, target_mask, target_border,extra_frames=extra_frames, target=True)

                            # old
                            # seeker_query_mask = concept_query_masks[i].detach().cpu().numpy()[0, 0]
                            # query_border = self.draw_segm_borders(seeker_query_mask[..., None], fill_white=False)
                            # vis_concept_assign = self.create_model_input_video(vis_concept_assign, seeker_query_mask, query_border,
                            #                                                    extra_frames=extra_frames)
                        if draw_mask_border:
                            concept_mask = np.array(concept_mask.float())
                            #  :param segm (T, H, W, K) array of uint8.
                            # :return rgb_vis (T, H, W, 3) array of float32.
                            mask_border = self.draw_segm_borders(concept_mask[..., None], fill_white=False)
                            vis_concept_assign = self.create_model_input_video(vis_concept_assign, concept_mask,
                                                                               mask_border, extra_frames=extra_frames,
                                                                               target=True, color='orange')

                        if len(args.single_save_layer_head) > 0:
                            # save rgb video
                            rgb_video_file_name = os.path.join(single_vid_layer_concept_head_dir, 'rgb_video_{}_{}'.format(num_segment, concept))

                            if 'timesformer' in self.args.model and self.args.dataset == 'ssv2':
                                rgb_video = single_post_resize_nearest(rgb_video.permute(0,3,1,2)).permute(0,2,3,1)
                            self.save_video(frames=np.array(rgb_video),
                                            file_name=rgb_video_file_name,
                                            extensions=extensions, fps=6,
                                            upscale_factor=1)

                            single_vid_file_name = os.path.join(single_vid_layer_concept_head_dir, f'layer{layer}_head{head}_{concept}_tube_{num_segment}')
                            # save video
                            if 'timesformer' in self.args.model and self.args.dataset == 'ssv2':
                                # grab every other frame
                                vis_concept_assign = vis_concept_assign[::2]
                                # repeat last frame
                                vis_concept_assign = np.concatenate([vis_concept_assign, vis_concept_assign[-1:]], axis=0)
                                vis_concept_assign = single_post_resize_nearest(torch.tensor(vis_concept_assign).permute(0, 3, 1, 2)).permute(0,2,3,1)
                            self.save_video(frames=vis_concept_assign,
                                            file_name=single_vid_file_name,
                                            extensions=extensions, fps=6,
                                            upscale_factor=1)
                        if args.save_concepts:
                            # insert video into canvas
                            row = num_segment // size
                            col = num_segment % size

                            # temporally pad vis_concept_assign to longest video
                            if vis_concept_assign.shape[0] < canvas.shape[0]:
                                vis_concept_assign = np.concatenate([vis_concept_assign, np.repeat(vis_concept_assign[-1:], canvas.shape[0] - vis_concept_assign.shape[0], axis=0)], axis=0)
                            canvas[:, row * rgb_video.shape[1]:(row + 1) * rgb_video.shape[1],col * rgb_video.shape[2]:(col + 1) * rgb_video.shape[2],:] = vis_concept_assign

                    if args.save_concepts:
                        file_name = os.path.join(layer_head_dir, '{}'.format(concept))
                        self.save_video(frames=canvas,
                                            file_name=file_name,
                                            extensions=extensions, fps=6,
                                            upscale_factor=1)
                    if len(args.single_save_layer_head) > 0:
                        return

    def create_concept_mask_video(self, input_video, concept_mask, alpha=0.6, blend_white=False):
        '''
        :param input_video (T, H, W, 3) array of float32 in [0, 1].
        :param concept_mask (T, H, W, 3) array of int in {0, 1}.
        :return video (T, H, W, 3) array of float32 in [0, 1].
        '''

        if blend_white:
            # Blend towards white for non-masked regions
            white_background = np.ones_like(input_video)
            video = np.where(concept_mask == 1, input_video, input_video * (1 - alpha) + white_background * alpha)
        else:
            # Darken the non-masked regions
            video = np.where(concept_mask == 1, input_video, input_video * alpha)
        return video

    def create_heatmap_video(self, input_video, concept_mask, alpha=0.6):
        '''
        :param input_video (T, H, W, 3) array of float32 in [0, 1].
        :param concept_mask (T, H, W) array of float32 in [0, 1].
        :return video (T, H, W, 3) array of float32 in [0, 1].
        '''

        # overlay mask with input
        # video = (alpha * input_video)  + ((1-alpha)*concept_mask)
        video = input_video * concept_mask
        video = np.clip(video, 0.0, 1.0)
        return video

    def draw_segm_borders(self, segm, fill_white=False):
        '''
        :param segm (T, H, W, K) array of uint8.
        :return rgb_vis (T, H, W, 3) array of float32.
        '''
        assert segm.ndim == 4

        border_mask = np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, :-2, 1:-1, :]) + \
                      np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, 2:, 1:-1, :]) + \
                      np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, 1:-1, :-2, :]) + \
                      np.abs(segm[:, 1:-1, 1:-1, :] - segm[:, 1:-1, 2:, :])
        border_mask = np.any(border_mask, axis=-1)
        # (T, Hf - 2, Wf - 2) bytes in {0, 1}.
        border_mask = np.pad(border_mask, ((0, 0), (1, 1), (1, 1)), mode='constant')
        # (T, Hf, Wf) bytes in {0, 1}.

        if fill_white:
            border_mask = np.repeat(border_mask[..., None], repeats=3, axis=-1)
            # (T, Hf, Wf, 3) bytes in {0, 1}.
            result = border_mask.astype(np.float32)

        else:
            result = border_mask

        return result

    def create_model_input_video(self, seeker_rgb, seeker_query_mask, query_border, extra_frames=10, target=False, color='green'):
        '''
        :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
        :param seeker_query_mask (T, H, W) array of float32 in [0, 1].
        :param query_border (T, H, W, 3) array of float32 in [0, 1].
        :return video (T, H, W, 3) array of float32 in [0, 1].
        '''
        if target:
            query_time = 0
        else:
            query_time = seeker_query_mask.any(axis=(1, 2)).argmax()
        # vis = seeker_rgb + seeker_query_mask[..., None]  # (T, H, W, 3).
        vis = seeker_rgb + query_border[..., None]  # (T, H, W, 3).

        if extra_frames > 0:
            vis[query_time] *= 0.6
        if color == 'green':
            vis[query_border, 0] = 0.0 # red
            vis[query_border, 1] = 1.0  # green
            vis[query_border, 2] = 0.0 # blue
        elif color == 'red':
            vis[query_border, 0] = 1.0 # red
            vis[query_border, 1] = 0.0  # green
            vis[query_border, 2] = 0.0 # blue
        elif color == 'blue':
            vis[query_border, 0] = 0.0 # red
            vis[query_border, 1] = 0.0  # green
            vis[query_border, 2] = 1.0 # blue
        elif color == 'pink':
            vis[query_border, 0] = 1.0 # red
            vis[query_border, 1] = 0.5  # green
            vis[query_border, 2] = 1.0 # blue
        elif color == 'orange':
            vis[query_border, 0] = 1.0 # red
            vis[query_border, 1] = 0.75  # green
            vis[query_border, 2] = 0.5 # blue
        elif color == 'dark_purple':
            vis[query_border, 0] = 0.4 # red
            vis[query_border, 1] = 0.0  # green
            vis[query_border, 2] = 0.6 # blue

        # Pause for a bit at query time to make the instance + mask very clear.
        if extra_frames > 0:
            vis = np.concatenate([vis[0:query_time]] + [vis[query_time:query_time + 1]] * (extra_frames+1) + [vis[query_time + 1:]], axis=0)

        video = np.clip(vis, 0.0, 1.0)
        return video

    def save_video(self, frames, file_name=None, fps=6,
                   extensions=None, upscale_factor=1, defer_log=False):
        '''
        inputs: frames: list of frames to save as video (T, H, W, 3) array of float32 in [0, 1].
        Records a single set of frames as a video to a file in visuals and/or the online dashboard.
        '''
        # Ensure before everything else that buffer exists.



        # Duplicate last frame three times for better visibility.
        # last_frame = frames[len(frames) - 1:len(frames)]
        # frames = np.concatenate([frames, last_frame, last_frame, last_frame], axis=0)

        if frames.dtype in [np.float32, np.float64]:
            frames = (frames * 255.0).astype(np.uint8)

        if upscale_factor > 1:
            frames = [cv2.resize(
                frame,
                (frame.shape[1] * upscale_factor, frame.shape[0] * upscale_factor),
                interpolation=cv2.INTER_NEAREST) for frame in frames]

        for_online_fp = None
        if file_name is not None:

            if extensions is None:
                extensions = ['']

            for ext in extensions:
                used_file_name = file_name + ext
                dst_fp = used_file_name
                # dst_fp = os.path.join(self.vis_dir, used_file_name)

                parent_dp = str(pathlib.Path(dst_fp).parent)
                if not os.path.exists(parent_dp):
                    os.makedirs(parent_dp, exist_ok=True)

                if dst_fp.lower().endswith('.mp4'):
                    imageio.mimwrite(dst_fp, frames, codec='libx264', format='ffmpeg', fps=fps, macro_block_size=None, quality=8)
                elif dst_fp.lower().endswith('.webm'):
                    # https://programtalk.com/python-more-examples/imageio.imread/?ipage=13
                    imageio.mimwrite(dst_fp, frames, fps=fps, codec='libvpx', format='ffmpeg',
                                     ffmpeg_params=["-b:v", "0", "-crf", "14"])
                else:
                    imageio.mimwrite(dst_fp, frames, fps=fps)
                if dst_fp.lower().endswith('.gif') or dst_fp.lower().endswith('.webm'):
                    for_online_fp = dst_fp

    def create_model_output_snitch_video(self, seeker_rgb, output_mask, query_border, snitch_border, grayscale=False):
        '''
        :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
        :param output_mask (1/3, T, H, W) array of float32 in [0, 1].
        :param query_border (T, H, W) array of float32 in [0, 1].
        :param snitch_border (T, H, W) array of float32 in [0, 1].
        :param grayscale (bool): Whether to convert the output to grayscale.
        :return video (T, H, W, 3) array of float32 in [0, 1].
        '''
        if grayscale:
            seeker_rgb = seeker_rgb.copy()
            seeker_gray = seeker_rgb[..., 0] * 0.2 + seeker_rgb[..., 1] * 0.6 + seeker_rgb[..., 2] * 0.2
            seeker_rgb[..., 0] = seeker_gray
            seeker_rgb[..., 1] = seeker_gray
            seeker_rgb[..., 2] = seeker_gray

        snitch_heatmap = plt.cm.magma(output_mask[0])[..., 0:3]
        # snitch_heatmap = plt.cm.summer(output_mask[0])[..., 0:3]
        vis = seeker_rgb * 0.6 + snitch_heatmap * 0.5  # (T, H, W, 3).

        # vis[snitch_border] = 0.0
        # vis[snitch_border, 1] = 1.0

        vis[query_border] = 0.0
        vis[query_border, 0] = 1.0
        vis[query_border, 1] = 1.0
        vis[query_border, 2] = 1.0

        video = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).
        return video
