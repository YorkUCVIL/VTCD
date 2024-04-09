'''
Data-related utils and helper methods.
Created by Basile Van Hoorick, Jun 2022.
'''

from models.hide_seek.tcow.__init__ import *

# Library imports.
import glob

# Internal imports.
import models.hide_seek.tcow.utils.geometry as geometry
import models.hide_seek.tcow.utils.my_utils as my_utils


def read_video_audio_clip(video_fp, audio_fp, sel_num_frames=16, sel_start_time=None,
                          read_audio=True):
    # https://pytorch.org/vision/main/auto_examples/plot_video_api.html
    # https://pytorch.org/vision/stable/io.html#video
    # NOTE: this doesn't work with pixel phone recorded videos.
    av_reader = torchvision.io.VideoReader(video_fp, 'video')
    metadata = copy.deepcopy(av_reader.get_metadata())

    video_duration = metadata['video']['duration'][0]
    video_fps = metadata['video']['fps'][0]
    assert video_duration > 0
    assert video_fps > 0

    # Set clip boundaries.
    available_frames = round(video_duration * video_fps)
    assert available_frames >= 4
    if sel_num_frames <= 0:
        sel_num_frames = available_frames + 1  # This ensures counter never breaks.
    if sel_start_time == 'random':
        # Max end value leaves a small margin to ensure audio synchronization is always possible.
        start_frame = np.random.randint(0, available_frames - sel_num_frames - 2)
    else:
        # Round to ensure the requested timestamp (in seconds) is followed accurately.
        start_frame = int(np.round(sel_start_time * video_fps))
    start_time = start_frame / video_fps
    end_frame = start_frame + sel_num_frames
    end_time = end_frame / video_fps

    # Read clip video frames always.
    av_reader.set_current_stream('video')
    av_reader.seek(start_time)
    frames = []
    video_pts = []
    counter = 0

    for frame in av_reader:
        frames.append(frame['data'])  # (C, H, W) tensor.
        video_pts.append(frame['pts'])  # single float.
        counter += 1
        if counter >= sel_num_frames:
            break

    frames = torch.stack(frames, 0)  # (T, C, H, W) tensor.
    video_real_start = video_pts[0]
    video_real_end = video_pts[-1] + 1.0 / video_fps

    # NOTE: For these requested values to be correct, we assume video_pts is equally spaced with
    # interval = 1 / FPS. Other cases are probably rare and difficult to deal with.
    metadata['vid_avail_frames'] = available_frames
    metadata['vid_duration'] = video_duration
    metadata['vid_fps'] = video_fps
    metadata['vid_start_frame'] = start_frame
    metadata['vid_start_time'] = start_time
    metadata['vid_end_frame'] = end_frame
    metadata['vid_end_time'] = end_time
    metadata['vid_real_start'] = video_real_start
    metadata['vid_real_end'] = video_real_end

    # Read clip audio waveform if desired.
    if read_audio:
        if audio_fp is not None:
            av_reader = torchvision.io.VideoReader(audio_fp, 'audio')
            audio_metadata = copy.deepcopy(av_reader.get_metadata())
            metadata['audio'] = audio_metadata['audio']

        audio_duration = metadata['audio']['duration'][0]
        audio_sample_rate = metadata['audio']['framerate'][0]
        assert audio_duration > 0
        assert audio_sample_rate > 0
        assert audio_duration >= video_duration

        # NOTE: We assume audio packets are always shorter than video frames, and that PTS
        # (presentation timestamp) denotes when to _start_ presenting a packet.
        # https://en.wikipedia.org/wiki/Presentation_timestamp
        av_reader.set_current_stream('audio')
        av_reader.seek(video_real_start - 1.0 / video_fps)

        wavelets = []
        audio_pts = []
        counter = 0

        for frame in av_reader:
            wavelets.append(frame['data'])  # (T, C) tensor with WXYZ if C = 4.
            audio_pts.append(frame['pts'])  # single float.
            counter += 1
            if audio_pts[-1] > video_real_end:
                break

        waveform = torch.cat(wavelets, 0)  # (T, C) tensor with WXYZ if C = 4.
        audio_real_start = audio_pts[0]
        audio_real_end = audio_pts[-1] + wavelets[-1].shape[0] / audio_sample_rate

        # NOTE: video_pts and audio_pts have different temporal strides / intervals, i.e.
        # (1 / video_fps) and (wavelet_size / sample_rate) respectively. We sampled audio packets in
        # such a way that it encompasses the video clip, so now we simply concatenate and mark the
        # appropriate subset to take.
        num_samples = waveform.shape[0]
        align_start = (video_real_start - audio_real_start) / (audio_real_end - audio_real_start)
        align_end = (video_real_end - audio_real_start) / (audio_real_end - audio_real_start)
        align_start_idx = int(align_start * num_samples)
        align_end_idx = int(align_end * num_samples)

        waveform = waveform[align_start_idx:align_end_idx]
        num_samples_per_frame = audio_sample_rate / video_fps
        nspf_align = waveform.shape[0] / frames.shape[0]
        np.testing.assert_approx_equal(nspf_align, num_samples_per_frame, significant=3)

        metadata['aud_duration'] = audio_duration
        metadata['aud_sample_rate'] = audio_sample_rate
        metadata['aud_read_samples'] = num_samples
        metadata['aud_read_real_start'] = audio_real_start
        metadata['aud_read_real_end'] = audio_real_end
        metadata['aud_align_start_idx'] = align_start_idx
        metadata['aud_align_end_idx'] = align_end_idx
        metadata['aud_samples_per_frame'] = num_samples_per_frame

    else:
        waveform = None
        audio_pts = None

    del av_reader
    del metadata['video']
    del metadata['audio']

    return (frames, waveform, video_pts, audio_pts, metadata)


def read_all_images(src_dp, exclude_patterns=None, count_only=False, use_tqdm=False, stack=False,
                    early_resize_height=None):
    '''
    :param src_dp (str).
    :return frames (T, H, W, 3) array with float32 in [0, 1].
    '''
    src_fps = list(sorted(glob.glob(os.path.join(src_dp, '*.jpg')) +
                          glob.glob(os.path.join(src_dp, '*.png'))))

    if count_only:
        return len(src_fps)

    if exclude_patterns is not None:
        if not(isinstance(exclude_patterns, list)):
            exclude_patterns = [exclude_patterns]
        for pattern in exclude_patterns:
            src_fps = [fp for fp in src_fps if not(pattern in fp)]

    frames = []
    if use_tqdm:
        src_fps = tqdm.tqdm(src_fps)
    
    for fp in src_fps:
        frame = plt.imread(fp)[..., 0:3]
        frame = (frame / 255.0).astype(np.float32)
        
        if early_resize_height is not None and early_resize_height > 0:
            (H1, W1) = frame.shape[:2]
            if H1 > early_resize_height:
                (H2, W2) = (early_resize_height, int(round(early_resize_height * W1 / H1)))
                frame = cv2.resize(frame, (W2, H2), interpolation=cv2.INTER_LINEAR)
        
        frames.append(frame)

    if stack:  # Otherwise, remain list for efficiency.
        frames = np.stack(frames)
    
    return frames


def cached_listdir(dir_path, allow_exts=[], recursive=False, force_live=False):
    '''
    Returns a list of all file paths if needed, and caches the result for efficiency.
    NOTE: Manual deletion of listdir.p is required if the directory ever changes.
    :param dir_path (str): Folder to gather file paths within.
    :param allow_exts (list of str): Only retain files matching these extensions.
    :param recursive (bool): Also include contents of all subdirectories within.
    :param force_live (bool): Same behavior but don't read or write any cache.
    :return (list of str): List of full image file paths.
    '''
    exts_str = '_'.join(allow_exts)
    recursive_str = 'rec' if recursive else ''
    cache_fp = f'{str(pathlib.Path(dir_path))}_{exts_str}_{recursive_str}_cld.p'

    if os.path.exists(cache_fp) and not(force_live):
        # Cached result already available.
        print('Loading directory contents from ' + cache_fp + '...')
        with open(cache_fp, 'rb') as f:
            result = pickle.load(f)

    else:
        # No cached result available yet. This call can sometimes be very expensive.
        raw_listdir = os.listdir(dir_path)
        result = copy.deepcopy(raw_listdir)

        # Append root directory to get full paths.
        result = [os.path.join(dir_path, fn) for fn in result]

        # Filter by files only (no folders), not being own cache dump,
        # belonging to allowed file extensions, and not starting with a dot.
        result = [fp for fp in result if os.path.isfile(fp)]
        result = [fp for fp in result if not fp.endswith('_cld.p')]
        if allow_exts is not None and len(allow_exts) != 0:
            result = [fp for fp in result
                      if any([fp.lower().endswith('.' + ext) for ext in allow_exts])
                      and fp[0] != '.']

        # Recursively append contents of subdirectories within.
        if recursive:
            for dn in raw_listdir:
                dp = os.path.join(dir_path, dn)
                if os.path.isdir(dp):
                    result += cached_listdir(dp, allow_exts=allow_exts, recursive=True)

        if not(force_live):
            print('Caching filtered directory contents to ' + cache_fp + '...')
            with open(cache_fp, 'wb') as f:
                pickle.dump(result, f)

    return result


def get_thing_collision_pointers(metadata, frame_inds):
    '''
    :param metadata (dict).
    :param frame_inds (list of int).
    :return coll_data (K, T) array of int32.
    '''
    K = metadata['scene']['num_valo_instances']
    T = len(frame_inds)
    timed_collisions = [(*c['instances'], c['frame']) for c in metadata['collisions']]

    # NOTE: Raw time values are float because of the 240 Hz simulation. We first check for invalid
    # collisions, then convert time values to frame values, filter by relevant time range, and
    # finally round to nearest int when using.
    frame_start = frame_inds[0]
    frame_stride = frame_inds[1] - frame_inds[0]
    sel_collisions = [(tc[0], tc[1], tc[2] / frame_stride - frame_start)
                      for tc in timed_collisions if tc[0] >= -1 and tc[1] >= -1]
    sel_collisions = [(tc[0], tc[1], tc[2])
                      for tc in sel_collisions if tc[2] > -0.5 and tc[2] < T - 0.5]
    sel_collisions = np.round(np.array(sel_collisions)).astype(np.int32)

    # Per frame: -2 = no collision, -1 = dome (typically floor), >= 0 = instance ID.
    # NOTE: Some IDs may be >= K, i.e. when the collision targets are never visible in the video.
    coll_data = np.ones((K, T), dtype=np.int32) * (-2)

    # Loop over all instances and frames to retrieve collision pairs.
    # NOTE: Sometimes, one entry in this discrete grid may actually match multiple collisions,
    # in which case we restort to a simple random choice among the other candidate IDs while giving
    # priority to non-dome objects.
    for k in range(-1, K):
        for f in range(T):

            rows1 = np.where(np.logical_and(
                sel_collisions[:, 0] == k, sel_collisions[:, 2] == f))[0]  # I am column index 0.
            rows2 = np.where(np.logical_and(
                sel_collisions[:, 1] == k, sel_collisions[:, 2] == f))[0]  # I am column index 1.

            if len(rows1) > 0 or len(rows2) > 0:
                other_ids = list(sel_collisions[rows1][..., 1]) + \
                    list(sel_collisions[rows2][..., 0])

                # Sanity check: you cannot collide with yourself.
                assert not(np.any(other_ids == k))

                nondome_ids = [id for id in other_ids if id != -1]
                if len(nondome_ids) != 0:
                    chosen_id = np.random.choice(nondome_ids)
                else:
                    chosen_id = -1

                coll_data[k, f] = chosen_id

                # print(k, f, len(rows1), len(rows2), chosen_id)
                # print(other_ids)
                # print()

            # Otherwise: this entry remains -2, indicating lack of collision activity for this
            # particular (instance, frame) pair.

    # There is a lot of noise due to many quick successive collisions, so we explicitly filter these
    # repetitions out.
    for k in range(K):
        for f in range(T - 1, 0, -1):
            if coll_data[k, f] == coll_data[k, f - 1]:
                coll_data[k, f] = -2

    return coll_data


def get_thing_occl_fracs_numpy(pv_segm, pv_div_segm):
    '''
    NOTE: This method amortizes recursive occlusion because it only considers visible pixels versus
        *all* pixels if no other object were to exist.
    :param pv_segm (T, Hf, Wf, 1) array of int32 in [0, inf): 1-based instance IDs (0 = background).
    :param pv_div_segm (T, Hf, Wf, K) array of uint8 in [0, 1].
    :return occl_fracs (K, T, 3) array of float32 with (f, v, t) in [0, 1].
    '''
    (T, Hf, Wf, K) = pv_div_segm.shape

    # Per object and per frame: 3 values (f, v, t) where:
    # f = occlusion fraction = v / t.
    # v = number of visible instance pixels / image size.
    # t = number of total instance pixels / image size.
    occl_fracs = np.zeros((K, T, 3), dtype=np.float32)

    for k in range(K):
        for f in range(T):

            num_visible_pxl = np.sum(pv_segm[f, ..., 0] == k + 1)
            num_total_pxl = np.sum(pv_div_segm[f, ..., k] == 1)
            # if num_visible_pxl > num_total_pxl:
            #     print('x')
            # assert num_visible_pxl <= num_total_pxl, \
            #     f'get_thing_occl_fracs_numpy: num_visible_pxl: {num_visible_pxl} > num_total_pxl: {num_total_pxl}'

            if num_total_pxl > 0:
                occl_frac = 1.0 - num_visible_pxl / num_total_pxl
            else:
                occl_frac = 0.0  # Probably out of frame; occlusions are irrelevant in that case.

            occl_fracs[k, f, 0] = occl_frac
            occl_fracs[k, f, 1] = num_visible_pxl / (Hf * Wf)
            occl_fracs[k, f, 2] = num_total_pxl / (Hf * Wf)

    return occl_fracs


def get_thing_occl_fracs_torch(pv_segm, pv_div_segm):
    '''
    NOTE: This method amortizes recursive occlusion because it only considers visible pixels versus
        *all* pixels if no other object were to exist.
    :param pv_segm (1, T, Hf, Wf) tensor of int32 in [0, inf): 1-based instance IDs (0 = background).
    :param pv_div_segm (K, T, Hf, Wf) tensor of uint8 in [0, 1].
    :return occl_fracs (K, T, 3) array of float32 with (f, v, t) in [0, 1].
    '''
    (K, T, Hf, Wf) = pv_div_segm.shape

    # Per object and per frame: 3 values (f, v, t).
    occl_fracs = np.zeros((K, T, 3), dtype=np.float32)

    for k in range(K):
        for f in range(T):

            num_visible_pxl = torch.sum(pv_segm[0, f] == k + 1)
            num_total_pxl = torch.sum(pv_div_segm[k, f] == 1)
            # if num_visible_pxl > num_total_pxl:
            #     print('x')
            # assert num_visible_pxl <= num_total_pxl, \
            #     f'get_thing_occl_fracs_torch: num_visible_pxl: {num_visible_pxl} > num_total_pxl: {num_total_pxl}'

            if num_total_pxl > 0:
                occl_frac = 1.0 - num_visible_pxl / num_total_pxl
            else:
                occl_frac = 0.0  # Probably out of frame; occlusions are irrelevant in that case.

            occl_fracs[k, f, 0] = occl_frac
            occl_fracs[k, f, 1] = num_visible_pxl / (Hf * Wf)
            occl_fracs[k, f, 2] = num_total_pxl / (Hf * Wf)

    return occl_fracs


def get_thing_occl_cont_dag(pv_segm, pv_div_segm, metadata, frame_inds):
    '''
    :param pv_segm (T, Hf, Wf, 1) array of int32 in [0, K]: 1-based instance IDs (0 = background).
    :param pv_div_segm (T, Hf, Wf, K) array of uint8 in [0, 1].
    :param metadata (dict).
    :param frame_inds (list of int).
    :return (occl_cont_dag, rel_order, recon_pv_segm, recon_error).
        occl_cont_dag (T, K, K, 3) array of float32 with (c, od, of) in [0, 1]: Hard containment
            flags and soft occlusion pointers per object pair per frame. Index ordering follows
            (containee, container) or (occludee, occluder).
        rel_order (T, K) array of int32 in [0, K): Instance indices sorted by depth from back
            to front per frame.
        recon_pv_segm (T, Hf, Wf, 1) array of int32 in [0, K].
        recon_error (float).
    '''
    (T, Hf, Wf, K) = pv_div_segm.shape

    recon_pv_segm = np.zeros((T, Hf, Wf, 1), dtype=np.int32)
    # Based on depth ordering with pv_div_segm; may sometimes deviate from pv_segm since we only use
    # 3D object center positions as a proxy for their spatial relations.

    # Pre-calculate some stuff for efficiency.
    div_segm_pxl_cnt = np.sum(pv_div_segm, axis=(1, 2))  # (T, K).
    vis_segm_pxl_cnt = np.zeros_like(div_segm_pxl_cnt)  # (T, K).
    for f in range(T):
        for k in range(K):
            vis_segm_pxl_cnt[f, k] = np.sum(pv_segm[f, ..., 0] == k + 1)

    oc_dag = np.zeros((T, K, K, 3), dtype=np.float32)
    rel_order = np.zeros((T, K), dtype=np.int32)

    for f, t in enumerate(frame_inds):
        cam_3d_pos = np.array(metadata['camera']['positions'][t])[None, :]
        # (1, 3).
        obj_3d_pos = np.array([metadata['instances'][k]['positions'][t] for k in range(K)])
        # (K, 3).
        distances = np.linalg.norm(cam_3d_pos - obj_3d_pos, ord=2, axis=-1)
        # (K).
        cur_order = np.argsort(distances)[::-1]  # Large to small = far to close = back to front.
        # (K).
        rel_order[f] = cur_order

        # Outer loop over all instances from back to front.
        for order_idx, ref_inst_id in enumerate(cur_order):
            before_ids = cur_order[order_idx + 1:]
            other_ids = np.concatenate([cur_order[:order_idx], before_ids])

            # We update the reconstructed visible segmentation map with this object ID, overwriting
            # all pixels where it exists. NOTE: == 1 is very important here, otherwise it may be
            # interpreted as an indexing / scattering operation instead of a boolean mask.
            recon_pv_segm[f, ..., 0][pv_div_segm[f, ..., ref_inst_id] == 1] = ref_inst_id + 1

            ref_pxl = div_segm_pxl_cnt[f, ref_inst_id]

            # Inner loop over all other instances that are not the reference one (they may be
            # potential containers); the iteration order does not matter here.
            for cand_inst_id in other_ids:

                # c = containment fraction; *LOWER BOUND* of how much volume of ref is inside cand.
                ref_box = np.array(metadata['instances'][ref_inst_id]['bboxes_3d'][t])
                cand_box = np.array(metadata['instances'][cand_inst_id]['bboxes_3d'][t])
                cur_c = geometry.get_containment_fraction_approx(ref_box, cand_box)
                oc_dag[f, ref_inst_id, cand_inst_id, 0] = cur_c

            # Inner loop over all instances in front of the current reference (they may be potential
            # direct occluders); also from back (nearest) to front (furthest) for recon_pv_segm.
            for cand_inst_id in before_ids:

                # od = direct occlusion fraction (considers only this pair of objects in isolation;
                # act as if nothing else exists).
                # NOTE: This is unreliable when the two objects are close to each other, since the
                # presumed direction of this edge is determined strictly by before_ids, which may be
                # wrong about the depth ordering!
                cand_pxl = div_segm_pxl_cnt[f, cand_inst_id]
                overlap_pxl = np.sum(np.logical_and(pv_div_segm[f, ..., ref_inst_id] == 1,
                                                    pv_div_segm[f, ..., cand_inst_id] == 1))
                cur_od_ptr = overlap_pxl / max(ref_pxl, 1)
                oc_dag[f, ref_inst_id, cand_inst_id, 1] = cur_od_ptr

                # or = responsible occlusion fraction (considers the not yet previously covered
                # reference pixels that this candidate now contributes to).
                # NOTE: Ignored for now.
                pass
            
            # Inner loop over all other instances that are not the reference one (they may be
            # potential frontmost occluders); the iteration order does not matter here.
            for cand_inst_id in other_ids:

                # of = final / frontmost occlusion fraction (considers only the visible pixels of
                # the candidate occluding object).
                # NOTE: of <= od at all times, and = iff candidate is fully visible.
                cand_pxl = div_segm_pxl_cnt[f, cand_inst_id]
                cand_vis_pxl = vis_segm_pxl_cnt[f, cand_inst_id]
                overlap_pxl = np.sum(np.logical_and(pv_div_segm[f, ..., ref_inst_id] == 1,
                                                    pv_segm[f, ..., 0] == cand_inst_id + 1))
                cur_of_ptr = overlap_pxl / max(ref_pxl, 1)
                oc_dag[f, ref_inst_id, cand_inst_id, 2] = cur_of_ptr
                
                # Sanity checks.
                assert cand_vis_pxl <= cand_pxl * 1.003, \
                    f'cand_vis_pxl: {cand_vis_pxl} > cand_pxl: {cand_pxl}'
                if cand_inst_id in before_ids:
                    cur_od_ptr = oc_dag[f, ref_inst_id, cand_inst_id, 1]
                    assert cur_of_ptr <= cur_od_ptr * 1.003, \
                        f'cur_of_ptr: {cur_of_ptr} > cur_od_ptr: {cur_od_ptr}'

    assert np.all(np.diagonal(oc_dag, 0, 1, 2) == 0.0), \
        'Objects cannot occlude or contain themselves.'

    recon_error = np.mean(pv_segm != recon_pv_segm)
    # print('recon_error:', recon_error)  # DEBUG

    return (oc_dag, rel_order, recon_pv_segm, recon_error)


# def get_thing_div_segm_pointers(pv_segm, pv_div_segm):
#     '''
#     Combines visible and xray segmentation maps to get per-pixel instance ID pointers.
#     pv_div_ptrs (T, Hf, Wf, K) array of uint8 in [0, K): Like pv_div_segm, but values are frontmost
#         occluding object IDs.
#     '''
#     pv_div_ptrs = np.zeros_like(pv_div_segm, dtype=np.uint8)


def pad_div_numpy(div_array, axes, max_size):
    '''
    Adds zeros to the first axis of an array such that it can be collated.
    :param div_array (K, *) array.
    :param axes (tuple) of int.
    :param max_size (int) = M.
    :return (padded_div_array, K).
        padded_div_array (M, *) array.
        K (int).
    '''
    # if axis == 0:
    #     cur_K = div_array.shape[0]
    #     to_append = np.zeros_like(div_array[0:1])
    #     to_append = np.broadcast_to(
    #         to_append, (max_size - cur_K, *to_append.shape[1:]))
    #     div_array = np.concatenate([div_array, to_append], axis=0)

    # elif axis == 1:
    #     cur_K = div_array.shape[1]
    #     to_append = np.zeros_like(div_array[:, 0:1])
    #     to_append = np.broadcast_to(
    #         to_append, (*to_append.shape[0:1], max_size - cur_K, *to_append.shape[2:]))
    #     div_array = np.concatenate([div_array, to_append], axis=1)

    # elif axis == 2:
    #     cur_K = div_array.shape[2]
    #     to_append = np.zeros_like(div_array[:, :, 0:1])
    #     to_append = np.broadcast_to(
    #         to_append, (*to_append.shape[0:2], max_size - cur_K, *to_append.shape[3:]))
    #     div_array = np.concatenate([div_array, to_append], axis=2)

    # elif axis == 3:
    #     cur_K = div_array.shape[3]
    #     to_append = np.zeros_like(div_array[:, :, :, 0:1])
    #     to_append = np.broadcast_to(
    #         to_append, (*to_append.shape[0:3], max_size - cur_K, *to_append.shape[4:]))
    #     div_array = np.concatenate([div_array, to_append], axis=3)

    # else:
    #     raise ValueError(axes)

    K = -1
    pad_width = [(0, 0) for _ in range(div_array.ndim)]

    for axis in axes:
        cur_K = div_array.shape[axis]
        if K == -1:
            K = cur_K
        else:
            assert cur_K == K

        pad_width[axis] = (0, max_size - K)

    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    padded_div_array = np.pad(div_array, pad_width, mode='constant', constant_values=0)

    return (padded_div_array, K)


def pad_div_torch(div_tensor, axes, max_size):
    '''
    Adds zeros to the first axis of a tensor such that it can be collated.
    :param div_tensor (K, *) tensor.
    :param axes (tuple) of int.
    :param max_size (int) = M.
    :return (padded_div_tensor, K).
        padded_div_tensor (M, *) tensor.
        K (int).
    '''
    K = -1
    pad_width = [(0, 0) for _ in range(div_tensor.ndim)]

    for axis in axes:
        cur_K = div_tensor.shape[axis]
        if K == -1:
            K = cur_K
        else:
            assert cur_K == K

        pad_width[axis] = (0, max_size - K)

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    pad_width = list(np.array(list(reversed(pad_width))).flatten())
    padded_div_tensor = torch.nn.functional.pad(div_tensor, pad_width, mode='constant', value=0)

    return (padded_div_tensor, K)


# def select_unpad_div_torch(padded_div_tensor, axes, sizes, index):
#     '''
#     Uncollates tensor to pick a specific example, and removes extra zeros.
#     :param padded_div_tensor (B, M, *) tensor.
#     :param axes (tuple) of int, excluding batch dimension.
#     :param sizes (B) tensor of int32 in [0, M] with value K at index.
#     :return (div_tensor, K).
#         div_tensor (K, *) tensor.
#         K (int).
#     '''
#     K = sizes[index].item()
#     div_tensor = padded_div_tensor[index]
#     for axis in axes:
#         div_tensor = div_tensor.select(axis, slice(0, K))
#     return (div_tensor, K)


def get_usage_modes(available_input_inds, available_query_inds, available_target_inds,
                    num_frames, query_time, min_target_frames_covered=2):
    '''
    Determines all the available ways in which a video can be subsampled into a clip, given
        information about its total duration, FPS, and frames where annotations exist.
    :param available_input_inds (list).
    :param available_query_inds (list).
    :param available_target_inds (list).
    :param num_frames (int).
    :param query_time (int).
    :return valid_modes (list) of tuples (frame_start, frame_stride, target_coverage).
        frame_start (int).
        frame_stride (int).
        target_coverage (float) in [0, 1].
    '''
    available_input_inds = sorted(list(set(available_input_inds)))
    available_query_inds = sorted(list(set(available_query_inds)))
    available_target_inds = sorted(list(set(available_target_inds)))
    valid_modes = []

    for query_idx in available_query_inds:
        for frame_stride in range(1, 11):

            frame_first = query_idx - query_time * frame_stride  # Inclusive.
            frame_last = frame_first + (num_frames - 1) * frame_stride  # Inclusive.
            if frame_first < 0 or frame_last > max(available_input_inds):
                continue

            target_frames_covered = 0
            for frame_idx in range(frame_first, frame_last + 1, frame_stride):
                if frame_idx not in available_input_inds:
                    continue
                if frame_idx in available_target_inds:
                    target_frames_covered += 1

            frame_start = frame_first
            target_coverage = target_frames_covered / num_frames

            if target_frames_covered >= min_target_frames_covered:
                valid_modes.append((frame_start, frame_stride, target_coverage))

    return valid_modes


def clean_remain_reproducible(data_retval):
    '''
    Prunes a returned batch of examples such that it can be reconstructed deterministically.
        This is useful to save space for debugging, evaluation, and visualization, because
        data_retval can be huge.
    '''
    data_retval_pruned = my_utils.dict_to_cpu(
        data_retval, ignore_keys=['pv_rgb_tf', 'pv_depth_tf', 'pv_segm_tf', 'pv_coords_tf',
                                  'pv_xyz_tf', 'pv_div_segm_tf', 'pv_query_tf', 'pv_target_tf'])

    return data_retval_pruned


def get_approx_occlusion_risk(segm_ids, inst_count):
    '''
    :param segm_ids (T, H, W, 1) array of int in [0, K].
    :param inst_count (int) = K.
    :return occl_risk (K, T, 2) array of float in [0, 1] with (percentage, increase): Estimated
        occlusion percentage and change. The higher this value per object and per frame, the more
        likely it is currently (either partially or fully) occluded.
    '''
    (T, H, W, _) = segm_ids.shape
    K = inst_count
    occl_risk = np.zeros((K, T, 2), dtype=np.float32)  # (percentage, increase).

    # TODX we could be more advanced and exclude objects going out of frame somehow.

    for k in range(K):
        cur_pxl_per_frame = np.sum(segm_ids == k, axis=(1, 2))  # (T).
        for f in range(1, T - 1):
            cur_pxl_window = [cur_pxl_per_frame[g] for g in range(f-1,f+2)]
            
            # This value represents a change in occlusion fraction by comparing pixel counts.
            increase = (cur_pxl_window[0] - cur_pxl_window[2]) / (max(cur_pxl_window) + 1e-4)
            occl_risk[k, f, 1] = increase

    # Accumulate delta values to approximate occlusion percentage over time.
    occl_risk[..., 0] = np.cumsum(occl_risk[..., 1], axis=0)

    # We do not know what occlusion percentage the video started with, but we do know that this
    # fraction has to be within [0, 1] at every frame.
    for k in range(K):
        if occl_risk[k, :, 0].min() < 0.0:
            occl_risk[k, :, 0] -= occl_risk[..., 0].min()
        if occl_risk[k, :, 0].max() > 1.0:
            occl_risk[k, :, 0] /= occl_risk[..., 0].max()

    return occl_risk


def subsample_occlusion_risk(avail_occl_risk, avail_time_inds, frame_inds):
    '''
    :param avail_occl_risk (K, Tv, 2) array of float in [0, 1].
    :param avail_time_inds (Tv) array of int: Video on disk.
    :param frame_inds (Tc) array of int: Clip used for model.
    :return occl_risk (K, Tc, 2) array of float in [0, 1].
        NOTE: Tc is not necessarily smaller than Tv; this also depends on frame rates.
    '''
    (K, Tv, _) = avail_occl_risk.shape
    avail_time_inds = np.array(avail_time_inds)
    Tc = len(frame_inds)
    occl_risk = np.zeros((K, Tc, 2), dtype=np.float32)
    
    for k in range(K):
        for fc, t in enumerate(frame_inds):
            fv = np.argmin(np.abs(avail_time_inds - t))
            occl_risk[k, fc] = avail_occl_risk[k, fv]
    
    return occl_risk


def get_inst_area(pv_segm_tf, inst_count):
    '''
    :param pv_segm_tf (1, T, H, W) tensor of int in [0, K] with instance ID + 1.
    :param inst_count (int) = K.
    :return inst_area (K, T) array of float in [0, 1]: Fraction of pixels belonging to each object.
    '''
    (_, T, H, W) = pv_segm_tf.shape
    K = inst_count
    inst_area = np.zeros((K, T), dtype=np.float32)

    # for k in range(K):
    #     for f in range(T):
    #         inst_area[k, f] = torch.mean((pv_segm_tf[0, f] == k + 1).float()).item()

    for k in range(K):
        inst_area[k, :] = torch.mean((pv_segm_tf[0, :] == k + 1).float(), dim=(1, 2)).numpy()

    return inst_area


def _paths_from_txt(txt_fp):
    # First, simply obtain all non-empty, non-commented lines.
    with open(txt_fp, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    lines = [line for line in lines if not(line.lower().startswith('#'))]
    
    # Then, prepend non-existent (presumed relative) paths with the directory of the text file.
    # This allows for sharing text files across machines without having to modify the contents.
    txt_dp = str(pathlib.Path(txt_fp).parent)
    paths = []
    for line in lines:
        if os.path.exists(line):
            paths.append(line)
        else:
            absolute_path = os.path.join(txt_dp, line)
            assert os.path.exists(absolute_path), absolute_path
            paths.append(absolute_path)
    
    return paths


def get_data_paths_from_args(given_data_paths):
    '''
    Converts any text file into the the list of actual paths it contains as lines within.
    '''
    actual_data_paths = []
    for data_path in given_data_paths:
        if data_path.lower().endswith('.txt'):
            actual_data_paths += _paths_from_txt(data_path)
        else:
            actual_data_paths.append(data_path)
    return actual_data_paths


def fill_kubric_query_target_mask_flags(
        all_segm, all_div_segm, query_idx, qt_idx, occl_fracs, occl_cont_dag, scene_dp,
        logger, train_args, device, phase):
    (B, _, T, H, W) = all_segm.shape

    seeker_query_mask = torch.zeros_like(all_segm, dtype=torch.uint8)  # (B, 1, T, Hf, Wf).
    snitch_occl_by_ptr = torch.zeros_like(all_segm, dtype=torch.uint8)  # (B, 1, T, Hf, Wf).
    full_occl_cont_id = torch.zeros(B, T, 2, dtype=torch.uint8)
    target_mask = torch.zeros_like(all_div_segm[:, 0:3], dtype=torch.uint8)
    # (B, 3, T, Hf, Wf).
    target_flags = torch.zeros(B, T, 3, dtype=torch.float32)
    # Flags are (occluded, contained, soft_fraction).

    for b in range(B):

        # Fill in query mask (concentrated at a single frame).
        if train_args.xray_query:
            # Full ground truth instance mask.
            assert not(train_args.annot_visible_pxl_only)
            seeker_query_mask[b, 0, qt_idx] = (all_div_segm[b, query_idx[b], qt_idx] == 1)
        else:
            # Only visible instance pixels.
            seeker_query_mask[b, 0, qt_idx] = (all_segm[b, 0, qt_idx] == query_idx[b] + 1)

        # Prepare "snitch occluded by" pointers for this instance, useful for optimization &
        # evaluation. This tensor is zero outside the target mask, but ID + 1 inside.
        cur_occl_mask = torch.logical_and(all_div_segm[b, query_idx[b]] == 1,
                                            all_segm[b, 0] != query_idx[b] + 1)
        snitch_occl_by_ptr[b, 0, cur_occl_mask] = all_segm[b, 0, cur_occl_mask]

        # Mark snitch itself.
        if train_args.annot_visible_pxl_only and not('test' in phase):
            # Only visible instance pixels over time.
            target_mask[b, 0] = (all_segm[b, 0] == query_idx[b] + 1)
        else:
            # Full ground truth instance masks over time.
            target_mask[b, 0] = (all_div_segm[b, query_idx[b]] == 1)

        recursive_containment_info = []

        for t in range(T):

            frontmost_inst_id = -2
            outermost_inst_id = -2

            # Mark frontmost occluder and store ID + 1, where appropriate.
            # NOTE: Occlusion is a 2D principle, so retrieving frontmost from oc_dag may
            # sometimes be inaccurate due to (1) cropping, or (2) joint occluders where
            # none are significant.
            if occl_fracs[b, query_idx[b], t, 0] >= train_args.front_occl_thres and \
                    occl_cont_dag[b, t, query_idx[b], :, 2].max() >= \
            train_args.front_occl_thres / 2.0:
                frontmost_inst_id = occl_cont_dag[b, t, query_idx[b], :, 2].argmax().item()
                full_occl_cont_id[b, t, 0] = frontmost_inst_id + 1
                target_flags[b, t, 0] = 1

                if train_args.annot_visible_pxl_only and not('test' in phase):
                    target_mask[b, 1, t] = (all_segm[b, 0, t] == frontmost_inst_id + 1)
                else:
                    target_mask[b, 1, t] = (all_div_segm[b, frontmost_inst_id, t] == 1)

            # Mark outermost container and store ID + 1, where appropriate.
            # NOTE: Containment is a 3D principle, so this is accurate regardless of
            # cropping.
            if occl_cont_dag[b, t, query_idx[b], :, 0].max() >= \
                    train_args.outer_cont_thres:
                container_ids = torch.nonzero(
                    occl_cont_dag[b, t, query_idx[b], :, 0] >=
                    train_args.outer_cont_thres).flatten().tolist()
                outermost_inst_id = occl_cont_dag[b, t, query_idx[b], :, 0].argmax().item()

                # We may not have picked the outermost container yet, so search by checking whether
                # the selected container is itself fully contained. This comes down to a min max
                # optimization problem.

                if len(container_ids) > 1:
                    # Obtain whichever k is the least contained by any other k.
                    outermost_inst_id = min(
                        container_ids, key=lambda l: occl_cont_dag[b, t, l, :, 0].max())
                    
                    recursive_containment_info.append((t, container_ids, outermost_inst_id))

                full_occl_cont_id[b, t, 1] = outermost_inst_id + 1
                target_flags[b, t, 1] = 1

                if train_args.annot_visible_pxl_only and not('test' in phase):
                    target_mask[b, 2, t] = (all_segm[b, 0, t] == outermost_inst_id + 1)
                else:
                    target_mask[b, 2, t] = (all_div_segm[b, outermost_inst_id, t] == 1)

        # Occlusion percentage is trivial.
        target_flags[b, :, 2] = occl_fracs[b, query_idx[b], :, 0]

        # If we detected recursive containment, tell me about it for later inspection.
        if len(recursive_containment_info) > 0 and logger is not None:
            rc_frames = [x[0] for x in recursive_containment_info]
            container_ids = [x[1] for x in recursive_containment_info]
            outermost_inst_ids = [x[2] for x in recursive_containment_info]
            logger.info(f'Recursive containment detected in {scene_dp}!')
            logger.info(f'Frames (inclusive): {rc_frames[0]} - '
                                f'{rc_frames[-1]}  Query ID: {query_idx[b].item()}  '
                                f'Container IDs: {container_ids[0]} - {container_ids[-1]}  '
                                f'Outermost container: {outermost_inst_ids[0]} - '
                                f'{outermost_inst_ids[-1]}')

    seeker_query_mask = seeker_query_mask.type(torch.float32).to(device)
    snitch_occl_by_ptr = snitch_occl_by_ptr.to(device)
    full_occl_cont_id = full_occl_cont_id.to(device)
    target_mask = target_mask.type(torch.float32).to(device)
    target_flags = target_flags.to(device)

    return (seeker_query_mask, snitch_occl_by_ptr, full_occl_cont_id, target_mask, target_flags)
