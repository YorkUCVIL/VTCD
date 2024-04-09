'''
Dataset and annotation visualization methods, usually for temporary debugging.
Created by Basile Van Hoorick, Sep 2022.
'''

from models.hide_seek.tcow.__init__ import *

# Internal imports.
import models.hide_seek.tcow.utils.my_utils as my_utils
import models.hide_seek.tcow.eval.visualization


def depth_to_rgb_vis(depth, max_depth=None):
    '''
    :depth (*, 1) array of float32.
    :return rgb_vis (*, 3) array of uint8.
    '''
    min_depth = 0.0
    if max_depth is None:
        max_depth = max(np.max(depth), 1e-6)

    depth = depth.copy().squeeze(-1)
    depth = np.clip(depth, 0.0, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth)

    rgb_vis = plt.cm.viridis(2.0 / (depth + 1.0) - 1.0)[..., :3]
    rgb_vis = (rgb_vis * 255.0).astype(np.uint8)

    return rgb_vis


def segm_rgb_to_ids_kubric(segm_rgb):  # , num_inst=None):
    '''
    :param segm_rgb (*, 3) array of RGB values.
    :return segm_ids (*, 1) array of 1-based instance IDs (0 = background).
    '''
    # We assume that hues are distributed across the range [0, 1] for instances in the image, ranked
    # by their integer ID. Check kubric plotting.hls_palette() for more details.
    hsv = matplotlib.colors.rgb_to_hsv(segm_rgb)
    to_rank = hsv[..., 0]  # + hsv[..., 2] * 1e-5
    unique_hues = np.sort(np.unique(to_rank))
    hue_start = 0.01
    assert np.isclose(unique_hues[0], 0.0, rtol=1e-3, atol=1e-3), str(unique_hues)
    assert np.isclose(unique_hues[1], hue_start, rtol=1e-3, atol=1e-3), str(unique_hues)

    # The smallest jump inbetween subsequent hues determines the highest instance ID that is VALO,
    # which is <= the total number of instances. Skip the very first hue, which is always 0 and
    # corresponds to background.
    hue_steps = np.array([unique_hues[i] - unique_hues[i - 1] for i in range(2, len(unique_hues))])

    # For this sanity check to work, we must never have more than ~95 instances per scene.
    assert np.all(hue_steps >= 1e-2), str(hue_steps)

    # IMPORTANT NOTE: The current VALO set may be a strict SUBSET of the original VALO set (recorded
    # in the metadata), because we already applied frame subsampling here! In practice, this
    # sometimes causes big (i.e. integer multiple) jumps in hue_steps.
    # BUG until 9/21: e.g. would return 0.035 because of outliers, but we need 0.033.
    # hue_step = np.mean(hue_steps)  
    # ALSO BUG: Avoid min because of variations -- this also causes one-off errors.
    # hue_step = np.min(hue_steps)
    # NEW: Ignore outliers the smart way.
    adjacent_steps = hue_steps[hue_steps <= np.min(hue_steps) * 1.5]
    hue_step = np.mean(adjacent_steps)

    # The jump from background to first instance is a special case, so ensure even distribution.
    nice_rank = to_rank.copy()
    nice_rank[nice_rank >= hue_start] += hue_step - hue_start
    ids_approx = (nice_rank / hue_step)

    segm_ids = np.round(ids_approx)[..., None].astype(np.int32)  # (T, H, W, 1).
    return segm_ids


def segm_rgb_to_ids_ytvos(segm_rgb, unique_hues):
    '''
    :param segm_rgb (*, 3) array of RGB values in [0, 1], or -1 if unavailable.
    :return (segm_ids, unique_hues).
        segm_ids (*, 1) array of 1-based instance IDs (0 = background, -1 = unavailable).
        unique_hues (list) of K found hue values in [0, 1].
    '''
    hsv = matplotlib.colors.rgb_to_hsv(segm_rgb)
    to_rank = hsv[..., 0]  # (T, H, W) array of float32 in [0, 1].
    
    # I verified that first instance (ID = 1) has hue > 0 to distinguish itself from background.
    if unique_hues is None:
        unique_hues = np.sort(np.unique(to_rank))
    assert len(unique_hues) <= 127, str(unique_hues)
    
    # Calculate dense pair-wise distance array between all pixels and all available hues.
    hue_dists = np.abs(to_rank[..., None] - unique_hues[None, None, None, :])  # (T, H, W, K+1).
    segm_ids = np.argmin(hue_dists, axis=-1).astype(np.int32)  # (T, H, W) array of int32 in [0, K].

    unavailable_frames = np.any(segm_rgb < -0.5, axis=(1, 2, 3))
    unavailable_frames = np.broadcast_to(unavailable_frames[..., None, None], segm_ids.shape)
    segm_ids[unavailable_frames] = -1
    # (T, H, W) array of int32 in [-1, K], where 0 = background, -1 = unavailable,
    # and other values = instance ID + 1.

    segm_ids = segm_ids[..., None]  # (T, H, W, 1).
    return (segm_ids, unique_hues)


def segm_ids_to_rgb(segm_ids, num_inst=None):
    '''
    NOTE: This is NOT consistent with segm_rgb_to_ids_kubric(), because background (0) gets mapped
        to red!
    :segm_ids (*, 1) array of uint32.
    :return segm_rgb (*, 3) array of uint8.
    '''
    if num_inst is None:
        num_inst = np.max(segm_ids) + 1
    num_inst = max(num_inst, 1)

    segm_ids = segm_ids.copy().squeeze(-1)
    segm_ids = segm_ids / num_inst

    segm_rgb = plt.cm.hsv(segm_ids)[..., :3]
    segm_rgb = (segm_rgb * 255.0).astype(np.uint8)

    return segm_rgb


def create_rich_annot_video(pv_rgb, pv_div_segm, occl_cont_dag, boxes_3d_vis, which, logger,
                            frame_rate, file_name=None, ignore_if_exist=False,
                            front_occl_thres=0.95, outer_cont_thres=0.75):
    '''
    Exports extra dataset visualizations in the logs/visuals folder by horizontally concatenating
        the specified types.
    :param pv_rgb (T, Hf, Wf, 3) array of float32 in [0, 1].
    :param pv_div_segm (T, Hf, Wf, K) array of uint8 in [0, 1].
    :param occl_cont_dag (T, K, K, 3) array of float32 with (c, od, of) in [0, 1].
    :param which (list): Choices are xray_segm, oc_mark, oc_dag.
    '''
    if file_name is not None and ignore_if_exist and os.path.exists(file_name + '.webm'):
        return

    (T, H, W, K) = pv_div_segm.shape
    gal_items = []

    if 'rgb' in which:
        my_vid = pv_rgb
        
        gal_items.append(my_vid)

    if 'xray_segm' in which:
        my_vid = my_utils.quick_pca(pv_div_segm, k=3, unique_features=True, normalize=[0.0, 1.0])
        # (T, Hf, Wf, 3) floats in [0, 1].

        my_vid += visualization.draw_segm_borders(pv_div_segm, fill_white=True)
        
        gal_items.append(my_vid)

    if 'oc_mark' in which:
        my_vid = np.zeros((T, H, W, 3), dtype=np.float32)
        
        for k in range(K):
            for f in range(T):
                
                if occl_cont_dag[f, :, k, 0].max() >= outer_cont_thres:  # I contain things => blue.
                    my_vid[f, :, :, 2] += pv_div_segm[f, :, :, k] * 0.9
                if occl_cont_dag[f, k, :, 0].max() >= outer_cont_thres:  # Contained by something => gray minus blue.
                    my_vid[f, :, :, 0] += pv_div_segm[f, :, :, k] * 0.4
                    my_vid[f, :, :, 1] += pv_div_segm[f, :, :, k] * 0.4
                    my_vid[f, :, :, 2] += pv_div_segm[f, :, :, k] * 0.4
                    my_vid[f, :, :, 2] -= pv_div_segm[f, :, :, k] * 0.8
                
                # NOTE: We look at direct occlusion (od) between pairs of isolated objects. This
                # means that recursive cases may turn yellow. Unfortunately, this value is not as
                # reliable as of (final / frontmost).
                if occl_cont_dag[f, :, k, 1].max() >= front_occl_thres:  # I occlude things => red.
                    my_vid[f, :, :, 0] += pv_div_segm[f, :, :, k] * 0.7
                if occl_cont_dag[f, k, :, 1].max() >= front_occl_thres:  # Occluded by something => green minus red.
                    my_vid[f, :, :, 1] += pv_div_segm[f, :, :, k] * 0.7
                    my_vid[f, :, :, 0] -= pv_div_segm[f, :, :, k] * 0.5

        my_vid += visualization.draw_segm_borders(pv_div_segm, fill_white=True)
        
        gal_items.append(my_vid)

    if 'oc_dag' in which:
        my_vid = np.zeros((T, H, W, 3), dtype=np.float32)
        
        for f in range(T):
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.tight_layout(pad=0.5)
            
            sns.heatmap(occl_cont_dag[f, ..., 1], ax=ax, fmt='g', linewidths=0.5, vmin=0.0, vmax=1.0)
            
            my_plot = cv2.resize(my_utils.ax_to_numpy(ax, dpi=192), (W, H))
            my_vid[f] = my_plot
            plt.close()
        
        gal_items.append(my_vid)

    if 'box3d' in which:
        my_vid = boxes_3d_vis * 0.9 + pv_rgb * 0.4
        # (T, Hf, Wf, 3) floats in [0, 1].
        
        # Copy part of oc_mark for containment.
        # TODX DRY
        for k in range(K):
            for f in range(T):
                
                if occl_cont_dag[f, :, k, 0].max() >= outer_cont_thres:  # I contain things => blue.
                    my_vid[f, :, :, 2] += pv_div_segm[f, :, :, k] * 0.9 / 1.5
                if occl_cont_dag[f, k, :, 0].max() >= outer_cont_thres:  # Contained by something => gray minus blue.
                    my_vid[f, :, :, 0] += pv_div_segm[f, :, :, k] * 0.4 / 1.5
                    my_vid[f, :, :, 1] += pv_div_segm[f, :, :, k] * 0.4 / 1.5
                    my_vid[f, :, :, 2] += pv_div_segm[f, :, :, k] * 0.4 / 1.5
                    my_vid[f, :, :, 2] -= pv_div_segm[f, :, :, k] * 0.8 / 1.5
        
        gal_items.append(my_vid)

    (G1, G2) = [(1, 1), (1, 2), (1, 3), (2, 2), (1, 5), (2, 3)][len(which) - 1]
    gal_vid = rearrange(gal_items, '(G1 G2) T H W C -> T (G1 H) (G2 W) C', G1=G1, G2=G2)
    gal_vid = np.clip(gal_vid, 0.0, 1.0)

    if file_name is not None:
        stub_idx = np.random.randint(0, 10000)
        logger.save_video(
            gal_vid, step=stub_idx, file_name=file_name,
            # online_name=None, caption=None, extensions=['.gif', '.mp4'],
            online_name=None, caption=None, extensions=['.webm'],
            fps=frame_rate // 2, upscale_factor=2)
            # fps=frame_rate, upscale_factor=2)
