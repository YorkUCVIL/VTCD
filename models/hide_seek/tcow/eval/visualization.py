'''
Tools / utils / helper methods pertaining to qualitative deep dives into train / test results.
Created by Basile Van Hoorick, Jul 2022.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'eval/'))
sys.path.insert(0, os.getcwd())

from __init__ import *


def draw_text(image, x, y, label, color, size_mult=1.0):
    '''
    :param image (H, W, 3) array of float32 in [0, 1].
    :param x (int): x coordinate of the top left corner of the text.
    :param y (int): y coordinate of the top left corner of the text.
    :param label (str): Text to draw.
    :param color (3) tuple of float32 in [0, 1]: RGB values.
    :param size_mult (float): Multiplier for font size.
    :return image (H, W, 3) array of float32 in [0, 1].
    '''
    # Draw background and write text using OpenCV.
    label_width = int((16 + len(label) * 10) * size_mult)
    label_height = int(22 * size_mult)
    image[y:y + label_height, x:x + label_width] = (0, 0, 0)
    image = cv2.putText(image, label, (x, y + label_height - 8), 2,
                        0.5 * size_mult, color, thickness=int(size_mult))
    return image


def draw_segm_borders(segm, fill_white=False):
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


def create_model_input_video(seeker_rgb, seeker_query_mask, query_border):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param seeker_query_mask (T, H, W) array of float32 in [0, 1].
    :param query_border (T, H, W, 3) array of float32 in [0, 1].
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    query_time = seeker_query_mask.any(axis=(1, 2)).argmax()
    vis = seeker_rgb + seeker_query_mask[..., None]    # (T, H, W, 3).

    vis[query_time] *= 0.6
    vis[query_border, 0] = 0.0
    vis[query_border, 1] = 1.0
    vis[query_border, 2] = 0.0

    # Pause for a bit at query time to make the instance + mask very clear.
    vis = np.concatenate([vis[0:query_time]] +
                         [vis[query_time:query_time + 1]] * 3 +
                         [vis[query_time + 1:]], axis=0)

    video = np.clip(vis, 0.0, 1.0)
    return video


def create_model_output_snitch_video(
        seeker_rgb, output_mask, query_border, snitch_border, grayscale=False):
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
    vis = seeker_rgb * 0.6 + snitch_heatmap * 0.5  # (T, H, W, 3).

    vis[query_border] = 0.0
    vis[snitch_border] = 0.0
    vis[query_border, 0] = 1.0
    vis[query_border, 2] = 1.0
    vis[snitch_border, 1] = 1.0

    video = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).
    return video


def create_model_output_snitch_occl_cont_video(
        seeker_rgb, output_mask, query_border, snitch_border, frontmost_border,
        outermost_border, grayscale=False):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param output_mask (1/3, T, H, W) array of float32 in [0, 1].
    :param query_border (T, H, W) array of float32 in [0, 1].
    :param snitch_border (T, H, W) array of float32 in [0, 1].
    :param frontmost_border (T, H, W) array of float32 in [0, 1].
    :param outermost_border (T, H, W) array of float32 in [0, 1].
    :param grayscale (bool): Whether to convert the output to grayscale.
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    if grayscale:
        seeker_rgb = seeker_rgb.copy()
        seeker_gray = seeker_rgb[..., 0] * 0.2 + seeker_rgb[..., 1] * 0.6 + seeker_rgb[..., 2] * 0.2
        seeker_rgb[..., 0] = seeker_gray
        seeker_rgb[..., 1] = seeker_gray
        seeker_rgb[..., 2] = seeker_gray

    vis = seeker_rgb * 0.6
    # colors of clips
    vis[..., 1] += output_mask[0] * 0.5  # Snitch = green.
    if output_mask.shape[0] >= 2:
        vis[..., 0] += output_mask[1] * 0.5  # Frontmost occluder = red.
    if output_mask.shape[0] >= 3:
        vis[..., 2] += output_mask[2] * 0.5  # Outermost container = blue.

    vis[query_border] = 0.0
    vis[snitch_border] = 0.0
    vis[frontmost_border] = 0.0
    vis[outermost_border] = 0.0
    vis[query_border] = 1.0  # Always all white.
    vis[snitch_border, 1] = 1.0
    vis[frontmost_border, 0] = 1.0
    vis[outermost_border, 2] = 1.0

    video = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).
    return video


def create_snitch_weights_video(seeker_rgb, snitch_weights):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param snitch_weights (T, H, W) array of float32 in [0, inf).
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    slw_norm = snitch_weights.max() + 1e-6
    lw_heatmap = plt.cm.viridis(snitch_weights / slw_norm)[..., 0:3]
    vis = seeker_rgb * 0.6 + lw_heatmap * 0.5  # (T, H, W, 3).

    video = np.clip(vis, 0.0, 1.0)  # (T, H, W, 3).
    return video


def create_model_input_target_video(
        seeker_rgb, seeker_query_mask, target_mask, query_border, snitch_border, frontmost_border,
        outermost_border, grayscale=False):
    '''
    :param seeker_rgb (T, H, W, 3) array of float32 in [0, 1].
    :param seeker_query_mask (T, H, W) array of float32 in [0, 1].
    :param target_mask (3, T, H, W) array of float32 in [0, 1].
    :param query_border (T, H, W, 3) array of float32 in [0, 1].
    :param snitch_border (T, H, W) array of float32 in [0, 1].
    :param frontmost_border (T, H, W) array of float32 in [0, 1].
    :param outermost_border (T, H, W) array of float32 in [0, 1].
    :param grayscale (bool): Whether to convert the output to grayscale.
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''
    if grayscale:
        seeker_rgb = seeker_rgb.copy()
        seeker_gray = seeker_rgb[..., 0] * 0.2 + seeker_rgb[..., 1] * 0.6 + seeker_rgb[..., 2] * 0.2
        seeker_rgb[..., 0] = seeker_gray
        seeker_rgb[..., 1] = seeker_gray
        seeker_rgb[..., 2] = seeker_gray
    
    vis = seeker_rgb.copy()    # (T, H, W, 3).

    # NOTE / TODO: I'm a bit torn here on how weak or strong to make the mask fillings,
    # since the reader may confuse ground truths with predictions that way?

    # Fill in query mask for clarity.
    vis += seeker_query_mask[..., None] * 0.3

    # Fill in all target mask channels for clarity.
    target_mask = np.clip(target_mask, 0.0, 1.0)
    vis[1:, ..., 1] += target_mask[0, 1:] * 0.2  # Snitch = green, but ignore first frame (query).
    if target_mask.shape[0] >= 2:
        vis[..., 0] += target_mask[1] * 0.2  # Frontmost occluder = red.
    if target_mask.shape[0] >= 3:
        vis[..., 2] += target_mask[2] * 0.2  # Outermost container = blue.

    vis[query_border] = 0.0
    vis[snitch_border] = 0.0
    vis[frontmost_border] = 0.0
    vis[outermost_border] = 0.0
    vis[query_border] = 1.0  # Always all white.
    vis[snitch_border, 1] = 1.0
    vis[frontmost_border, 0] = 1.0
    vis[outermost_border, 2] = 1.0

    video = np.clip(vis, 0.0, 1.0)
    return video


def mark_no_gt_risky_frames_bottom_video(old_video, no_gt_frames, risky_frames):
    '''
    :return old_video (T, H, W, 3) array of float32 in [0, 1].
    :param no_gt_frames (T) array of bool.
    :param risky_frames (T) array of bool.
    :return new_video (T, H, W, 3) array of float32 in [0, 1].
    '''
    (T, H, W, _) = old_video.shape
    new_video = old_video.copy()
    banner = int(H / 36.0)

    if no_gt_frames is not None or risky_frames is not None:
        new_video[no_gt_frames, -banner:, :, 0] = 0.0
        new_video[no_gt_frames, -banner:, :, 1] = 0.5
        new_video[no_gt_frames, -banner:, :, 0] = 0.0

    if no_gt_frames is not None:
        new_video[no_gt_frames, -banner:, :, 2] = 1.0
    if risky_frames is not None:
        new_video[risky_frames, -banner:, :, 0] = 1.0

    return new_video


def append_flags_top_video(old_video, output_mask, target_mask, output_occl_pct, target_occl_pct):
    '''
    :param old_video (T, H, W, 3) array of float32 in [0, 1].
    :param output_mask (1/3, T, H, W) array of float32 in [0, 1].
    :param target_mask (1/3, T, H, W) array of float32 in [0, 1].
    :param output_occl_pct (T) array of float32 in [0, 1].
    :param target_occl_pct (T) array of float32 in [0, 1].
    :return new_video (T, H, W, 3) array of float32 in [0, 1].
    '''
    (T, H, W, _) = old_video.shape
    padding = int(H / 8.0)
    font_size = H / 220.0
    new_video = np.zeros((T, H + padding, W, 3), dtype=np.float32)
    new_video[:, padding:, :, :] = old_video

    # target_occl_flag = (target_mask[1].mean(axis=(1, 2)) > 0.0)  # (T).
    # target_cont_flag = (target_mask[2].mean(axis=(1, 2)) > 0.0)  # (T).
    if output_mask.shape[0] >= 2:
        output_occl_flag = ((output_mask[1] >= 0.7).mean(axis=(1, 2)) >= 0.01)  # (T).
    else:
        output_occl_flag = None
    if output_mask.shape[0] >= 3:
        output_cont_flag = ((output_mask[2] >= 0.7).mean(axis=(1, 2)) >= 0.01)  # (T).
    else:
        output_cont_flag = None

    frames = [x for x in new_video]
    if output_cont_flag is not None:
        for f in range(T):
            frames[f] = draw_text(
                frames[f], 4, 3, f'Cont: {"Yes" if output_cont_flag[f] else "No"}',
                (0.4, 0.4, 1.0) if output_cont_flag[f] else (0.2, 0.2, 0.4), font_size)
    if output_occl_flag is not None:
        for f in range(T):
            frames[f] = draw_text(
                frames[f], 4 + int(W / 3.0), 3, f'Occl: {"Yes" if output_occl_flag[f] else "No"}',
                (1.0, 0.4, 0.4) if output_occl_flag[f] else (0.4, 0.2, 0.2), font_size)
    if output_occl_pct is not None:
        for f in range(T):
            pct_100 = int(round(np.clip(output_occl_pct[f], 0.0, 1.0) * 100.0))
            frames[f] = draw_text(
                frames[f], 4 + int(W * 2.0 / 3.0), 3, f'Pct: {pct_100}%',
                (0.3, 0.7, 0.3), font_size)
    new_video = np.stack(frames)

    if output_occl_pct is not None:
        for f in range(T):
            occl_bar = int(round(np.clip(output_occl_pct[f], 0.0, 1.0) * W))
            new_video[f, padding - 3:padding, :occl_bar] = 0.9

    return new_video



def create_concept_mask_video(input_video, concept_mask, alpha=0.5):
    '''
    :param input_video (T, H, W, 3) array of float32 in [0, 1].
    :param concept_mask (T, H, W) array of int in {0, 1}.
    :return video (T, H, W, 3) array of float32 in [0, 1].
    '''

    video = np.where(concept_mask == 1, input_video, input_video*alpha)

    return video

def create_heatmap_video(input_video, concept_mask, alpha=0.5):
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