import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from tqdm import tqdm
def save_concept_frames(vcd, single_save_layer_head, mode='random', num_vids_per_concept=10, alpha=0.6, extra_frames=0, mask_video_with_concepts=True,draw_mask_border=True ):

    concept_frames_dir = os.path.join(vcd.args.save_dir, 'concept_frames')

    if not os.path.exists(concept_frames_dir):
        os.makedirs(concept_frames_dir, exist_ok=True)

    for layer in tqdm(vcd.args.cluster_layer):


        for head in vcd.args.attn_head:



            if len(single_save_layer_head) > 0:
                layer = int(single_save_layer_head[0])
                head = int(single_save_layer_head[1])
                print('saving only layer {} head {}'.format(layer, head))

            # create concept directory
            layer_head_dir = os.path.join(concept_frames_dir, 'layer_{}/head_{}'.format(layer, head))
            if not os.path.exists(layer_head_dir):
                os.makedirs(layer_head_dir, exist_ok=True)

            # for each concept, save the video in order of closest to center
            for concept in vcd.dic[layer][head]['concepts']:
                # make directories for concept
                concept_layer_head_dir = os.path.join(layer_head_dir, concept)
                if not os.path.exists(concept_layer_head_dir):
                    os.makedirs(concept_layer_head_dir, exist_ok=True)

                # grab rgb videos and masks in concept
                if vcd.args.dataset == 'kubric':
                    concept_rgb_videos = vcd.dataset['pv_rgb_tf'][vcd.dic[layer][head][concept]['video_numbers']]
                    # concept_query_masks = torch.stack(vcd.dataset['seeker_query_mask'])[vcd.dic[layer][head][concept]['video_numbers']]
                    if 'timesformer' in vcd.args.model:
                        target_masks = torch.stack(vcd.dataset['target_mask'])[vcd.dic[layer][head][concept]['video_numbers']]
                else:
                    concept_rgb_videos = vcd.dataset[vcd.dic[layer][head][concept]['video_numbers']]

                if mode == 'closest' and 'cnmf' in vcd.args.inter_cluster_method:
                    concept_masks = vcd.dic[layer][head][concept]['video_mask']
                    concept_idx = (int(concept.split('_')[-1] ) -1)
                    G = vcd.dic[layer][head]['cnmf']['G']
                    assign_matrix = G[G.argmax(1) == concept_idx, concept_idx]
                    vid_nums = np.argsort(assign_matrix)[::-1]
                else:
                    # join masks if we have more than one from the same video
                    vid_nums = range(len(vcd.dic[layer][head][concept]['video_numbers']))
                    concept_masks = vcd.dic[layer][head][concept]['video_mask']


                # [6 18  4 34 40 31 43 10 11 38 29 15 30 37 41 26  7 19  0  1  8 32 25  3, 28 33 23 22 27  5 17 39 16 20 13  9 44  2 35 24 42 21 14 12 36]
                # save each video in concept
                for num_vid, i in enumerate(vid_nums):

                    # replace vcd.dic[layer][head][concept]['video_numbers'] with a list of video numbers based on 1) distance from center 2) random_order


                    if num_vid == num_vids_per_concept:
                        break
                    rgb_video = concept_rgb_videos[i].permute(1 ,2 ,3 ,0)
                    mask_path = concept_masks[i]
                    # load mask from path
                    if isinstance(mask_path, list):
                        # combine all masks into one
                        single_mask = torch.tensor(np.stack([np.load(path) for path in mask_path], axis=0).sum(axis=0))
                    else:
                        single_mask = torch.tensor(np.load(mask_path))
                    single_mask = vcd.post_resize_nearest(single_mask)
                    mask = np.array \
                        (np.repeat(single_mask.unsqueeze(0), 3, axis=0).permute(1, 2, 3, 0))  # repeat along channels
                    if mask_video_with_concepts:
                        vis_concept_assign = vcd.create_concept_mask_video(rgb_video, mask, alpha=alpha, blend_white=True)
                    else:
                        vis_concept_assign = np.array(rgb_video)

                    if vcd.args.dataset == 'kubric' and 'timesformer' in vcd.args.model:
                        # draw query mask
                        # new
                        target_mask = target_masks[i].detach().cpu().numpy()[0, 0]
                        target_border = vcd.draw_segm_borders(target_mask[..., None], fill_white=False) # target_mask[..., None] = 30 x 240 x 320 x 1
                        vis_concept_assign = vcd.create_model_input_video(vis_concept_assign, target_mask, target_border ,extra_frames=extra_frames, target=True, color='green')
                    if draw_mask_border:
                        concept_mask = np.array(single_mask.float())
                        mask_border = vcd.draw_segm_borders(concept_mask[..., None], fill_white=False)
                        vis_concept_assign = vcd.create_model_input_video(vis_concept_assign, concept_mask, mask_border,extra_frames=extra_frames, target=True, color='orange')

                    concept_layer_head_vid_dir = os.path.join(concept_layer_head_dir, 'vid_{}'.format(num_vid))
                    if not os.path.exists(concept_layer_head_vid_dir):
                        os.makedirs(concept_layer_head_vid_dir, exist_ok=True)
                    # save frames
                    for frame_idx in range(vis_concept_assign.shape[0]):
                        img = Image.fromarray((vis_concept_assign[frame_idx] * 255).astype(np.uint8))
                        img.save(os.path.join(concept_layer_head_vid_dir, 'frame_{}.png'.format(frame_idx)))

            if len(single_save_layer_head) > 0:
                return