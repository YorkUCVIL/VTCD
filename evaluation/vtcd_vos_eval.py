import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from scipy import ndimage
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.ops import masks_to_boxes
resize_longest_side = ResizeLongestSide(1024)

def compute_davis16_vos_score(vcd, first_frame_query=False, per_video_head=True,
                      train_head_search=False, post_sam=False, num_points=1, mode='random',
                      sample_factor=8, use_last_centroid=False, use_box=False):
    '''
    Computes VOS score for each concept in each layer and head for the DAVIS dataset
    :param vcd:
    :return:
    '''

    if post_sam:
        import sys
        sys.path.append("..")
        from segment_anything import sam_model_registry, SamPredictor

        sam_checkpoint = "segment_anything/ckpts/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        # sam_checkpoint = "segment_anything/ckpts/sam_vit_b_01ec64.pth"
        # model_type = "vit_b"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        predictor = SamPredictor(sam)

    # load RGB dataset
    if 'timesformer' in vcd.args.model:
        sampling_rate = 1
        num_frames = 30
    else:
        sampling_rate = 2
        num_frames = 16
        # reloading dataset
    vcd.dataset = vcd.load_davis16_videos(sampling_rate, num_frames)

    save_path = os.path.join(vcd.args.save_dir, 'davis16_vos_results.json')

    # just for printing best heads
#     if os.path.exists(save_path) and not first_frame_query and train_head_search:
#         with open(save_path, 'r') as f:
#             results = json.load(f)
#         best_heads = []
#         for video_idx in results.keys():
#             best_heads.append(results[video_idx][0][0].split('concept')[0][:-1])
#
#         best_heads_norepeat_print = list(dict.fromkeys(best_heads))
#         # print best heads
#         for head in best_heads_norepeat_print:
#             print(head)

    if train_head_search:
        train_path = save_path.replace('val', 'train')
        with open(train_path, 'r') as f:
            results = json.load(f)
        best_heads = []
        for video_idx in results.keys():
            best_heads.append(results[video_idx][0][0].split('concept')[0][:-1])

        best_heads_norepeat = list(dict.fromkeys(best_heads))



    if per_video_head:
        results = {}
        per_frame_iou = []
        # compute iou between every concept and every video
        for video_idx in tqdm(range(len(vcd.labels))):
            results[video_idx] = {}
            label = vcd.labels[video_idx]
            if first_frame_query:
                best_dict = {
                    'layer': None,
                    'head': None,
                    'concept': None,
                    'iou': 0
                }

                for layer in reversed(vcd.dic.keys()):
                    for head in vcd.args.attn_head:
                        if train_head_search:
                            lay_head = 'layer_{} head_{}'.format(layer, head)
                            if lay_head not in best_heads_norepeat:
                                continue
                        for concept in vcd.dic[layer][head]['concepts']:
                            # grab concept mask and video ids
                            video_ids = vcd.dic[layer][head][concept]['video_numbers']
                            concept_masks = vcd.dic[layer][head][concept]['video_mask']
                            if isinstance(concept_masks, list):
                                if not vcd.args.process_full_video:
                                    concept_masks = torch.stack([torch.tensor(np.load(mask)) for mask in concept_masks],
                                                                dim=0)
                                else:
                                    concept_masks = [torch.tensor(np.load(mask)) for mask in concept_masks]

                            # grab labels and concept masks for video_idx
                            video_ids = torch.tensor(video_ids)
                            video_ids = video_ids == video_idx
                            if video_ids.sum() != 0:
                                if not vcd.args.process_full_video:
                                    # compute iou for first frames
                                    iou = compute_iou(concept_masks[:, 0], label[:, 0])
                                else:
                                    # concept_masks = [concept_masks[i] for i in range(len(concept_masks)) if video_ids[i]]
                                    concept_masks = torch.stack([concept_masks[i] for i in range(len(concept_masks)) if video_ids[i]], dim=0)
                                    single_tubelet_mask = concept_masks.sum(0) > 0
                                    iou = compute_iou(single_tubelet_mask[0], label[0])

                                if iou > best_dict['iou']:
                                    best_dict['layer'] = layer
                                    best_dict['head'] = head
                                    best_dict['concept'] = concept
                                    best_dict['iou'] = iou
                                    best_concept_mask = single_tubelet_mask


                            # compute iou for best concept over all frames
                video_ids = vcd.dic[best_dict['layer']][best_dict['head']][best_dict['concept']]['video_numbers']
                if video_idx not in video_ids:
                    final_iou = 0
                else:
                    if not vcd.args.process_full_video:
                        labels = torch.stack(vcd.labels)[video_idx]
                        labels = labels.unsqueeze(0).repeat(concept_masks, 1, 1, 1)
                        if post_sam:
                            # postprocess masks
                            concept_masks = postprocess(concept_masks, vcd.dataset[video_idx], predictor,
                                                        num_points=num_points, mode=mode,
                                                        sample_factor=sample_factor,use_last_centroid=use_last_centroid,
                                                        use_box=use_box)
                            labels = torch.stack(vcd.labels)[video_idx]
                        # compute iou
                        final_iou = compute_iou(concept_masks, labels)
                    else:
                        # grab label for video_idx
                        if post_sam:
                            # postprocess masks
                            final_mask = postprocess(best_concept_mask.unsqueeze(0), vcd.dataset[video_idx], predictor,
                                                        num_points=num_points, mode=mode,
                                                        sample_factor=sample_factor,use_last_centroid=use_last_centroid,
                                                        use_box=use_box)
                            # video_iou = compute_iou(final_mask, label)
                            per_video_frame_iou = []
                            for frame_idx in range(label.shape[0]):
                                iou = compute_iou(final_mask[frame_idx], label[frame_idx])
                                per_video_frame_iou.append(iou)
                                per_frame_iou.append(iou)
                            per_video_frame_iou = np.mean(per_video_frame_iou)
                        else:
                            final_mask = best_concept_mask
                            # video_iou = compute_iou(best_concept_mask, label)
                            per_video_frame_iou = []
                            for frame_idx in range(label.shape[0]):
                                iou = compute_iou(final_mask[frame_idx], label[frame_idx])
                                per_video_frame_iou.append(iou)
                                per_frame_iou.append(iou)
                            per_video_frame_iou = np.mean(per_video_frame_iou)

                        # save prediction
                        save_prediction(vcd, vcd.dataset[video_idx], final_mask, best_concept_mask, label, video_idx, 0, best_dict, save_prefix='d16')

                # print('Best concept based on query frame: {}, iou: {}'.format(best_concept, iou))
                key = 'layer_{} head_{} {}'.format(best_dict['layer'], best_dict['head'], best_dict['concept'])
                results[video_idx][key] = per_video_frame_iou
            else:
                for layer in reversed(vcd.dic.keys()):
                    for head in vcd.args.attn_head:
                        for concept in vcd.dic[layer][head]['concepts']:
                            # video ids for concept
                            video_ids = vcd.dic[layer][head][concept]['video_numbers']

                            # pass if video not in concept
                            if video_idx not in video_ids:
                                iou = 0
                            else:
                                # otherwise grab concept mask and measure iou
                                concept_masks = vcd.dic[layer][head][concept]['video_mask']
                                if isinstance(concept_masks, list):
                                    if not vcd.args.process_full_video:
                                        concept_masks = torch.stack([torch.tensor(np.load(mask)) for mask in concept_masks], dim=0)
                                    else:
                                        concept_masks = [torch.tensor(np.load(mask)) for mask in concept_masks]

                                # grab labels and concept masks for video_idx
                                video_ids = torch.tensor(video_ids)
                                video_ids = video_ids == video_idx
                                if not vcd.args.process_full_video:
                                    concept_masks = concept_masks[video_ids]

                                    # compute iou
                                    iou = compute_iou(concept_masks, labels)
                                else:
                                    concept_masks = torch.stack([concept_masks[i] for i in range(len(concept_masks)) if video_ids[i]], dim=0)

                                    single_tubelet_mask = concept_masks.sum(0) > 0

                                    # compute iou
                                    iou = compute_iou(single_tubelet_mask, label)

                                    per_video_frame_iou = []
                                    for frame_idx in range(label.shape[0]):
                                        iou = compute_iou(single_tubelet_mask[frame_idx], label[frame_idx])
                                        per_video_frame_iou.append(iou)
                                        per_frame_iou.append(iou)
                                    per_video_frame_iou = np.mean(per_video_frame_iou)

                            key = 'layer_{} head_{} {}'.format(layer, head, concept)
                            results[video_idx][key] = iou


            # sort results by iou for current video
            curr_video_results = sorted(results[video_idx].items(), key=lambda x: x[1], reverse=True)
            print('video: {}, best head: {}, iou: {}'.format(video_idx, curr_video_results[0][0], curr_video_results[0][1]))

        # sort results by iou
        for video_idx in results.keys():
            results[video_idx] = sorted(results[video_idx].items(), key=lambda x: x[1], reverse=True)

        # calculate average iou over all videos for the best head
        best_ious = []
        for video_idx in results.keys():
            best_ious.append(results[video_idx][0][1])
        mIoU = np.mean(best_ious)
        print('mIoU: {}'.format(mIoU))

        per_frame_iou = np.mean(per_frame_iou)
        print('per frame iou: {}'.format(per_frame_iou))

    else:
        results = {}

        for layer in tqdm(vcd.dic.keys()):
            for head in vcd.args.attn_head:

                # compute iou for each concept and first frame to see which concept is most important
                if first_frame_query:
                    best_concept = None
                    best_iou = 0
                    for concept in vcd.dic[layer][head]['concepts']:
                        # grab concept mask and video ids
                        concept_masks = vcd.dic[layer][head][concept]['video_mask']
                        video_ids = vcd.dic[layer][head][concept]['video_numbers']

                        if isinstance(concept_masks, list):
                            concept_masks = torch.stack([torch.tensor(np.load(mask)) for mask in concept_masks], dim=0)

                        # grab gt concept masks
                        labels = torch.stack(vcd.labels)[video_ids]

                        # compute iou for first frames
                        iou = compute_iou(concept_masks[:, 0], labels[:, 0])

                        if iou > best_iou:
                            best_concept = concept
                            best_iou = iou

                    # compute iou for best concept over all frames
                    concept_masks = vcd.dic[layer][head][best_concept]['video_mask']
                    video_ids = vcd.dic[layer][head][best_concept]['video_numbers']

                    if isinstance(concept_masks, list):
                        concept_masks = torch.stack([torch.tensor(np.load(mask)) for mask in concept_masks], dim=0)

                    # grab gt concept masks
                    labels = torch.stack(vcd.labels)[video_ids]

                    # compute iou
                    iou = compute_iou(concept_masks, labels)

                    print('Best concept based on query frame: {}, iou: {}'.format(best_concept, iou))
                    return

                else:
                    for concept in vcd.dic[layer][head]['concepts']:
                        # grab concept mask and video ids
                        concept_masks = vcd.dic[layer][head][concept]['video_mask']
                        video_ids = vcd.dic[layer][head][concept]['video_numbers']

                        if isinstance(concept_masks, list):
                            concept_masks = torch.stack([torch.tensor(np.load(mask)) for mask in concept_masks], dim=0)

                        # grab gt concept masks
                        labels = torch.stack(vcd.labels)[video_ids]

                        # compute iou
                        iou = compute_iou(concept_masks, labels)

                        key = 'layer_{} head_{} {}'.format(layer, head, concept)
                        results[key] = iou

        # sort results by iou
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # save results as json file
    with open(save_path, 'w') as f:
        json.dump(results, f)

    return results

def postprocess(concept_mask, video, predictor, num_points=1, mode='random',use_torch=False,
                use_highest_overlap_with_concept=True, sample_factor=8, use_last_centroid=False,
                use_box=False):
    '''
    inputs:
    mask: (B, T, H, W) torch.bool
    video: (3, T, H, W) torch.float32
    predictor: sam predictor
    :return:
    new_mask: (T, H, W) torch.bool
    '''

    new_mask = torch.zeros_like(concept_mask[0])

    # video = video.cuda()
    # iterate through images
    for frame_idx in range(concept_mask.shape[1]):
        image = (video[:, frame_idx]*255).clamp(0,255).type(torch.uint8)

        # resize
        concept_mask_frame = concept_mask[:, frame_idx]


        if use_torch:
            concept_mask_frame = torch.tensor(resize_longest_side.apply_image(concept_mask_frame.type(torch.uint8).permute(1,2,0).cpu().numpy()))

        if len(concept_mask_frame.shape) == 2:
            concept_mask_frame = concept_mask_frame.unsqueeze(0)

        # use centroid of last frame as initial point
        if use_last_centroid:
            if frame_idx == 0:
                # get points
                input_points, com_pre = sample_points(concept_mask_frame, num_points=num_points, mode=mode, sample_factor=sample_factor)
            else:
                # use centroid of last frame as initial point
                # get points
                input_points, com_post = sample_points(concept_mask_frame, num_points=num_points-1, mode=mode,sample_factor=sample_factor)

                try:
                    input_points = np.concatenate([input_points, com_pre[None]], axis=0)
                    com_pre = com_post
                except:
                    input_points, com_post = sample_points(concept_mask_frame, num_points=num_points, mode=mode,
                                                       sample_factor=sample_factor)
        else:
            # get points
            input_points, com_pre = sample_points(concept_mask_frame, num_points=num_points, mode=mode, sample_factor=sample_factor)
        if use_box:
            try:
                input_box = masks_to_boxes(concept_mask_frame)
                input_box = np.array(input_box)
            except:
                pass
        # if no points, return concept mask
        if input_points is None:
            new_mask[frame_idx] = concept_mask_frame[0]
            continue
        input_points = input_points.astype(np.int32)
        input_labels = np.array([1]*num_points)

        if use_torch:
            # resize longest side of image to 1024
            resized_image = torch.tensor(resize_longest_side.apply_image(image.permute(1,2,0).cpu().numpy())).cuda()
            predictor.set_torch_image(resized_image.permute(2,0,1).unsqueeze(0), (video.shape[-2], video.shape[-1]))
        else:
            predictor.set_image(image.permute(1, 2, 0).numpy())


        # resize concept mask to 256
        # resize_longest_side = ResizeLongestSide(256)
        # sam_mask_input = resize_longest_side.apply_image(concept_mask_frame[0].type(torch.uint8)).astype(bool)
        # sam_masks, scores, logits = predictor.predict(mask_input=sam_mask_input[None],multimask_output=False)

        if use_box:
            sam_masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box[None, :],
                multimask_output=True,
            )
        else:
            sam_masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
                # mask_input=concept_mask_frame.unsqueeze(0),
            )

        if use_highest_overlap_with_concept:
            # choose mask with highest overlap with concept mask
            best_sam_mask = sam_masks[np.argmax([compute_iou(torch.tensor(mask), concept_mask_frame[0]) for mask in sam_masks])]
        else:
            # choose mask with highest score
            best_sam_mask = sam_masks[np.argmax(scores)]

        # add to new mask
        new_mask[frame_idx] = torch.tensor(best_sam_mask).bool()


        # # visualize selected sampled points over image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image.permute(1, 2, 0).cpu().numpy())
        # # resize mask
        # # resized_sam_mask = resize_longest_side.apply_image(mask.astype(np.uint8)).astype(bool)
        # # show_mask(resized_sam_mask, plt.gca())
        # show_mask(best_sam_mask, plt.gca())
        # show_points(input_points, input_labels, plt.gca())
        # plt.title(f"Mask {np.argmax(scores) + 1}", fontsize=18)
        # plt.show()
        # print()

        # visualize sampled points over image for all masks
        # for i, (mask, score) in enumerate(zip(sam_masks, scores)):
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(image.permute(1,2,0).cpu().numpy())
        #     # resize mask
        #     # resized_sam_mask = resize_longest_side.apply_image(mask.astype(np.uint8)).astype(bool)
        #     # show_mask(resized_sam_mask, plt.gca())
        #     show_mask(mask, plt.gca())
        #     show_points(input_points, input_labels, plt.gca())
        #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        #     plt.show()
        #     print()


        # debug
        # predictor.set_image(image*255)
        # input_point = np.array([[150, 50]])
        # input_label = np.array([1])
        # masks, scores, logits = predictor.predict(
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     multimask_output=True,
        # )
        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(image.permute(1, 2, 0))
        #     show_mask(mask, plt.gca())
        #     show_points(input_point, input_label, plt.gca())
        #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        #     plt.axis('off')
        #     plt.show()


    return new_mask


def sample_points(mask, num_points=1, mode='random', mask_area=True, sample_factor=8):
    '''
    inputs:
    :param mask:
    :param num_points:
    :return: points = np.array([[500, 375], [1125, 625]]) shape (2, 2)
    '''
    # combine masks into one, where 1 is foreground and 0 is background
    mask = mask.float().sum(0).clamp(0, 1).bool()
    com = ndimage.center_of_mass(np.array(mask).astype(np.uint8))
    # sample num_points points from mask as (x, y) coordinates
    if mode == 'random':
        points = torch.nonzero(mask).float()
        points = points[torch.randperm(points.shape[0])[:num_points]]
        # if there are no non-zero points, sample the center of the image num_points times
        if points.shape[0] == 0:
            return None, None
    elif mode == 'centroid':
        if mask.sum() == 0:
            return None, None
        else:
            # calculate centroid of mask
            if mask_area:
                # get bounding box around mask in form [x0, y0, x1, y1]
                mask_box = np.array(np.where(mask)).T
                mask_box = np.concatenate([mask_box.min(0), mask_box.max(0)])
                # sample from gaussian with mean at centroid and std of 1/2 the mask size
                points = np.random.multivariate_normal(com, (mask_box[2:] - mask_box[:2] * np.eye(2)) / sample_factor, num_points)
                # check if points fall outside the mask
                # points = points[(points[:, 0] >= 0) & (points[:, 0] < mask.shape[-1]) & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[-2])]
            else:
                # sample from gaussian with mean at centroid and std of 1/8 of image size
                points = np.random.multivariate_normal(com, mask.shape[-1] / sample_factor * np.eye(2), num_points)
            # convert to torch
            points = torch.tensor(points)
    elif mode == 'center':
        points = torch.tensor([[mask.shape[-1] / 2, mask.shape[-2] / 2]])
    elif mode == 'box':
        points = torch.tensor([[0, 0], [0, mask.shape[-2]], [mask.shape[-1], 0], [mask.shape[-1], mask.shape[-2]]])
    else:
        raise NotImplementedError

    # convert to numpy and switch x and y
    points = points.numpy()
    points = points[:, [1, 0]]

    # switch x and y
    com = np.array(com)[[1, 0]]

    return points, com

def compute_iou(v1, v2):
    '''
    v1: (B, T, H, W)
    v2: (B, T, H, W) or mask
    '''

    # compute intersection and union
    intersection = torch.logical_and(v1, v2).sum()
    union = torch.logical_or(v1, v2).sum()

    # compute iou
    iou = intersection / union

    return float(iou)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def save_prediction(vcd, rgb_video, prediction, non_processed_mask, label, video_idx, object_idx, best_dict,
                    save_prefix='d17'):
    # saves: rgb_video, prediction, non_processed_mask, label

    rgb_video = rgb_video.permute(1,2,3,0)


    # create canvas of size (T, H, W*4, 3)
    canvas_video = np.zeros((rgb_video.shape[0], rgb_video.shape[1], rgb_video.shape[2]*4, 3), dtype=np.uint8)
    # add rgb video
    canvas_video[:, :, :rgb_video.shape[2], :] = (rgb_video*255).numpy().astype(np.uint8)

    # create segmentation mask for prediction
    pred_mask = np.array(prediction.float())
    # stack along last axis
    pred_mask = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)
    # where mask is 1, alpha blend rgb_video towards green in 0-1 range
    prediction_rgb = np.where(pred_mask, np.array([0, 1, 0]), rgb_video)
    # add to canvas
    canvas_video[:, :, rgb_video.shape[2]:rgb_video.shape[2]*2, :] = (prediction_rgb*255).astype(np.uint8)

    # create segmentation mask for non_processed_mask
    non_processed_mask = np.array(non_processed_mask.float())
    # stack along last axis
    non_processed_mask = np.stack([non_processed_mask, non_processed_mask, non_processed_mask], axis=-1)
    # where mask is 1, alpha blend rgb_video towards red in 0-1 range
    non_processed_mask_rgb = np.where(non_processed_mask, np.array([1, 0, 0]), rgb_video)
    # add to canvas
    canvas_video[:,:, rgb_video.shape[2]*2:rgb_video.shape[2]*3, :] = (non_processed_mask_rgb*255).astype(np.uint8)

    # create segmentation mask for label
    label = np.array(label.float())
    # stack along last axis
    label = np.stack([label, label, label], axis=-1)
    # where mask is 1, alpha blend rgb_video towards blue in 0-1 range
    label_rgb = np.where(label, np.array([0, 0, 1]), rgb_video)
    # add to canvas
    canvas_video[:,:, rgb_video.shape[2]*3:rgb_video.shape[2]*4, :] = (label_rgb*255).astype(np.uint8)

    # open folder for saving
    save_folder = os.path.join(vcd.args.save_dir, save_prefix + "_predictions")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # create directory per video
    save_folder = os.path.join(save_folder, 'video_{}'.format(video_idx))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # file name
    file_name = os.path.join(save_folder, 'object_{}_layer_{}_head_{}_concept_{}'.format(object_idx, best_dict['layer'], best_dict['head'], best_dict['concept']))
    # save video
    vcd.save_video(frames=canvas_video,
                    file_name=file_name,
                    extensions=['.mp4'], fps=6,
                    upscale_factor=1)
    return