

def get_thing_occl_cont_pointers(occl_fracs, pv_div_segm, metadata, frame_inds, visibility_thres=0.05):
    '''
    :param occl_fracs (K, T, 3) array of float32 with (f, v, t).
    :param pv_div_segm (T, Hf, Wf, K) array of uint8 in [0, 1].
    :param metadata (dict).
    :param frame_inds (list of int).
    :return occl_cont_data (K, T, 4) array of int32.
    '''
    (T, Hf, Wf, K) = pv_div_segm.shape
    assert T == len(frame_inds)
    assert K == metadata['scene']['num_valo_instances']  # Always <= num_instances!

    # Per object and per frame: 4 values (IO, OB, IC, CB) where:
    # IO = I occlude: -2 = nothing, >= 0 = instance ID.
    # OB = occluded by: -2 = nothing, >= 0 = instance ID.
    # IC = I contain: -2 = nothing, >= 0 = instance ID.
    # CB = contained by: -2 = nothing, >= 0 = instance ID.
    # NOTE: Some IDs may be >= K, i.e. when the occludees or containees are never visible.
    occl_cont_data = np.ones((K, T, 4), dtype=np.int32) * (-2)

    for k in range(K):
        for f, t in enumerate(frame_inds):
            
            # TODX: This part will not handle recursive occlusion correctly, because the real order
            # of objects is currently impossible to always get right.

            if occl_fracs[k, f, 0] >= 1.0 - visibility_thres:
                # Check which other instance has the maximum recall with respect to this mask.
                # NOTE: In some rare cases, it may not be solely responsible for occluding me to the
                # degree necessary to reach the threshold, but we will still mark it as the primary
                # occluder.
                num_total_pxl = int(np.round(occl_fracs[k, f, 2] * Hf * Wf))
                recalls = []
                for l in range(K):
                    if l != k:
                        num_overlap_pxl = np.sum(np.logical_and(
                            pv_div_segm[f, ..., k], pv_div_segm[f, ..., l]))
                        recall = num_overlap_pxl / num_total_pxl
                    else:
                        recall = -1.0
                    recalls.append(recall)
                
                occluded_by = np.argmax(recalls)
                occl_cont_data[k, f, 1] = occluded_by
                occl_cont_data[occluded_by, f, 0] = k

            # TODX: This part does not yet handle recursive containment correctly.

            my_bbox_3d = metadata['instances'][k]['bboxes_3d'][t]
            for l in range(K):
                if l != k:
                    other_bbox_3d = metadata['instances'][l]['bboxes_3d'][t]
                    # TODX finish containment

    return occl_cont_data

