
    for b in range(B):
        output_xyz = model_retval['output_traject'][b, ..., 0:3].detach().cpu().numpy()  # (Q, T, 3).
        target_uvd = model_retval['target_traject'][b, ..., 0:3].detach().cpu().numpy()  # (Q, T, 3).
        target_xyz = model_retval['target_traject'][b, ..., 3:6].detach().cpu().numpy()  # (Q, T, 3).

        eucl_all = np.linalg.norm(output_xyz - target_xyz, ord=2, axis=-1)  # (Q, T).
        eucl_oof = []
        eucl_occ = []
        eucl_vis = []
        eucl_moving = []

        count_all = Q * T
        count_oof = 0
        count_occ = 0
        count_vis = 0
        count_moving = 0

        for q in range(Q):

            if np.linalg.norm(target_xyz[q, 0] - target_xyz[q, -1], ord=2, axis=-1) >= 0.01:
                eucl_moving.append(eucl_all[q, :])
                count_moving += T

            for t in range(T):
                # TODO DRY (update & use data loader)
                u_idx = np.floor(target_uvd[q, t, 0] * W).astype(np.int32)
                v_idx = np.floor(target_uvd[q, t, 1] * H).astype(np.int32)
                out_of_frame = (u_idx < 0 or u_idx >= W or v_idx < 0 or v_idx >= H)

                if out_of_frame:
                    # video_depth = -999999.0
                    eucl_oof.append(eucl_all[q, t])
                    count_oof += 1

                else:
                    video_depth = data_retval['kubric_retval']['pv_depth_tf'][b, 0, t, v_idx, u_idx]
                    point_depth = target_uvd[q, t, 2]

                    if video_depth + 0.01 <= point_depth:
                        eucl_occ.append(eucl_all[q, t])
                        count_occ += 1

                    else:
                        eucl_vis.append(eucl_all[q, t])
                        count_vis += 1

        eucl_oof = np.array(eucl_oof)
        eucl_occ = np.array(eucl_occ)
        eucl_vis = np.array(eucl_vis)
        eucl_moving = np.array(eucl_moving)

        metric_retval['eucl_all'].append(eucl_all.mean())
        metric_retval['eucl_oof'].append(eucl_oof.mean() if count_oof != 0 else np.nan)
        metric_retval['eucl_occ'].append(eucl_occ.mean() if count_occ != 0 else np.nan)
        metric_retval['eucl_vis'].append(eucl_vis.mean() if count_vis != 0 else np.nan)
        metric_retval['eucl_moving'].append(eucl_moving.mean() if count_moving != 0 else np.nan)
        
        metric_retval['count_all'].append(count_all)
        metric_retval['count_oof'].append(count_oof)
        metric_retval['count_occ'].append(count_occ)
        metric_retval['count_vis'].append(count_vis)
        metric_retval['count_moving'].append(count_moving)

    Report per-metric mean across examples within this batch, excluding invalid ones with zero
    data points for a particular metric.
    for key in metric_retval.keys():
        metric_retval[key] = np.array(metric_retval[key])
        if not('count' in key):
            values = [value for value in metric_retval[key] if not(np.isnan(value))]
            if len(values) != 0:
                logger.report_scalar(key, np.array(values).mean())
