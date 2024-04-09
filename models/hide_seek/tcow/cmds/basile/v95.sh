# BVH, Oct 2022.

cdb3
cd hide-seek
conda activate bcv11

# NOTE: hyperparams I wanna vary: which pretrained url, aot_loss, attention_type, patch_size, norm_embeddings, track_map_stride, causal_attention, drop_path_rate, network_depth, mixed data ratios, xray_query, annot_visible_pxl_only.

# v95 (visible ablation): enable annot_visible_pxl_only, on cv12.
# => better curves, but probably because metrics measure different things.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0,1,2 python train.py --name v95 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6 --annot_visible_pxl_only 1

# v96: relative to v94, mix kubric + ytvos after critical occl_risk bugfix, num_queries 3 to 2, on cv11.
# => frequently gets stuck during training, so have to set batch_size 1?
# => something mysteriously went wrong when restarting after epoch 58, so we have to use --epoch 58 when testing.
# => much worse test sim numbers, though sometimes better test real numbers, than v94.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=1 python train.py --name v96 --data_path /local/vondrick/datasets/kubcon_v10/ /local/vondrick/datasets/YouTube-VOS/ --use_data_frac 0.7 --batch_size 1 --num_workers 40 --num_queries 2 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6 --xray_query 0 --resume v96

# v97: relative to v94, occl_cont_zero_weight 0.02 to 0.06, on cv13.
# => frequently gets stuck during training, so have to set batch_size 1?
# => has nice train curves, best val snitch_during_occl_iou @ epoch 50.
# => better test sim numbers, and sometimes better test real numbers, than v96.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0 python train.py --name v97 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 1 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --occl_cont_zero_weight 0.06 --aot_loss 0.6

# v98: aot_loss 0.6 to 1.0, on cv13.
# => actually slightly worse val numbers, yet slightly better test numbers all across than v97.
# => test numbers seem slightly worse than v94 overall, BUT qualitatively, _y looks better here! (fewer false OC positives)
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6,7 python train.py --name v98 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --occl_cont_zero_weight 0.06 --aot_loss 1.0 --resume v98

# # v99 (attempted ablation): relative to v94, enable causal_attention (to avoid containment foresight), on cv12.
# # => IGNORE; halted because not actually online due to cls_token bug! later fixed.
# ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=3,4,5 python train.py --name v99 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6

# # v100 (ablation): fixed cls_token bug for causal_attention, on cv12.
# # => much lower GPU RAM usage! so can increase batch_size.
# # => I thought last cls_token still allows for leakage in theory, but the pre-query frame predictions look exactly the same, so we can keep using causal_attention = 1.
# # => IGNORE; very bad accuracy because of blocks[2:3] bug.
# ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0,1,2 python train.py --name v100 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 12 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6

# # v101 (ablation): seeker_query_time 0.05 to 0 to avoid wasting energy on priors (frame before query), on cv12.
# # => IGNORE; very bad accuracy because of blocks[2:3] bug.
# ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=3,4,5 python train.py --name v101 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 12 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --aot_loss 0.6

# # v102 (ablation): relative to v100, reduce batch_size (may be the reason behind bad performance?), on cv14.
# # => IGNORE; very bad accuracy because of blocks[2:3] bug.
# ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6 python train.py --name v102 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6

# # v103 (ablation): relative to v101, reduce batch_size (may be the reason behind bad performance?), on cv14.
# # => IGNORE; very bad accuracy because of blocks[2:3] bug.
# ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=7 python train.py --name v103 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --aot_loss 0.6
