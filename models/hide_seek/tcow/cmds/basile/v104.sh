# BVH, Oct 2022.

cdb3
cd hide-seek
conda activate bcv11

# NOTE: hyperparams I wanna vary: which pretrained url, aot_loss, attention_type, patch_size, norm_embeddings, causal_attention, drop_path_rate, network_depth, mixed data ratios, xray_query, annot_visible_pxl_only.

# v104 (maybe ablation): after blocks[2:3] bugfix, causal_attention remains 1, match v98 (occl_cont_zero_weight 0.02 to 0.06), aot_loss 0.6 to 0.8, on cv12.
# => very similar test numbers as v98 (but maybe conflated with aot_loss).
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0,1 python train.py --name v104 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.06 --aot_loss 0.8

# v105 (maybe ablation): causal_attention 1 to 2 (no cls_token), on cv12.
# => essentially same test numbers as v104.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=2,3 python train.py --name v105 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 2 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.06 --aot_loss 0.8

# v106: causal_attention 2 to -1 (offline again, but still no cls_token), seeker_query_time 0 to 0.05, on cv12.
# => essentially same test numbers as v98 (apart from cls_token, only difference is aot_loss).
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=4,5 python train.py --name v106 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention -1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --occl_cont_zero_weight 0.06 --aot_loss 0.8

# v107: restore causal_attention -1 to 0, on cv12.
# => now, considering aot_loss 0.8, params are exactly inbetween v97 and v98.
# => essentially same test numbers as v106.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6,7 python train.py --name v107 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --occl_cont_zero_weight 0.06 --aot_loss 0.8

# v108: augs_version 4 to 5 (default; less grayscale), occl_cont_zero_weight 0.06 to 0.04, on cv12.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=4,5 python train.py --name v108 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --occl_cont_zero_weight 0.04 --aot_loss 0.8 --avoid_wandb 1 --log_rarely 1 --log_level debug --resume v108

# v109: seeker_query_time 0.05 to 0 for fair comparison with baselines & causal ablations, on cv14.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=4,5 python train.py --name v109 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.04 --aot_loss 0.8 --avoid_wandb 1 --log_rarely 1 --log_level debug --resume v109

# v110 (maybe ablation): causal_attention 0 to 1 to be consistent with latest hyperparams, on cv14.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6,7 python train.py --name v110 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.04 --aot_loss 0.8 --avoid_wandb 1 --log_rarely 1 --log_level debug

# v111 (maybe ablation): occl_cont_zero_weight 0.04 to 0.02 (to try to match v94 numbers), on cv13.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6,7 python train.py --name v111 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.02 --aot_loss 0.8 --avoid_wandb 1 --log_rarely 1 --log_level debug

# v112: restore causal_attention 1 to 0 (to try to match v94 numbers), on cv12.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0,3 python train.py --name v112 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.02 --aot_loss 0.8 --avoid_wandb 1 --log_rarely 1 --log_level debug
