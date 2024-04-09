# BVH, Oct 2022.

cdb3
cd hide-seek
conda activate bcv11

# NOTE: I cancelled these early to start training our newest ablations:
# - v108, v109, v110.

# v113 (visible ablation): relative to v111, keep causal_attention 1, enable annot_visible_pxl_only, on cv12.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=4,5 python train.py --name v113 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.02 --aot_loss 0.8 --avoid_wandb 1 --log_rarely 1 --log_level debug --annot_visible_pxl_only 1

# v114 (cartoon ablation): disable annot_visible_pxl_only, kubric_degrade none to randclr, on cv14.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=1,2 python train.py --name v114 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --causal_attention 1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --occl_cont_zero_weight 0.02 --aot_loss 0.8 --avoid_wandb 1 --log_rarely 1 --log_level debug --kubric_degrade randclr --resume v114
