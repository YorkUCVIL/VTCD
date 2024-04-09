# BVH, Oct 2022.

cdb3
cd hide-seek
conda activate bcv11

# NOTE: hyperparams I wanna vary: which pretrained url, aot_loss, attention_type, patch_size, norm_embeddings, track_map_stride, causal_attention, drop_path_rate, network_depth, mixed data ratios, xray_query.

# v86: relative to v84, re-enable drop_path_rate, seeker_query_time 0.15 to 0.1, on cv12.
# => some sim2real results are somehow much better than v84.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0,1,2 python train.py --name v86 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 6 --num_workers 24 --num_queries 2 --num_frames 22 --track_map_resize bilinear --seeker_query_time 0.1 --aot_loss 0.8

# v87: num_frames 22 to 32, on cv14.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0,1,2 python train.py --name v87 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 2 --num_frames 32 --track_map_resize bilinear --seeker_query_time 0.1 --aot_loss 0.8

# v88: num_queries 2 to 3, on cv14.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=3,4,5 python train.py --name v88 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 3 --num_frames 32 --track_map_resize bilinear --seeker_query_time 0.1 --aot_loss 0.8

# v89: relative to v86, after desirability changes (encourage mostly visible query), after wrong occl_mask bugfix, num_frames 22 to 26, enable kubric_max_delay (0 to 6), enable kubric_reverse_prob (0 to 0.1), enable kubric_palindrome_prob (0 to 0.1), seeker_query_time 0.1 to 0, disable voc_cls_lw (0.01 to 0), on cv12.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=0,1,2 python train.py --name v89 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 2 --num_frames 26 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --aot_loss 0.8

# v90: mix kubric + ytvos, aot_loss 0.8 to 0 in attempt to mitigate instability, on cv12.
# => still very unstable, later fixed.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=3,4,5 python train.py --name v90 --data_path /local/vondrick/datasets/kubcon_v10/ /local/vondrick/datasets/YouTube-VOS/ --batch_size 3 --num_workers 36 --num_queries 2 --num_frames 26 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0 --voc_cls_lw 0 --aot_loss 0

# v91: relative to v89, seeker_query_time 0.0 to 0.05, aot_loss 0.8 to 0.6 on cv12.
# => allows more real videos, so hard to compare.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=3,4,5 python train.py --name v91 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 3 --num_workers 24 --num_queries 2 --num_frames 26 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6

# v92: num_queries 2 to 3, num_frames 26 to 30, on cv14.
# => similar performance as v91, though also hard to compare.
# => much better than v91 on real benchmarks, especially deepmind.
# => when compared with deepmind to all until v99, this one is best for occl & cont (second is v94).
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6,7 python train.py --name v92 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6

# v93 (ablation): augs_version 3 to 4 (default), degrade randclr, on cv12.
# => within its own cartoon domain, even better numbers than all other runs.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6,7 python train.py --name v93 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6 --xray_query 0 --kubric_degrade randclr --resume v93

# v94: relative to v92, augs_version 3 to 4 with random grayscale (default), on cv11.
# => really nice curves, best val snitch_during_occl_iou @ epoch 54.
# => when compared with deepmind to all until v99, this one is second best for occl & cont (next to v92).
# => most test results are slightly worse than v92, which suggests that augs_version update was not an upgrade -- later, see augs_version 5.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6,7 python train.py --name v94 --data_path /local/vondrick/datasets/kubcon_v10/ --batch_size 2 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --kubric_palindrome_prob 0.1 --track_map_resize bilinear --seeker_query_time 0.05 --voc_cls_lw 0 --aot_loss 0.6 --xray_query 0
