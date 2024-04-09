# BVH, Oct 2022.

cdb3
cd hide-seek
conda activate bcv11

# ba8 (ablation): no training at all (via tiny use_data_frac), use PRE_YTB_DAV directly, num_queries 2 to 3, seeker_query_time 0.05 to 0, on cv10.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=5 python train.py --name ba8 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.01 --batch_size 5 --num_workers 24 --num_queries 3 --checkpoint_every 20 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0 --aot_max_gap 4 --aot_arch aotb --aot_pretrain_path pretrained/AOTB_PRE_YTB_DAV.pth --xray_query 0 --annot_visible_pxl_only 1

# ba9: relative to ba7, num_queries 2 to 3, seeker_query_time 0.05 to 0, re-enable xray_query, re-disable annot_visible_pxl_only, on cv13.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=7 python train.py --name ba9 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 1 --batch_size 5 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0 --aot_max_gap 4 --aot_arch aotb --aot_pretrain_path pretrained/AOTB_PRE_YTB_DAV.pth --xray_query 1 --annot_visible_pxl_only 0 --resume ba9

# ba10: pretrained PRE_YTB_DAV to PRE (cleaner for comparison), on cv12.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=2 python train.py --name ba10 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 1 --batch_size 6 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0 --aot_max_gap 4 --aot_arch aotb --aot_pretrain_path pretrained/AOTB_PRE.pth --xray_query 1 --annot_visible_pxl_only 0 --avoid_wandb 1 --log_rarely 1

# ba11 (visible ablation): disable xray_query, enable annot_visible_pxl_only, on cv12.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=1 python train.py --name ba11 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 1 --batch_size 6 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0 --aot_max_gap 4 --aot_arch aotb --aot_pretrain_path pretrained/AOTB_PRE.pth --xray_query 0 --annot_visible_pxl_only 1 --avoid_wandb 1 --log_rarely 1

# ba12 (cartoon ablation): relative to ba10, kubric_degrade randclr, on cv13.
# => X
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6 python train.py --name ba12 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 1 --batch_size 5 --num_workers 24 --num_queries 3 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0 --aot_max_gap 4 --aot_arch aotb --aot_pretrain_path pretrained/AOTB_PRE.pth --xray_query 1 --annot_visible_pxl_only 0 --kubric_degrade randclr --avoid_wandb 1 --log_rarely 1 --resume ba12
