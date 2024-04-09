# BVH, Oct 2022.

cdb3
cd hide-seek
conda activate bcv11

# ba1: relative to v72, try AOT baseline, on cv10.
# => output_mask zero bugfix around epoch 40.
# => test time prediction always static (given query, copies it)?
WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=1 python train.py --name ba1 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.5 --batch_size 2 --num_workers 32 --num_queries 2 --num_frames 22 --tracker_arch aot --seeker_query_time 0.15 --resume ba1

# ba2: frame_stride 1 to 2 to try to mitigate static prediction.
# => still copies query :(
WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=1 python train.py --name ba2 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.5 --batch_size 2 --num_workers 32 --num_queries 2 --num_frames 18 --kubric_frame_stride 2 --tracker_arch aot --seeker_query_time 0.15

# ba3: frame_stride 2 to 3 to try to mitigate static prediction.
# => still copies query :(
WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=3 python train.py --name ba3 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.5 --batch_size 2 --num_workers 32 --num_queries 2 --num_frames 12 --kubric_frame_stride 3 --tracker_arch aot --seeker_query_time 0.15

# ba4: relative to ba1, align with v94 (many changes, but disable palindrome), aot_max_gap 1 to 4, enable xray_query (was default before for baselines), on cv13.
# => still mostly static predictions :(
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=6 python train.py --name ba4 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.5 --batch_size 6 --num_workers 24 --num_queries 2 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0.05 --aot_max_gap 4 --xray_query 1

# ba5 (visible ablation): disable xray_query, enable annot_visible_pxl_only, on cv13.
# => still mostly static predictions :(
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=7 python train.py --name ba5 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.5 --batch_size 6 --num_workers 24 --num_queries 2 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0.05 --aot_max_gap 4 --xray_query 0 --annot_visible_pxl_only 1

# ba6 (visible ablation): set pretrain_path to PRE (it was nothing before), on cv11.
# => finally works! breaks down during occlusion, as expected.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=2 python train.py --name ba6 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.7 --batch_size 8 --num_workers 24 --num_queries 2 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0.05 --aot_max_gap 4 --aot_arch aotb --aot_pretrain_path pretrained/AOTB_PRE.pth --xray_query 0 --annot_visible_pxl_only 1

# ba7 (visible ablation): set pretrain_path to PRE_YTB_DAV, on cv11.
# => slightly better test results on average than ba6.
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 CUDA_VISIBLE_DEVICES=3 python train.py --name ba7 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.7 --batch_size 8 --num_workers 24 --num_queries 2 --num_frames 30 --kubric_max_delay 6 --kubric_reverse_prob 0.1 --tracker_arch aot --track_map_resize bilinear --seeker_query_time 0.05 --aot_max_gap 4 --aot_arch aotb --aot_pretrain_path pretrained/AOTB_PRE_YTB_DAV.pth --xray_query 0 --annot_visible_pxl_only 1
