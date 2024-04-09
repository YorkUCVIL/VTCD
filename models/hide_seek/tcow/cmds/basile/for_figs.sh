# BVH, Nov 2022.

# f0123: random, for data_sim figure, with clr augs.
for KS in 0010 0132 0148 0267 0281 0310 0323 0340 0400 0638 0885 0936 1731 1876 2205 2357 2691 3254 3294 3380
do
CUDA_VISIBLE_DEVICES=1 python train.py --name f${KS} --data_path /local/vondrick/datasets/kubcon_v10/train/kubcon_v10_scn0${KS}/ --batch_size 1 --num_workers 4 --use_data_frac 4 --num_queries 3 --num_epochs 1 --num_frames 30 --augs_2d 0 --kubric_max_delay 6 --track_map_resize bilinear --seeker_query_time 0 --avoid_wandb 2 --log_level debug
done


# fna0123: random, for data_sim figure, no clr augs.
for KS in 0010 0132 0148 0267 0281 0310 0323 0340 0400 0638 0885 0936 1731 1876 2205 2357 2691 3254 3294 3380
do
CUDA_VISIBLE_DEVICES=2 python train.py --name fna${KS} --data_path /local/vondrick/datasets/kubcon_v10/train/kubcon_v10_scn0${KS}/ --batch_size 1 --num_workers 4 --use_data_frac 4 --num_queries 3 --num_epochs 1 --num_frames 30 --do_val_aug 0 --do_val_noaug 1 --augs_2d 0 --kubric_max_delay 6 --track_map_resize bilinear --seeker_query_time 0 --avoid_wandb 2 --log_level debug
done


# fbn0123: bench, for data_sim figure, no clr augs.
for KS in 003 009 012 013 016 017 020 024 025 026 028
do
CUDA_VISIBLE_DEVICES=3 python train.py --name fbn${KS} --data_path /proj/vondrick3/basile/kubbench_v3/kubbench_v3_scn${KS}_box_push_container_slide/ --batch_size 1 --num_workers 4 --use_data_frac 4 --num_queries 3 --num_epochs 1 --num_frames 30 --do_val_aug 0 --do_val_noaug 1 --augs_2d 0 --kubric_max_delay 6 --track_map_resize bilinear --seeker_query_time 0 --avoid_wandb 2 --log_level debug
done


# f2: for data_real figure, especially dav/ytb, after more annots.
python eval/test.py --resume v113 --name fv113_fy3 --gpu_id 6 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017/train_dancing \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --extra_visuals 1 --avoid_wandb 2
