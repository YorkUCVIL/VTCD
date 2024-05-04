echo "Running VTCD for TCOW on Kubric"
python run_vcd.py --exp_name Occ_Kubric --attn_head 0 1 2 3 4 5 6 7 8 9 10 11 --max_num_videos 30 --model timesformer_occ --dataset kubric --cluster_layer 0 1 2 3 4 5 6 7 8 9 10 11 --cluster_subject keys --inter_cluster_method cnmf_elbow --intra_cluster_method slic --inter_max_cluster 15 --slic_compactness 0.01  --n_segments 10 --spatial_resize_factor 0.5 --inter_elbow_threshold 0.95
python CRIS.py --exp_name Occ_Kubric --num_masks 8000 --heads_removed_each_step 100 --masking_ratio 0.5
