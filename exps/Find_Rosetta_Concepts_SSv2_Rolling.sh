echo "########## Finding Rosetta Concepts for SSv2 class Rolling something on a flat surface ##########"
echo "#################################################################################################"

echo "Running VTCD for TCOW on SSv2"
python run_vcd.py --exp_name Occ_SSv2_Rolling --attn_head 0 1 2 3 4 5 6 7 8 9 10 11 --target_class 'Rolling something on a flat surface' --max_num_videos 29 --model timesformer_occ --dataset ssv2 --cluster_layer 0 1 2 3 4 5 6 7 8 9 10 11  --slic_compactness 0.01 --force_reload_videos
echo "Running CRIS for TCOW on SSv2"
python CRIS.py --exp_name Occ_SSv2_Rolling --num_masks 8000 --heads_removed_each_step 100 --masking_ratio 0.5

echo "Running VTCD for VideoMAE PRE on SSv2"
python run_vcd.py --exp_name VideoMAE_Pretrained_SSv2_Rolling --target_class 'Rolling something on a flat surface' --attn_head 0 1 2 3 4 5 6 7 8 9 10 11  --max_num_videos 29 --model vidmae_ssv2_pre --dataset ssv2 --cluster_layer 0 1 2 3 4 5 6 7 8 9 10 11  --slic_compactness 0.1 --force_reload_videos
echo "Running CRIS for VideoMAE PRE on SSv2"
python CRIS.py --exp_name VideoMAE_Pretrained_SSv2_Rolling --num_masks 4000 --heads_removed_each_step 100 --masking_ratio 0.95

echo "Running VTCD for Intern on SSv2"
python run_vcd.py --exp_name Intern_SSv2_Rolling --attn_head 0 1 2 3 4 5 6 7 8 9 10 11 --target_class 'Rolling something on a flat surface' --max_num_videos 29 --model intern --dataset ssv2 --cluster_layer 0 1 2 3 4 5 6 7 8 9 10 11 --slic_compactness 0.15  --force_reload_videos
echo "Running CRIS for Intern on SSv2"
python CRIS.py --exp_name Intern_SSv2_Rolling --num_masks 4000 --heads_removed_each_step 100 --masking_ratio 0.5

echo "Running VTCD for VideoMAE FT on SSv2"
python run_vcd.py --exp_name VideoMAE_FT_SSv2_Rolling --target_class 'Rolling something on a flat surface' --attn_head 0 1 2 3 4 5 6 7 8 9 10 11  --max_num_videos 29 --model vidmae_ssv2_ft --dataset ssv2 --cluster_layer 0 1 2 3 4 5 6 7 8 9 10 11  --slic_compactness 0.1 --force_reload_videos
echo "Running CRIS for VideoMAE FT on SSv2"
python CRIS.py --exp_name VideoMAE_FT_SSv2_Rolling --num_masks 4000 --heads_removed_each_step 100 --masking_ratio 0.5


echo "########## Finding Rosetta Concepts for SSv2 class Rolling something on a flat surface ##########"
echo "#################################################################################################"
echo "Running find_rosetta_concepts.py and saving videos to results/Rosetta"
python find_rosetta_concepts.py --exp_name Occ_SSv2_Rolling VideoMAE_Pretrained_SSv2_Rolling Intern_SSv2_Rolling VideoMAE_FT_SSv2_Rolling --importance_file_name ConceptImportance_4000Masks.pkl \
 --frac_important_concepts_keep 0.075 --rosetta_iou_thresholds 0.2 0.2 0.2 --save_rosetta_videos