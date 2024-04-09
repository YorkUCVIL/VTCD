# BVH, Oct 2022.

# on cv11+:
cdb3
cd hide-seek
conda activate bcv11

# Test-time hyperparams I wanna vary: annots_must_exist, also_grayscale, measure_baselines.
# Be careful when judging XX.
# Group of consistent (num_frames, query_time):
# - X
# - X
# Changes from t2 to t3:
# - Enable center_crop (default) for plugin videos.
# - Upgrade test data loading mechanism to avoid overloading servers, also keep num_workers low.
# Changes from b3 to b4:
# - num_queries 4 to 1 to ensure metrics are determined by object of interest only.
# Intended table (outdated):
# - Ours causal = v104 (for now, later v110 with consistent occl_cont_zero_weight)
# - Ours non-causal = v109 (but maybe unused)
# - Kubric cartoon = v93 (for now, TODO newer hyperparams)
# - AOT trained = ba9
# - AOT direct = ba8
# - Perfect visible snitch only (sim only)
# - Static propagation (sim only)
# - Linear extrapolation (sim only)
# - Jump nearest object (sim only)


# =====================
# ======== SIM ========
# =====================

# (1) Kubric random test (_t).

for MV in ba9 # v92 v93 v94 v95 v96 v97 v104 v105 v106 v107 v108 v109 ba6 ba7 ba8 ba9 # v98 # 
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_t3 --gpu_id 2 --data_path \
/local/vondrick/datasets/kubcon_v10/ \
--use_data_frac 0.5 --num_workers 8 --num_queries 4
done

# # (2) Kubric curated benchmark (_b) (30 scenes).

# for MV in v92 v93 v94 v95 v96 v97 v104 v105 v106 v107 v108 v109 ba6 ba7 ba8 ba9 # v98 # 
# do
# ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_b3 --gpu_id 2 --data_path \
# /proj/vondrick3/basile/kubbench_v3/ \
# --use_data_frac 1.0 --num_workers 6 --num_queries 4
# done
# # ^ => X

# (2) Kubric curated benchmark (_b) (30 scenes).

for MV in v92 v93 v94 v95 v96 v97 v98 v104 v105 v106 v107 v108 v109 ba6 ba7 ba8 ba9 # 
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_b4 --gpu_id 2 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1
done


# ======================
# ======== REAL ========
# ======================

# (3) Real benchmark (_r).

for MV in ba9 # v92 v93 v94 v95 v96 v97 v104 v105 v106 v107 v108 v109 ba6 ba7 ba8 ba9 # v98 # 
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_r3 --gpu_id 3 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_8.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done

# (4) DeepMind perception test (_d).

for MV in v92 v93 v94 v95 v96 v97 v104 v105 v106 v107 v108 v109 ba6 ba7 ba8 ba9 # v98 # 
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_d3 --gpu_id 3 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done

# (5) Davis + YouTube-VOS / in the wild (_y).

for MV in v92 v93 v94 v95 v96 v97 v104 v105 v106 v107 v108 v109 ba6 ba7 ba8 ba9 # v98 # 
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_y3 --gpu_id 3 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_ytvos.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done
 

# Now, download results to local PC.

# ===========================
# ======== REPRESENT ========
# ===========================

# See launch.json.


# =============================
# ======== CONCLUSIONS ========
# =============================

# v94 stubbornly has very good numbers across the board!
# cls_token barely matters.
# v108 at epoch 16 has surprisingly good cont and occl numbers.


