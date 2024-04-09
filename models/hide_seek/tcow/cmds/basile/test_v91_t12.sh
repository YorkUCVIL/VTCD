# BVH, Oct 2022.

# on cv11+:
cdb3
cd hide-seek
conda activate bcv11

# Test-time hyperparams I wanna vary: annots_must_exist, also_grayscale, measure_baselines.
# Include all ablations (v93, v95) & baselines (ba6, ba7) in for loops too.
# Largest group of consistent (num_frames, query_time): v92, v93, v94, v95, v96, v97, v98, v99, etc.
# Which means, be careful with judging v91!


# =====================
# ======== SIM ========
# =====================

# (1) Kubric random test (_t).

for MV in v93 # v91 v92 v94 v95 v96 v97 v98 v99 ba6 ba7
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_t2 --gpu_id 4 --data_path /local/vondrick/datasets/kubcon_v10/ --use_data_frac 0.5 --num_queries 4
done
# ^ => best numbers: v92, v94, v98.

# (2) Kubric curated benchmark (_b) (30 scenes).

for MV in v93 # v91 v92 v94 v95 v96 v97 v98 v99 ba6 ba7
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_b2 --gpu_id 5 --data_path /proj/vondrick3/basile/kubbench_v3/ --use_data_frac 1.0 --num_queries 4
done
# ^ => best numbers: v91, v92, v94.


# ======================
# ======== REAL ========
# ======================

# (3) Real benchmark (_r).

for MV in ba6 ba7 # v91 v92 v93 v94 v95 v96 v97 v98 v99
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_r2 --gpu_id 4 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_5.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_7.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done
# ^ => best numbers: v92, v94, v96, though kinda random and not consistent.

# (4) DeepMind perception test (_d).

for MV in v91 v92 v93 v94 v95 v99 # v96 v97 v98 ba6 ba7 # 
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_d2 --gpu_id 4 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done
# ^ => best numbers: v92, v94 (by far!)

# (5) YouTube-VOS / in the wild (_y) (only 18 videos so far).

for MV in ba6 ba7 # v91 v92 v93 v94 v95 v96 v97 v98 v99
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_y2 --gpu_id 4 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done
# ^ => best numbers: v92, v95 (surprisingly), v96 (not surprising, but only good for snitch, may actually be bad for occl, which is also logical).
 



# =============================
# ======== CONCLUSIONS ========
# =============================

# prefer high num_queries (3 is better than 2).
# prefer low occl_cont_zero_weight (0.02 is better than 0.06), but this may be because non-OC frames are not included at all in any OC metric.
# aot_loss seems to have goldilocks zone (0.6 is better than 1, though not consistently, but perhaps yet another value is optimal).
# mixed sim + real training (i.e. v96, which has ytvos) is hard to evaluate because conflated with num_queries, but seems generally worse, both on sim & real tests.

