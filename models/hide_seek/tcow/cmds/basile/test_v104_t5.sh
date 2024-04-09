# BVH, Oct 2022.

# on cv11+:
cdb3
cd hide-seek
conda activate bcv11

# NOTE: This is strictly nf=30, qt=0 for consistent evaluation.
# Intended table: see model_notes.txt.
# Changes from t3/4 to t5:
# - Ignore all models with seeker_query_time > 0 (v<104, v106, v107, v108, ba<8).

# - TODO LATER Select epoch ~52 instead of latest.
# - TODO LATER test-time baselines.
# - TODO LATER bigger real test sets.

# =====================
# ======== SIM ========
# =====================

# (1) Kubric random test (_t).

for MV in v104 v105 v109 v110 v111 v112 v113 v114 ba8 ba9 ba10 ba11 ba12
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_t5 --gpu_id 7 --data_path \
/local/vondrick/datasets/kubcon_v10/ \
--use_data_frac 0.4 --num_workers 8 --num_queries 4
done

# (2) Kubric curated benchmark (_b) (30 scenes).

for MV in v104 v105 v109 v110 v111 v112 v113 v114 ba8 ba9 ba10 ba11 ba12
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_b5 --gpu_id 7 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1
done


# ======================
# ======== REAL ========
# ======================

# (3) Real benchmark (_r).

for MV in v104 v105 v109 v110 v111 v112 v113 v114 ba8 ba9 ba10 ba11 ba12
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_r5 --gpu_id 7 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_8.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done

# (4) DeepMind perception test (_d).

for MV in v104 v105 v109 v110 v111 v112 v113 v114 ba8 ba9 ba10 ba11 ba12
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_d5 --gpu_id 7 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1
done

# (5) Davis + YouTube-VOS / in the wild (_y).

for MV in v104 v105 v109 v110 v111 v112 v113 v114 ba8 ba9 ba10 ba11 ba12
do
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume ${MV} --name ${MV}_y5 --gpu_id 7 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
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



