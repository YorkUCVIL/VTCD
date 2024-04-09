# BVH, Oct 2022.

# on cv<=10:
cdb3
cd hide-seek
conda activate base

# NOTE: This is strictly nf=30, qt=0 for consistent evaluation.
# Changes from t8 to t9:
# - Improved tgt visualizations.
# - AOT test forward query bugfix & aot_eval_size_fix only for ba8 (no training).
# Changes from t9 to t10:
# - Also measure snitch_during_vis metrics.
# - Adjusted tgt vis again.
# - Add perfect baselines cross metrics.
# - Updated annotation of ytvos video (that I forgot in v2 guide).

# - TODO LATER Select epoch ~52 instead of latest.

# =====================
# ======== SIM ========
# =====================

# (1) Kubric random test (_t).

for MV in v111 v112 v113 v114 ba10 ba11 ba12
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_t10 --gpu_id 2 --data_path \
/proj/vondrick3/datasets/kubcon_v10/ \
--use_data_frac 0.5 --num_workers 8 --num_queries 4 --avoid_wandb 2
done

# (2) Kubric curated benchmark (_b) (30 scenes).

for MV in v111 v112 v113 v114 ba10 ba11 ba12
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_b10 --gpu_id 3 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1 --avoid_wandb 2
done


# ======================
# ======== REAL ========
# ======================

# (3) Real benchmark (_r).

for MV in v111 v112 v113 v114 ba10 ba11 ba12
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_r10 --gpu_id 4 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_8.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --avoid_wandb 2
done

# (4) DeepMind perception test (_d).

for MV in v111 v112 v113 v114 ba10 ba11 ba12
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_d10 --gpu_id 5 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --avoid_wandb 2
done

# (5) Davis + YouTube-VOS / in the wild (_y).

for MV in v111 v112 v113 v114 ba10 ba11 ba12
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_y10 --gpu_id 6 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_skip.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --avoid_wandb 2
done


# =====================
# ======== AOT ========
# =====================

# NOTE: For ba8 only, --aot_eval_size_fix 1 here and 0 everywhere else.

# sim (1) + (2) + real (3) + (4) + (5)

for MV in ba8
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_t10 --gpu_id 7 --data_path \
/proj/vondrick3/datasets/kubcon_v10/ \
--use_data_frac 0.5 --num_workers 8 --num_queries 4 --aot_eval_size_fix 1 --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_b10 --gpu_id 7 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_r10 --gpu_id 7 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_8.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_d10 --gpu_id 7 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_y10 --gpu_id 7 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_skip.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
done


# =========================
# ======== PERFECT ========
# =========================

# sim (1) + (2) - all

for MV in v111
do
for PB in query static linear jump
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_t10 --gpu_id 0 --data_path \
/proj/vondrick3/datasets/kubcon_v10/ \
--use_data_frac 0.5 --num_workers 8 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_b10 --gpu_id 0 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
done
done

# real (3) + (4) + (5) - query only

for MV in v111
do
for PB in query
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_r10 --gpu_id 0 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_8.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_d10 --gpu_id 0 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_y10 --gpu_id 0 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_skip.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
done
done



# Now, download results to local PC.

# ===========================
# ======== REPRESENT ========
# ===========================

# See launch.json.


# =============================
# ======== CONCLUSIONS ========
# =============================



