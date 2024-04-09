# BVH, Oct 2022.

# on cv11+:
cdb3
cd hide-seek
conda activate bcv11

# NOTE: This is strictly nf=30, qt=0 for consistent evaluation.
# Changes from t5 to t6:
# - Follow model_notes_v3.txt.
# - Enable aot_eval_size_fix for ba8.
# - Test-time baselines implemented.
# Changes from t6 to t7:
# - Adjusted test visuals folder structure.
# - All test-time baselines implemented.
# Changes from t7 to t8:
# - Updated rubric & deepmind & davis & ytvos & real.

# - TODO LATER Select epoch ~52 instead of latest.

# =====================
# ======== SIM ========
# =====================

# (1) Kubric random test (_t).

for MV in v112 v113 v114 ba8 ba10 ba11 ba12 # v111 # 
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_t8 --gpu_id 2 --data_path \
/proj/vondrick3/datasets/kubcon_v10/ \
--use_data_frac 0.5 --num_workers 8 --num_queries 4 --aot_eval_size_fix 1 --avoid_wandb 2
done

# (2) Kubric curated benchmark (_b) (30 scenes).

for MV in v112 v113 v114 ba8 ba10 ba11 ba12 # v111 # 
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_b8 --gpu_id 3 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
done


# ======================
# ======== REAL ========
# ======================

# (3) Real benchmark (_r).

for MV in v112 v113 v114 ba8 ba10 ba11 ba12 # v111 # 
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_r8 --gpu_id 4 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_8.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
done

# (4) DeepMind perception test (_d).

for MV in v112 v113 v114 ba8 ba10 ba11 ba12 # v111 # 
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_d8 --gpu_id 5 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
done

# (5) Davis + YouTube-VOS / in the wild (_y).

for MV in v112 v113 v114 ba8 ba10 ba11 ba12 # v111 # 
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_y8 --gpu_id 6 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_skip.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --aot_eval_size_fix 1 --avoid_wandb 2
done


# =========================
# ======== PERFECT ========
# =========================

# sim (1) + (2)

for MV in v111
do
for PB in query static linear jump
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_t8 --gpu_id 0 --data_path \
/proj/vondrick3/datasets/kubcon_v10/ \
--use_data_frac 0.5 --num_workers 4 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_b8 --gpu_id 0 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
done
done

# real (3) + (4) + (5)

for MV in v111
do
for PB in query
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_r8 --gpu_id 0 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_1_8.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_d8 --gpu_id 0 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_y8 --gpu_id 0 --data_path \
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



