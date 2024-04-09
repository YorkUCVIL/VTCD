# BVH, Nov 2022.

# on cv<=10:
cdb3
cd hide-seek
conda activate base

# NOTE: Here only (model, dataset) pairs that I want to include in appendix.

# Changes from t10 to t11:
# - Recorded more real videos (+ 9_bowlbox).
# - Switch to v3 guides (incl forgotten ytvos vid and above new rubric office vids).
# - Updated logvis visualizations for appendix.

# - TODO LATER Select epoch ~52 instead of latest.

# (1) Kubric random test (_t).

for MV in v111 ba8 ba10
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_t11 --gpu_id 2 --data_path \
/proj/vondrick3/datasets/kubcon_v10/ \
--use_data_frac 0.5 --num_workers 8 --num_queries 4 --avoid_wandb 2 --extra_visuals 1
done

# (2) Kubric curated benchmark (_b) (30 scenes).

for MV in v111 ba8 ba10
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_b11 --gpu_id 3 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1 --avoid_wandb 2 --extra_visuals 1
done

# (3) Real benchmark (_r).

for MV in v111 ba8 ba10
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_r11 --gpu_id 4 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/phone_2_9.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --avoid_wandb 2 --extra_visuals 1
done

# (4) DeepMind perception test (_d).

for MV in v111 ba8 ba10
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_d11 --gpu_id 5 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/dmpt_cgt.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --avoid_wandb 2 --extra_visuals 1
done

# (5) Davis + YouTube-VOS / in the wild (_y).

for MV in v111 ba8 ba10
do
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_y11 --gpu_id 6 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_skip.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1 --avoid_wandb 2 --extra_visuals 1
done



# =========================
# ======== PERFECT ========
# =========================

# sim (1) + (2) - all

for MV in v112
do
for PB in linear jump static query
do
# ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_t11 --gpu_id 0 --data_path \
# /proj/vondrick3/datasets/kubcon_v10/ \
# --use_data_frac 0.5 --num_workers 8 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2
ulimit -n 65536 && python eval/test.py --resume ${MV} --name ${MV}_b11 --gpu_id 0 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 8 --num_queries 1 --perfect_baseline ${PB} --avoid_wandb 2 --extra_visuals 1
done
done

