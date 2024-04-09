# BVH, Oct 2022.
# After data_plugin image reading efficiency updates.

# timing
ulimit -n 65536 && WANDB_API_KEY=2199a720836a729b92919e12cee30afd8f0646f3 python eval/test.py --resume v111 --name dv111_dy2 --gpu_id 7 --data_path \
/proj/vondrick3/basile/hide-seek/plugin-ready/davis_2017.txt \
/proj/vondrick3/basile/hide-seek/plugin-ready/ytvos_2019.txt \
--use_data_frac 1.0 --num_workers 4 --num_queries 1

# get target visuals
ulimit -n 65536 && python eval/test.py --resume v112 --name dv112_db1 --gpu_id 7 --data_path \
/proj/vondrick3/basile/kubbench_v3/ \
--use_data_frac 1.0 --num_workers 4 --num_queries 1  --aot_eval_size_fix 1 --avoid_wandb 2 --extra_visuals 1
