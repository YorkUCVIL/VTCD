
# plugin-ready-example
#CUDA_VISIBLE_DEVICES=1 python test.py --resume v111 --name v111_mytest --gpu_id 0 \
#--data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 \
#--cluster_layer 0 3 5 7 9 11
#
#
#CUDA_VISIBLE_DEVICES=1 python test.py --resume ba8 --name ba8_mytest --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0
#
#CUDA_VISIBLE_DEVICES=1 python test.py --resume ba10 --name ba10_mytest --gpu_id 0 \
#--data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 \
#--cluster_layer 0


# kubric
#CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume v111 --name v111_KubricVal1 --gpu_id 0 \
#--data_path /home/mattkowal/Desktop/data/kubcon_v10/val --num_workers 8 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 \
#--cluster_layer 0 3 5 7 9 11
#
#
#CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_KubricVal1 --gpu_id 0 \
#--data_path /home/mattkowal/Desktop/data/kubcon_v10/val --num_workers 8 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 \
#--cluster_layer 0
#
#CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba10 --name ba10_KubricVal1 --gpu_id 0 \
#--data_path /home/mattkowal/Desktop/data/kubcon_v10/val --num_workers 8 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 \
#--cluster_layer 0



CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume v111 --name v111_SampleKey --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 3 5 7 9 11 --concept_clustering --cluster_subject keys
CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume v111 --name v111_SampleValue --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 3 5 7 9 11 --concept_clustering --cluster_subject values
CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume v111 --name v111_SampleQuery --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 3 5 7 9 11 --concept_clustering --cluster_subject queries
CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume v111 --name v111_SampleToken --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 3 5 7 9 11 --concept_clustering --cluster_subject tokens


CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_CurrSampleKey --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering --cluster_memory curr --cluster_subject keys
CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_CurrSampleValue --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering --cluster_memory curr --cluster_subject values

CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_LongSampleKey --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering --cluster_memory long --cluster_subject keys
CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_LongSampleValue --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering --cluster_memory long --cluster_subject values

CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_ShortSampleKey --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering --cluster_memory short --cluster_subject keys
CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_ShortSampleValue --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering --cluster_memory short --cluster_subject values

CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba8 --name ba8_TokenSampleToken --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering --cluster_subject tokens


#CUDA_VISIBLE_DEVICES=1 python eval/test.py --resume ba10 --name ba10_Sample --gpu_id 0 --data_path data/plugin-ready-example/phone_8.txt --num_workers 0 --num_queries 1 --avoid_wandb 2 --extra_visuals 1 --cluster_layer 0 1 2 --concept_clustering

