# Understanding Video Transformers via Universal Concept Discovery
Official Implementation of our CVPR 2024 (Highlight) Paper.

[Paper](https://arxiv.org/abs/2401.10831), [project page](https://yorkucvil.github.io/VTCD/)


# Create Conda Environment
```
conda create -n VTCD python=3.10.12
conda activate VTCD
cd models/hide_seek/tcow/TimeSformer ; pip install -e .
cd segment-anything; pip install -e .
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/fvcore'
```

# Data Preparation
## Kubric Data Preparation
1) TBD

## SSV2 Data Preparation
1) Download SSV2 dataset from https://developer.qualcomm.com/software/ai-datasets/something-something and follow the instructions to extract the dataset
2) Set args.ssv2_path in run_vcd.py to the path of the SSV2 dataset

## DAVIS16 Data Preparation
1) Download and extract the DAVIS16 dataset from https://davischallenge.org/davis2016/code.html
2) Set args.davis16_path in run_vcd.py to the path of the DAVIS16 dataset

# Runing VTCD to discover and rank the importance of concepts

## 0) Download pretained models
Download the pre-trained models from the following link and extract the directory into the root directory of the project:
https://drive.google.com/drive/folders/1SGfDjA35BhxsJ8k-HwElzWn2YVSCoIot?usp=sharing

## 1) Run VTCD for concept discovery
To run VTCD with a specific model, dataset and target class, use the following arguments and launch run_vcd.py:
- --exp_name: Experiment name
- --model: Model name - currently supports the following models
  - timesformer_occ (TCOW)
  - vidmae_ssv2_pre (VideoMAE pre-trained on SSV2)
  - vidmae_ssv2_ft (VideoMAE fine-tuned on SSV2)
  - intern (InternVideo)
- --dataset : Dataset name - currently supports the following datasets
  - ssv2 (Something-Something-V2)
  - davis16 (DAVIS16)
  - kubric (Kubric)
- --attn_head: Attention heads to be clustered (0-11 for ViT-b)
- --cluster_layer: Layers to be clustered (0-11 for ViT-b)
- --target_class: Target class for concept if using ssv2 dataset
- --max_num_videos: Maximum number of videos to be used for clustering
- --slic_compactness: Compactness parameter for SLIC superpixels (tune first with log scale, then fine-tune with linear scale)
- --save_concepts: Whether to save concepts for visualization (True/False)

## 2) Run CRIS for concept importance evaluation
Then to evaluate the importance of the concepts, run head_concept_attribution_fidelity.py with the following arguments:
- --exp_name: Experiment name from the previous step
- --num_masks: Number of masks to be used for attribution (default: 4000 but 8000 for TCOW)
- --heads_removed_each_step: Number of heads to be removed at each step (default: 100)
- --masking_ratio: Ratio of heads to be masked at each step (default: 0.5 but 0.95 for VideoMAE pre-trained)

## Example
To run VTCD with VideoMAE pre-trained on SSV2 for the target class 'Rolling something on a flat surface', while saving the concepts, and then evaluate the importance of the concepts using CRIS, use the following commands:
```
python run_vcd.py --exp_name VideoMAE_Pretrained_SSv2_Rolling --target_class 'Rolling something on a flat surface' --attn_head 0 1 2 3 4 5 6 7 8 9 10 11  --max_num_videos 30 --model vidmae_ssv2_pre --dataset ssv2 --cluster_layer 0 1 2 3 4 5 6 7 8 9 10 11 --save_concepts
python evaluation/head_concept_attribution_fidelity.py --exp_name VideoMAE_Pretrained_SSv2_Rolling --num_masks 4000 --heads_removed_each_step 100 --masking_ratio 0.5
```

# Adding a new model
1) Load model and checkpoint into utilities/utils.py -> load_model (add model name to args.model in run_vcd.py)
2) Provide forward pass of model in get_layer_activations function in vcd.py (this needs to return a features variable with shape # features.shape = num_layers X channels X num_heads X time X height X width)
3) Ensure that the resizing functions work appropriately for the features in intra_video_clustering function in vcd.py (some models require different resizing functions due to different temporal and spatial resolutions)
4) If you want importance rankings, you will need to provide the forward pass and metric (e.g., loss, accuracy) in head_concept_attribution_fidelity.py, at lines 250 and 314
5) To add a new dataset, you need to add two args (path_to_DATASET and dataset) and then add a function, load_DATASET_videos, that loads the videos (b x c x t x h x w) and returns them in the same format as the current datasets in vcd.py

# Acknowledgements
Code structure modified from ACE
