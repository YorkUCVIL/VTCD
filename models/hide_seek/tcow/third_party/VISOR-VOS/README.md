# VISOR-VOS

This repository contains the code to replicate the Video Object Segmentations benchmark of the [VISOR](https://epic-kitchens.github.io/VISOR/) dataset. It replicates the results of table 3 in our paper: EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations

<br>

## Download pre-trained models

| backbone |  training stage | training dataset | J&F | J |  F  | weights |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| resnet-50 |  stage 1 | MS-COCO | 56.9 | 55.5 | 58.2 | [`link`](https://www.dropbox.com/s/bsy577kflurboav/coco_res50.pth?dl=0) |
| resnet-50 | stage 2 | MS-COCO -> VISOR | 75.8 | 73.6 | 78.0 | [`link`](https://www.dropbox.com/s/6vkkr6vbx7ybku3/coco_lr_fix_skip_0_1_release_resnet50_400000_32_399999.pth?dl=0) |


# Requirements
- Python 3.9.7
- Numpy 1.20.3
- Pillow 8.4.0
- Opencv-python 4.5.5
- Imgaug 0.4.0
- Scipy 1.7.1
- Tqdm 4.62.3
- Pandas 1.3.4
- Torchvision 0.12.0

## Datasets

#### [MS-COCO](https://cocodataset.org/#home)
MS-COCO instance segmentation dataset is used to generate synthitic video out of 3 frames to train STM. This could be helpful as a pretraining stage before doing the main training on VISOR. <br>

![image](https://user-images.githubusercontent.com/19390123/115352832-62fb7d00-a1ea-11eb-9fbe-1f84bf74905d.png)


#### [VISOR](https://epic-kitchens.github.io/VISOR/)
After pretrain on MS-COCO, we fine-tune on VISOR dataset by sample 3 frames from a sequence in each training iteration. To visualize VISOR dataset, you can check [VISOR-VIS](https://github.com/epic-kitchens/VISOR-VIS)
![00230](https://user-images.githubusercontent.com/24276671/192192037-bec3f981-0cc5-405d-85bc-610e883d0466.jpg)


#### Dataset Structure
To run the training or evaluation scripts, the dataset format should be as follows (following [DAVIS](https://davischallenge.org/) format), a script is given in the next step to convert VISOR to DAVIS-like dataset.
```

|- VISOR_2022
  |- JPEGImages
  |- Annotations
  |- ImageSets
     |- 2022
        |- train.txt
        |- val.txt
        |- val_unseen.txt

|- MS-COCO
  |- train2017
  |- annotations
      |- instances_train2017.json
```

#### VISOR to DAVIS-like format
To generate the required structure you have to download the [VISOR](https://epic-kitchens.github.io/VISOR/) train/val images and json files first , then you can run ```visor_to_davis.py``` script with the following parameters:

`set`: **train** or **val**, which is the split that you want to generate DAVIS-like dataset for. <br>
`keep_first_frame_masks_only`: **0** or **1**, this flag to keep all masks for each sequence or the masks in the first frame only, this flag usually 1 when generating  **val** and 0 when generating **train**<br>
`visor_jsons_root`: path to the json files of visor, the train and val folders should exists under this root directory as follows: 
```
|- visor_jsons_root
   |- train
      |- P01_01.json
      |- PXX_(X)XX.json
   |- val
      |- P01_107.json
      |- PXX_(X)XX.json
```
`images_root`: path to the RGB images root directory. The images should be in the following structure: 
```
|- images_root
   |- P01_01
      |- P01_01_frame_xxxxxxxxxx.jpg
   |- PXX_XXX
      |- PXX_(X)XX_frame_xxxxxxxxxx.jpg
```
`output_directory`: path to the directory where you want VISOR to be, a VISOR_2022 directory would be automatically created with DAVIS-like formatting. <br>
`output_resolution`: resolution of the output images and masks, however, the VOS baseline tested on 480p which is the default value for this parameter.
<br>
This is sample run of the script to generate train and val with 480p resolution, **you must run it twice, one to generate train and another one to generate val**, note that the keep_first_frame_masks_only changes since you have to keep all masks in the training split unlike the validation where we have to keep the masks in the first frame only for proper evaluation:
```
To generate val:
python visor_to_davis.py -set val -keep_first_frame_masks_only 1  -visor_jsons_root . -images_root ../VISOR_Images/Images_fixed -output_directory ../out_data

To generate train:
python visor_to_davis.py -set train -keep_first_frame_masks_only 0  -visor_jsons_root . -images_root ../VISOR_Images/Images_fixed -output_directory ../out_data
```

The scripts also will create the txt files that should be in the DAVIS-like dataset structre. Also it creates mapping files under the `output_directory` to maps each colors in the images with the object name in VISOR for any object-related analysis.
## Training

#### Stage 1
To pretrain on MS-COCO, you can run the following script.
```
python train_coco.py -Dvisor "path to visor" -Dcoco "path to coco" -backbone "[resnet50,resnet18]" -save "path to save models"
#e.g.
python train_coco.py -Dvisor ../data/Davis/ -Dcoco ../data/Ms-COCO/ -backbone resnet50 -save ../coco_weights/
```

#### Stage 2
Main traning on VISOR, to get the best performance, you should resume from the MS-COCO pretrained model in Stage 1.
```
python train_stm_baseline.py -Dvisor "path to visor" -total_iter "total number of iterations" -test_iter "test every this number of iterations" -backbone "[resnet50,resnet18]" -wandb_logs "1 if you want to save the logs into your wandb account (offline)" -save "path to save models" -resume "path to coco pretrained weights"
#e.g. 
python train_stm_baseline.py -Dvisor  ../VISOR_2022/ -total_iter 400000 -test_iter 40000 -batch 32 -backbone resnet50 -save ../visor_weights/ -name experiment1 -wandb_logs 0  -resume ../coco_weights/coco_res50.pth
```

## Evaluation
Evaluating on VISOR based on DAVIS evaluation codes, we adjusted the codes to include the last frame of the sequence in our scores 
```
python eval.py -g "gpu id" -s "set" -y "year" -D "path to visor" -p "path to weights" -backbone "[resnet50,resnet18,resnest101]"
#e.g.
python eval.py -g 0 -s val -y 22 -D ../data/VISOR -p ../visor_weights/coco_lr_fix_skip_0_1_release_resnet50_400000_32_399999.pth -backbone resnet50
```

## Acknowledgement

When use this repo, any of our models or dataset, you need to cite the VISOR paper

## Citing VISOR
```
@inproceedings{VISOR2022,
  title = {EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations},
  author = {Darkhalil, Ahmad and Shan, Dandan and Zhu, Bin and Ma, Jian and Kar, Amlan and Higgins, Richard and Fidler, Sanja and Fouhey, David and Damen, Dima},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
  year = {2022}
}
```

We use the code in the original STM implementation from [official STM repository](https://github.com/seoungwugoh/STM) and the implementation from [STM training repository](https://github.com/haochenheheda/Training-Code-of-STM). Using this code, you also need to cite STM

## Citing STM
```
@inproceedings{oh2019video,
  title={Video object segmentation using space-time memory networks},
  author={Oh, Seoung Wug and Lee, Joon-Young and Xu, Ning and Kim, Seon Joo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9226--9235},
  year={2019}
}
```

# License

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
