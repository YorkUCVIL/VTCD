# Unidentified Video Objects (UVO) Dataset: UVO v1.0 Kit Guide

## Introduction to UVO
UVO is a video instance segmentation dataset designed for open-world detection and segmentation with **ALL** objects annotated exhaustively in any given video.

Videos annotated in UVO are from the Kinetics400 dataset. We refer to the [DeepMind website](https://deepmind.com/research/open-source/kinetics) and [Computer Vision Data Foundation (CVDF)'s snapshot](https://github.com/cvdfoundation/kinetics-dataset) for downloading the videos.

For more information about the UVO dataset, please refer to our [paper](https://arxiv.org/abs/2104.04691).

## Annotation introduction
UVO v1.0 is the full version of the dataset. It is around 2x compared to UVO v0.5 dense set released previously. It also includes per-category label for COCO taxonomy on the dense set. We hope the annotations can serve as a playground for researchers to develope algorithms towards open-world segmentation

UVO 1.0 contains three splits and folders: ``` FrameSet ```, ``` VideoSparseSet ``` and ``` VideoDenseSet ```.

### FrameSet
``` FrameSet ``` is the same as v0.5 and consists of videos from Kinetics training split, annotated at 1fps without temporal consistency (tracking). We annotate 3 frames per video. The goal of ``` FrameSet ``` is to study the open-world segmentation problem at an image level. Inside ``` FrameSet ```, we provide a training split (``` UVO_frame_train.json ```) of **15414** frames from 5138 videos and a validation split (``` UVO_frame_val.json ```) of **7356** frames from 2452 videos. Both sets can be used for the final challenge on test splits. We encourage researchers interested only in image modeling work on this set. 

For details on baselines and guides for ```FrameSet```, we refer to ``` FrameSet/README.md ```.

### VideoSparseSet
``` VideoSparseSet ``` is the same annotation and video as ``` FrameSet ```, but represented in the video format (see ``` VideoDenseSet\README.md ``` for details). We provide 30fps interpolation to generate pseudo-groundtruth to un-annotated frames using [STM](https://arxiv.org/abs/1904.00607) with matching (see UVO paper for more details). They are provided in ``` UVO_sparse_train_video_with_interpolation.json ``` and ``` UVO_sparse_val_video_with_interpolation ```. It is optional to use them for either frame or video task. We also provide the 1fps version without interpolation in ``` UVO_sparse_train_video.json ``` ``` UVO_sparse_val_video.json ```.

### VideoDenseSet

``` VideoDenseSet ``` consists of videos from Kinetics400 validation split, annotated densely at 30fps and tracked over time. The goal of this set is to study video open-world segmentation. On the other hand, these annotations can optional be used towards the frame/image modeling. All objects are annotated in the 3-second clip (90 annotated frames per video). Objects within COCO categories are labeled with their respective COCO labels; objects whose categories are uncertain to human annotators or are outside COCO taxonomy are labeled as "other". Current training split includes **503** videos (same as PlaySet, ``` UVO_video_train_dense.json ```) and validation split includes **256** videos (``` UVO_video_val_dense ```).


## About ECCV22 Challenge 
Current UVO v1.0 will be used for Open-world segmentation challenge at ECCV2022 in conjunction with [MOTComplex workshop](https://motcomplex.github.io/). For more details on the compeition, please see [UVO website](https://sites.google.com/view/unidentified-video-object/challenge-intro). 

The test set will be udpated soon.

We also provide two example submission files in ```ExampleSubmission``` folder. They contain baseline prediction for validation set for Image/Frame segmentation and Video segmentation.

## Downloading and Preprocessing Kinetics400 Videos
We provide an example downloading linux cmd generation script ```download_kinetics.py```. The package relies on [youtube-dl](https://youtube-dl.org/) to download raw videos from Kinetics400. You can use the youtube ids we summarized in pickle in folder ```YT-IDs```.

To maintain a reasonable cost, annotation size and cross-video consistency, annotations and videos are pre-processed such that the shortest edge is **480 pixels** at **30fps**. We used FFmpeg's highest quality codec to achieve the resizing. We provided an example script in ```preprocess_kinetics.py``` for reference. The script will generate a csv file with a list of shell commands (based on ffmpeg) to process the videos.

We provide an example script to split videos into frames that are annotated: ```video2frames.py```. It depends on [opencv's python version ](https://docs.opencv.org/4.5.0/d6/d00/tutorial_py_root.html).

## Example Evaluation Script
We provide the APIs we used to evaluate model performance on both image/frame segmentation and video segmentation. They are located in ```EvaluationAPI``` folder.

## Contact
For questions or feedback about the dataset, don't hesitate to email the authors of UVO paper.
