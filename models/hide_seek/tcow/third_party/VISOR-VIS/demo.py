#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:04:58 2022

@author: Ahmad Darkhalil
"""
from vis import *
import os
import sys

# json_files_path = '../json_files'
# output_directory ='../results'
# output_resolution= (854,480)
# is_overlay=False
# rgb_frames = '../Images'
# generate_video=True


if sys.argv[1] == '1':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/annotations/train'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/train_vis_overlay'
    output_resolution = (854, 480)
    is_overlay = True
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/train'
    generate_video = True

elif sys.argv[1] == '2':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/annotations/train'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/train_vis'
    output_resolution = (854, 480)
    is_overlay = False
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/train'
    generate_video = True

elif sys.argv[1] == '3':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/annotations/val'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/val_vis_overlay'
    output_resolution = (854, 480)
    is_overlay = True
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/val'
    generate_video = True

elif sys.argv[1] == '4':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/annotations/val'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/val_vis'
    output_resolution = (854, 480)
    is_overlay = False
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/val'
    generate_video = True

elif sys.argv[1] == '5':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis2'
    output_resolution = (854, 480)
    is_overlay = False
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis'
    generate_video = True

elif sys.argv[1] == '6':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis2_overlay'
    output_resolution = (854, 480)
    is_overlay = True
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis'
    generate_video = True

elif sys.argv[1] == '7':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis3_overlay'
    output_resolution = (854, 480)
    is_overlay = True
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/all_rgb_flat'
    generate_video = True

elif sys.argv[1] == '8':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis4'
    output_resolution = (854, 480)
    is_overlay = False
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/merged_extracted_frames/'
    generate_video = True

elif sys.argv[1] == '9':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis4_overlay'
    output_resolution = (854, 480)
    is_overlay = True
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/merged_extracted_frames/'
    generate_video = True

elif sys.argv[1] == '10':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/train'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/train_vis4'
    output_resolution = (854, 480)
    is_overlay = False
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/merged_extracted_frames/'
    generate_video = True

elif sys.argv[1] == '11':

    json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/train'
    output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/train_vis4_overlay'
    output_resolution = (854, 480)
    is_overlay = True
    rgb_frames = '/proj/vondrick/datasets/epic-kitchens/VISOR/merged_extracted_frames/'
    generate_video = True

else:

    raise ValueError(sys.argv)


folder_of_jsons_to_masks(
    json_files_path, output_directory, is_overlay=is_overlay,
    rgb_frames=rgb_frames, output_resolution=output_resolution, generate_video=generate_video)
