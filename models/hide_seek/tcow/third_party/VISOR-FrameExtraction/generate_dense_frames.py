#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:56:25 2022

@author: Ahmad Darkhalil - University of Bristol
"""

import traceback
import glob
import os
import shutil
from PIL import Image
import cv2
import json
from tqdm import tqdm

def generate_images_n_V2(video_path,dest_folder,start_frame_id,number_of_frames,output_resolution=(854, 480)):
    #this function generates N number of images (frames) of a video. start_frame_id: suppose video starts with frame#1 as 'generate_images' works
    start_idx = int(start_frame_id.split("_")[-1][:-4]) - 1 # minus 1 since the following function suppose first frame is 0 where as when 'generate_images' started with frame 1
    prefix = '_'.join(start_frame_id.split("_")[:2])+'_'
    #print(start_frame_id)

    vid = video_path.split("/")[-1].split('.')[0]
    frame_rate = 60 if len(vid.split('_')[-1]) == 2 else 50
    ##print(f'vid: {vid}, fr: {frame_rate}')
    #-vf scale=854:480
    start_idx = start_idx/frame_rate
    command = "ffmpeg -loglevel error -threads 16 -ss "+str(start_idx)+" -i "+video_path+" -qscale:v 4 -qscale 2 -vf scale="+str(output_resolution[0])+":"+str(output_resolution[1])+" -frames:v  "+str(number_of_frames)+" "+dest_folder+"/"+prefix+"frame__%10d.jpg"
    #print(command)
    os.system(command)
    

def copy_jpg_image(image_name,in_path,out_path,output_resolution=(854, 480)):

    image_name = image_name.replace('png','jpg')
    
    img = cv2.imread(os.path.join(in_path,image_name))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, output_resolution, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_path,image_name),resized)

def rename_images(path, seq_start_frame_str,start, stop):
    #print(path)
    start_frame_id = seq_start_frame_str.split('/')[-1]
    prefix = '_'.join(start_frame_id.split('_')[:2]) + '_'
    for i in range(1,stop+1):
        try:
            from_name = os.path.join(path,'{}frame__{}.jpg'.format(prefix,str(i).zfill(10)))
            to_name = os.path.join(path,'{}frame_{}.jpg'.format(prefix,str(start+i-1).zfill(10)))
            #print('From: ', from_name)
            #print('To: ', to_name)
            os.rename(from_name,to_name)
        except:
            break;

            
def generate_dense_images_from_video_jsons(json_files_path, output_directory, sparse_rgb_images_root,videos_path,output_resolution=(854, 480)):
    for json_file in tqdm(glob.glob(os.path.join(json_files_path,'*.json')),desc = 'Processed videos'):

        # BVH MOD
        try:

            #print(json_file)
            seq_name = ""
            seq_start_frame_str=""
            seq_end_frame_str = ""
            seq_start_frame=0
            seq_end_frame = 0
            video_path = ''
            with open(json_file, 'r') as f:
                data = json.load(f)
            data = sorted(data["video_annotations"], key=lambda k: k["image"]['name'])
            
            for datapoint in data:
                infile = datapoint["image"]["image_path"] 
                #print (infile)
                v = datapoint["image"]["video"]
                #print('v',v)
                    
                video_path = os.path.join(videos_path, v+'.MP4')
                if seq_name != datapoint["image"]["interpolation"]: #check if the current seq is not the same with previous one (new seq)
                    if ((seq_name != "")  and (not os.path.exists(os.path.join(output_directory,seq_name)))): # if it is not the before starting (seq = "" as inital value)
                        #print(len(os.walk("/".join(infile.split("/")[:-2])).__next__()[1]))
                        #os.makedirs(os.path.join(output_dir,seq_name+"_"+str(seq_start_frame)+"_"+str(seq_end_frame)), exist_ok=True) this for start and end frame
                        
                        ##print("Seq: ",seq_name)
                        ##print("Start: ",seq_start_frame_str)
                        ##print("Start: ",seq_start_frame)
                        ##print("END: ",seq_end_frame)
                        ##print("-----------------------")
                        current_v =  datapoint["image"]["video"]
                        os.makedirs(os.path.join(output_directory,current_v), exist_ok=True)
                        #import pdb
                        #pdb.set_trace() 
                        generate_images_n_V2(video_path.replace(v,current_v), os.path.join(output_directory,current_v), seq_start_frame_str,
                                            seq_end_frame - seq_start_frame + 1,output_resolution)
                        rename_images(os.path.join(output_directory, current_v), seq_start_frame_str,seq_start_frame,seq_end_frame - seq_start_frame + 1)
                        copy_jpg_image(os.path.join(current_v,seq_start_frame_str),sparse_rgb_images_root,os.path.join(output_directory),output_resolution)
                        copy_jpg_image(os.path.join(current_v,seq_end_frame_str),sparse_rgb_images_root,os.path.join(output_directory),output_resolution)
                        #pdb.set_trace()
                        #generate_images_n_V2_index1(video_path, os.path.join(output_dir,seq_name), seq_start_frame_str,1)
                        #rename_images(os.path.join(output_dir, seq_name), seq_start_frame_str,seq_start_frame,1)
        
                        #generate_images_n_V2_index1(video_path, os.path.join(output_dir,seq_name), seq_start_frame_str.replace(str(seq_start_frame),str(seq_end_frame)),1)
                        #rename_images(os.path.join(output_dir, seq_name), seq_start_frame_str.replace(str(seq_start_frame),str(seq_end_frame)),seq_end_frame,1)
                        
                        seq_start_frame_str=datapoint["image"]["name"] 
                        seq_start_frame = int(datapoint["image"]["name"][-14:-4])
                    else: # if it is the first iternation, then assign the start frame
                        seq_start_frame_str=datapoint["image"]["name"] 
                        seq_start_frame = int(datapoint["image"]["name"][-14:-4])
        
                seq_name = datapoint["image"]["interpolation"]  # set the new seq
                seq_end_frame_str = datapoint["image"]["name"]  #store each frame as a final frame until the above if statement comes true
                seq_end_frame = int(datapoint["image"]["name"][-14:-4])
    
            if os.path.exists(video_path):
                #for final folder
                ##print("Seq: ", seq_name)
                ##print("Start: ", seq_start_frame_str)
                ##print("Start: ", seq_start_frame)
                ##print("END: ", seq_end_frame)
                ##print("-----------------------")
                os.makedirs(os.path.join(output_directory,current_v), exist_ok=True)
                generate_images_n_V2(video_path, os.path.join(output_directory,current_v), seq_start_frame_str,
                                    seq_end_frame - seq_start_frame + 1,output_resolution)
                rename_images(os.path.join(output_directory, current_v), seq_start_frame_str,seq_start_frame,seq_end_frame - seq_start_frame + 1)
        
                copy_jpg_image(os.path.join(current_v,seq_start_frame_str),sparse_rgb_images_root,os.path.join(output_directory),output_resolution)
                copy_jpg_image(os.path.join(current_v,seq_end_frame_str),sparse_rgb_images_root,os.path.join(output_directory),output_resolution)

        except Exception as e:
            
            print(f'FAILED FOR {json_file}')
            print(e)
            traceback.print_exc()
            continue
    

if __name__ == '__main__':

    import sys
    
    # json_files_path = '../jsons'
    # output_directory = '../out'
    # sparse_rgb_images_root = '../Images'
    # videos_path = '../videos' 
    # output_resolution = (854,480) #original interpolation resolution
    # generate_dense_images_from_video_jsons(json_files_path,output_directory,sparse_rgb_images_root,videos_path,output_resolution)

    if sys.argv[1] == '1':
        
        json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val'  # input
        output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val_vis'  # output
        sparse_rgb_images_root = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/val'  # input
        videos_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/videos_flat_from_v2'   # input
        output_resolution = (854,480) #original interpolation resolution
        generate_dense_images_from_video_jsons(json_files_path,output_directory,sparse_rgb_images_root,videos_path,output_resolution)
    
    elif sys.argv[1] == '2':
        
        json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/train'  # input
        output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/train_vis'  # output
        sparse_rgb_images_root = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/train'  # input
        videos_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/videos_flat_from_v2'   # input
        output_resolution = (854,480) #original interpolation resolution
        generate_dense_images_from_video_jsons(json_files_path,output_directory,sparse_rgb_images_root,videos_path,output_resolution)

    elif sys.argv[1] == '3':
        
        json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/train'  # input
        output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/train_extracted_frames'  # output
        sparse_rgb_images_root = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/train'  # input
        videos_path = '/proj/vondrick/datasets/epic-kitchens/all_videos_flat'   # input
        output_resolution = (854,480) #original interpolation resolution
        generate_dense_images_from_video_jsons(json_files_path,output_directory,sparse_rgb_images_root,videos_path,output_resolution)

    elif sys.argv[1] == '4':
        
        json_files_path = '/proj/vondrick/datasets/epic-kitchens/VISOR/Interpolations-DenseAnnotations/val'  # input
        output_directory = '/proj/vondrick/datasets/epic-kitchens/VISOR/val_extracted_frames'  # output
        sparse_rgb_images_root = '/proj/vondrick/datasets/epic-kitchens/VISOR/GroundTruth-SparseAnnotations/rgb_frames/val'  # input
        videos_path = '/proj/vondrick/datasets/epic-kitchens/all_videos_flat'   # input
        output_resolution = (854,480) #original interpolation resolution
        generate_dense_images_from_video_jsons(json_files_path,output_directory,sparse_rgb_images_root,videos_path,output_resolution)

