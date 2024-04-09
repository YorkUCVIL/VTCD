import os
import json
import argparse

from utils.vis import *
import cv2
import numpy as np
import shutil
import glob
import csv
import pandas as pd
import json
import collections, functools, operator
from PIL import Image
from tqdm import tqdm

import traceback

set_of_jsons = {}
global_keys={}
sequences = set()

#the unseen kitchens in train
unseen_kitchens = ['P07_101','P07_103','P07_110','P09_02','P09_07','P09_104','P09_103','P09_106','P21_01','P21_01','P29_04']


def json_to_masks(filename,output_directory,images_root,object_keys=None,output_resolution="854x480"):
    height = int(output_resolution.split('x')[1])
    width = int(output_resolution.split('x')[0])
    os.makedirs(output_directory, exist_ok=True)
    global sequences
    f = open(filename)
    # returns JSON object as a dictionary
    data = json.load(f)
    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data['video_annotations'], key=lambda k: k['image']['image_path'])
    # Iterating through the json list
    full_path=""
    for datapoint in data:
        
        # BVH MOD
        try:
        
            image_name = datapoint["image"]["name"]
            image_path = datapoint["image"]["image_path"]
            seq_name = datapoint["image"]["subsequence"]
            masks_info = datapoint["annotations"]
            full_path =output_directory+'/' +seq_name+'/'#until the end of sequence name
            os.makedirs(full_path,exist_ok= True)
            os.makedirs(full_path.replace('Annotations','JPEGImages'), exist_ok=True,mode=0o777)
            
            # BVH MOD
            img_path = os.path.join(images_root,datapoint["image"]["video"]+'/'+image_name)
            if not(os.path.exists(img_path)):
                print('img_path not found? ',img_path)
                continue
            
            img1 = cv2.imread(img_path)
            resized1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(full_path.replace('Annotations','JPEGImages'),image_name),resized1)
            object_keys_values = generate_masks(image_name, image_path, masks_info, full_path,object_keys[seq_name],(width,height)) #this is for unique id for each object throughout the video // image_path[2] is the seq
            global_keys[seq_name] = object_keys_values
            sequences.add(full_path[:-1])

        except Exception as e:
            
            print(f'FAILED FOR {image_name}')
            print(e)
            traceback.print_exc()
            continue

def folder_of_jsons_to_masks(input_directory,output_directory,images_root,mapping_file,k,keep_first_frame_masks_only=False,output_resolution="854x480"):

    global sequences
    objects_keys = {}
    for json_file in tqdm(sorted(glob.glob(input_directory + '/*.json'))):

        # BVH MOD
        try:

            #print(json_file)
            if keep_first_frame_masks_only:
                objects = do_stats_stage2_jsons_single_file_vos(json_file)
                
            else:
                objects = do_stats_stage2_jsons_single_file(json_file)
            
            json_to_masks(json_file,output_directory,images_root,objects,output_resolution)
    
        except Exception as e:
            
            print(f'FAILED FOR {json_file}')
            print(e)
            traceback.print_exc()
            continue
    
    file_of_seq = os.path.join('/'.join(output_directory.split('/')[:-2]),'ImageSets/2022/'+os.path.basename(input_directory)+'.txt')
    
    if os.path.basename(input_directory) == 'val':
        unseen_sequences = filter_sequences_with_less_than_k(sequences,file_of_seq,k,include_unseen=True)
        file_of_seq = os.path.join('/'.join(output_directory.split('/')[:-2]),'ImageSets/2022/'+os.path.basename(input_directory)+'_unseen.txt')
        textfile = open(file_of_seq, "w")
        for element in sorted(unseen_sequences):
            textfile.write(element)
            if unseen_sequences.index(element) != (len(unseen_sequences)-1):
               textfile.write('\n') 
        textfile.close()       
    else:
        filter_sequences_with_less_than_k(sequences,file_of_seq,k,include_unseen=False)



    
    
    out_file = open(mapping_file, "w")
    json.dump(global_keys, out_file)
    out_file.close()
def filter_sequences_with_less_than_k(sequences,file_of_seq, k,include_unseen=False):
    global unseen_kitchens
    unseen_sequences = []
    print('Data cleaning . . . ')
    os.makedirs('/'.join(file_of_seq.split('/')[:-1]),exist_ok= True)
    files,included_sequences = find_number_of_images_per_seq(sequences,k)
    print(f'Number of sequences with less than {k} images is {len(files)} (deleted)')
    print(f'Number of sequences AFTER cleaning is {len(included_sequences)}')

    textfile = open(file_of_seq, "w")
    included_sequences = sorted(included_sequences) 
    for element in sorted(included_sequences):
        textfile.write(element)
        if '_'.join(element.split('_')[:2]) in unseen_kitchens:
            unseen_sequences.append(element)
        if included_sequences.index(element) != (len(included_sequences)-1):
           textfile.write('\n') 
    textfile.close()
    return unseen_sequences
def find_number_of_images_per_seq(sequences,k):

    files = []
    included_sequences = []
    for seq in sequences:
        num_files = len(glob.glob(seq+'/*.png'))
        if num_files < k:
            files.append({seq.split('/')[-1]:num_files})
            if os.path.exists(seq):
                shutil.rmtree(seq)
            if os.path.exists(seq.replace('Annotations','JPEGImages')):
                shutil.rmtree(seq.replace('Annotations','JPEGImages'))
            #print(seq.replace('Annotations','JPEGImages'))
        else:
            #print(seq.split('/')[-1])
            included_sequences.append(seq.split('/')[-1])
            
    return files,included_sequences 

def do_stats_stage2_jsons_single_file(file):

    total_number_of_images=0
    total_number_of_objects = 0
    total_number_of_seq = 0
    total_number_objects_per_image=[]
    objects=[]
    infile=file
    f = open(infile)
    # returns JSON object as a dictionary
    data = json.load(f)

    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data['video_annotations'], key=lambda k: k['image']['image_path'])

    total_number_of_images = total_number_of_images + len(data)

    # Iterating through the json list
    index = 0
    full_path=""
    prev_seq = ""
    obj_per_image=0
    masks_per_seq = {}
    for datapoint in data:
        obj_per_image=0 # count number of objects per image
        #seq = datapoint['documents'][0]['directory'].split("/")[2]
        seq = datapoint['image']['subsequence']
        
        if (seq != prev_seq):
            total_number_of_seq = total_number_of_seq + 1
            
            if prev_seq != "":
                objects_counts = collections.Counter(objects)
                masks_per_seq[prev_seq] = objects_counts.most_common()
                objects = []
            prev_seq = seq
        image_name = datapoint['image']['name']
        image_path = datapoint['image']['image_path']
        masks_info = datapoint["annotations"]
        #generate_masks(image_name, image_path, masks_info, full_path) #this is for saving the same name (delete the if statemnt as well)
        entities = masks_info
        for entity in entities: #loop over each object
            object_annotations = entity["segments"]
            if not len(object_annotations) == 0: #if there is annotation for this object, add it
                total_number_of_objects = total_number_of_objects + 1
                objects.append(entity["name"])
                obj_per_image = obj_per_image + 1
        total_number_objects_per_image.append(obj_per_image)


    objects_counts = collections.Counter(objects)

    df = pd.DataFrame.from_dict(objects_counts, orient='index').reset_index()
    if len(objects) != 0:
        objects_counts = collections.Counter(objects)
        masks_per_seq[seq] = objects_counts.most_common()


    return masks_per_seq
def do_stats_stage2_jsons_single_file_vos(file):

    total_number_of_images=0
    total_number_of_objects = 0
    total_number_of_seq = 0
    total_number_objects_per_image=[]
    objects=[]
    infile=file
    f = open(infile)
    # returns JSON object as a dictionary
    data = json.load(f)

    #sort based on the folder name (to guarantee to start from its first frame of each sequence)
    data = sorted(data['video_annotations'], key=lambda k: k['image']['image_path'])

    total_number_of_images = total_number_of_images + len(data)

    # Iterating through the json list
    index = 0
    full_path=""
    prev_seq = ""
    obj_per_image=0
    masks_per_seq = {}
    for datapoint in data:
        obj_per_image=0 # count number of objects per image
        seq = datapoint['image']['subsequence']
        
        if (seq != prev_seq):
            total_number_of_seq = total_number_of_seq + 1
            
            if prev_seq != "":
                objects_counts = collections.Counter(objects)
                masks_per_seq[prev_seq] = objects_counts.most_common()
                objects = []
            prev_seq = seq
            image_name = datapoint['image']['name']
            image_path = datapoint['image']['image_path']
            masks_info = datapoint["annotations"]
            entities = masks_info
            for entity in entities: #loop over each object
                object_annotations = entity["segments"]
                if not len(object_annotations) == 0: #if there is annotation for this object, add it
                    total_number_of_objects = total_number_of_objects + 1
                    objects.append(entity["name"])
                    obj_per_image = obj_per_image + 1
            total_number_objects_per_image.append(obj_per_image)


    objects_counts = collections.Counter(objects)
    import pandas as pd

    df = pd.DataFrame.from_dict(objects_counts, orient='index').reset_index()
    if len(objects) != 0:
        objects_counts = collections.Counter(objects)
        masks_per_seq[seq] = objects_counts.most_common()


    return masks_per_seq


if __name__ == "__main__":
    def get_arguments():
        parser = argparse.ArgumentParser(description="parameters for VISOR to DAVIS conversion")
        parser.add_argument("-set", type=str, help="train, val", required=True)
        parser.add_argument("-keep_first_frame_masks_only", type=int, help="this flag to keep all masks or the masks in the first frame only, this flag usually 1 when generating VAL and 0 when generating Train", required=True)
        parser.add_argument("-visor_jsons_root", type=str, help="path to the json files of visor",default='../VISOR')
        parser.add_argument("-images_root", type=str, help="path to the images root directory",default='../VISOR_images')
        parser.add_argument("-output_directory", type=str, help="path to the directory where you want VISOR to be",default='../data')
        parser.add_argument("-output_resolution", type=str, help="resolution of the output images and masks",default='854x480')

        return parser.parse_args()

    args = get_arguments()

    visor_set = args.set
    visor_jsons_root = args.visor_jsons_root
    output_directory = args.output_directory
    images_root = args.images_root
    keep_first_frame_masks_only = False if args.keep_first_frame_masks_only == 0 else True
    output_resolution = args.output_resolution
    height = output_resolution.split('x')[1]+'p'
    mapping_file = os.path.join(os.path.join(output_directory,'VISOR_2022'),visor_set+'_data_mapping.json')

    if os.path.exists(mapping_file):
        os.remove(mapping_file)


    print('Converting VISOR to DAVIS . . .')
    if visor_set =='val':
        if not keep_first_frame_masks_only:
            print('Warning!!, usually "keep_first_frame_masks_only" flag is True when generating Val except if you want to generate the data to train on Train/val')
        folder_of_jsons_to_masks(os.path.join(visor_jsons_root,visor_set), os.path.join(output_directory,'VISOR_2022/Annotations/'+height),images_root,mapping_file,2,keep_first_frame_masks_only,output_resolution)

    elif visor_set =='train':
        if keep_first_frame_masks_only:
            print('The "keep_first_frame_masks_only" flag should be False when generating Train!! please double check!!')
        folder_of_jsons_to_masks(os.path.join(visor_jsons_root,visor_set), os.path.join(output_directory,'VISOR_2022/Annotations/'+height),images_root,mapping_file,3,keep_first_frame_masks_only,output_resolution)
