import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2

import torch
import torchvision
from torch.utils import data

import random
import glob

import pdb
from dataset.aug import aug_heavy


MAX_OBJECT_NUM_PER_SAMPLE = 15
class VISOR_MO_Train(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.skip = 0
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        self.img_files = {}
        self.mask_files = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.img_files[_video] = sorted(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                self.mask_files[_video] = sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')))
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                #print("mask dir:",self.mask_dir)
                #print("mask video:",_video)
                _mask = np.array(Image.open(sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')))[0]).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask_480 = np.array(Image.open(sorted(glob.glob(os.path.join(self.mask480_dir, _video, '*.png')))[0]).convert("P"))
                self.size_480p[_video] = np.shape(_mask_480)


        self.K = 16
        self.single_object = single_object
        self.aug = aug_heavy()

    def change_skip(self,f):
        self.skip = f

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def mask_process(self,mask,f,num_object,ob_list):
        n = num_object
        mask_ = np.zeros(mask.shape).astype(np.uint8)
        if f == 0:
            for i in range(1,11):
                if np.sum(mask == i) > 0:
                    n += 1
                    ob_list.append(i)

            if n > MAX_OBJECT_NUM_PER_SAMPLE:
                n = MAX_OBJECT_NUM_PER_SAMPLE
                ob_list = random.sample(ob_list,n)

        for i,l in enumerate(ob_list):
            mask_[mask == l] = i + 1
        return mask_,n,ob_list 

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        img_files = self.img_files[video]
        mask_files = self.mask_files[video]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]
        info['start_frame'] = int(sorted(glob.glob(os.path.join(self.mask_dir, video, '*.png')))[0].split("/")[-1][-14:-4])

        N_frames = np.empty((3,)+(384,384,)+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+(384,384,), dtype=np.uint8)
        frames_ = []
        masks_ = []
        n1 = random.sample(range(0,self.num_frames[video] - 2),1)[0]
        n2 = random.sample(range(n1+1,min(self.num_frames[video] - 1,n1 + 2 + self.skip)),1)[0]
        n3 = random.sample(range(n2+1,min(self.num_frames[video],n2 + 2 + self.skip)),1)[0]
        frame_list = [n1,n2,n3]
        #print(video, ":",frame_list)
        num_object = 0
        ob_list = []

        for f in range(3):
            img_file = img_files[frame_list[f]]
            tmp_frame = np.array(Image.open(img_file).convert('RGB'))
            try:
                mask_file = mask_files[frame_list[f]]  
                tmp_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                tmp_mask = 255

            frames_.append(tmp_frame.astype(np.float32))
            masks_.append(tmp_mask)
        frames_,masks_ = self.aug(frames_,masks_)

        for f in range(3):
            masks_[f],num_object,ob_list = self.mask_process(masks_[f],f,num_object,ob_list)
            N_frames[f],N_masks[f] = frames_[f],masks_[f]
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()

        if num_object == 0:
            num_object += 1
        num_objects = torch.LongTensor([num_object])
        return Fs, Ms, num_objects, info

if __name__ == '__main__':


    from helpers import overlay_davis
    import matplotlib.pyplot as plt
    import os
    import pdb

    visor_root = '../data/VISOR_2022'
    dataset = VISOR_MO_Train(visor_root)
    dataset.skip = 2
    palette = Image.open(os.path.join(visor_root + 'Annotations/480p/P01_01_seq_00001/P01_01_frame_0000000140.png')).getpalette()


    output_dir = 'tmp'
    print('Generating 3-images samples for the whole dataset:')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i,(Fs,Ms,num_objects,info) in enumerate(dataset):
        pred = np.argmax(Ms.numpy(), axis=0).astype(np.uint8)
        img_list = []
        for f in range(3):
            pF = (Fs[:,f].permute(1,2,0).numpy()*255.).astype(np.uint8)
            pE = pred[f]
            canvas = overlay_davis(pF, pE, palette,alpha=0.3)
            img = np.concatenate([canvas],axis = 0) #img = np.concatenate([pF,canvas],axis = 0)
            img_list.append(img)
        out_img = np.concatenate(img_list,axis = 1)
        out_img = Image.fromarray(out_img)
        out_img.save(os.path.join(output_dir, str(i).zfill(5) + '.jpg'))
        #pdb.set_trace()
