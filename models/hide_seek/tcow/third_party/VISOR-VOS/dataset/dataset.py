import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob
direction_m = False
class VISOR_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = self.mask_dir #os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                #print(self.num_frames[_video])
                _mask = np.array(Image.open(sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')),reverse=direction_m)[0]).convert("P"))
                #print(sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')))[0])
                self.num_objects[_video] = np.max(_mask)
                #print(self.num_objects[_video])
                self.shape[_video] = np.shape(_mask)
                _mask_480 = np.array(Image.open(sorted(glob.glob(os.path.join(self.mask480_dir, _video, '*.png')),reverse=direction_m)[0]).convert("P"))
                self.size_480p[_video] = np.shape(_mask_480)


        self.K = 16
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


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

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]
        info['start_frame'] = int(sorted(glob.glob(os.path.join(self.mask_dir, video, '*.png')),reverse= direction_m)[0].split("/")[-1][-14:-4])

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        return num_objects, info

    def load_single_image(self,video,f):
        import glob
        #print("F=:",f)
        #print("DIR",self.image_dir)
        #print(sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")))[f])
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")),reverse=direction_m)[f]
        #print("FIRST",img_file)

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            mask_file = sorted(glob.glob(os.path.join(self.mask_dir, video,"*.png")),reverse=direction_m)[f]
            #print("MASK",mask_file)
            N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms      

    def load_single_image_reserse(self,video,f):
        import glob
        #print("F=:",f)
        #print("DIR",self.image_dir)
        #print(sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")))[f])
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")),reverse= not direction_m)[f]
        #print(f"Added dded image of video {video} is {img_file}")

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            mask_file = sorted(glob.glob(os.path.join(self.mask_dir, video,"*.png")),reverse= not direction_m)[f]
            print("Added MASK",mask_file)
            N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms      


    def load_single_image_name(self,video,f):
        import glob
        file_name_jpg = sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")),reverse=direction_m)[f]
        return file_name_jpg.split("/")[-1][:-4]

    def load_single_image_name_path(self,video,f):
        import glob
        file_name_jpg = sorted(glob.glob(os.path.join(self.mask_dir, video,"*.png")),reverse=direction_m)[f]
        return file_name_jpg

    def load_palette(self,video):
        import glob
        palette = Image.open(glob.glob(os.path.join(self.mask_dir, video,"*.png"))[0]).getpalette()
        return palette
if __name__ == '__main__':
    pass
