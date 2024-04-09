from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy


### My libs
from dataset.dataset import VISOR_MO_Test
#from dataset.dataset import DAVIS_MO_Test
from model.model import STM

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
from scipy.optimize import linear_sum_assignment


def Run_video(dataset,video, num_frames, num_objects,model,Mem_every=None, Mem_number=None,out_directory='../results_png'):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        to_memorize=  [0]

    palette = dataset.load_palette(video) # to load the dataset palette
    #    raise NotImplementedError
    F_last,M_last = dataset.load_single_image(video,0)
    F_last = F_last.unsqueeze(0)
    M_last = M_last.unsqueeze(0)
    E_last = M_last
    pred = np.zeros((num_frames,M_last.shape[3],M_last.shape[4]))
    all_Ms = []
    for t in range(1,num_frames):

        #print('current_frame: {},num_frames: {}, num_objects: {}'.format(t, num_frames, num_objects.numpy()[0]))

        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(F_last[:,:,0], E_last[:,:,0], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        del prev_key,prev_value

        F_,M_ = dataset.load_single_image(video,t)

        F_ = F_.unsqueeze(0)
        M_ = M_.unsqueeze(0)
        all_Ms.append(M_.cpu().numpy())
        del M_
        # segment
        with torch.no_grad():
            logit = model(F_[:,:,0], this_keys, this_values, torch.tensor([num_objects]))
        E = F.softmax(logit, dim=1)
        del logit
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
            del this_keys,this_values
        pred[t] = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        
        
        
        ##this to save images
        #palette = Image.open(os.path.join(dataset + 'Annotations/480p/P01_01_seq_00001/P01_01_frame_0000000140.png')).getpalette()

        #print(video)
        test_path = os.path.join(out_directory, video)

        if not os.path.exists(test_path):
            os.makedirs(test_path)
        #print(pred[t].shape)
        jpg_filename = dataset.load_single_image_name(video, t)
        #print(jpg_filename)
        #print(type(pred[t]))
        pred1 = torch.argmax(E[0], dim=0).cpu().numpy().astype(np.uint8)
        #print(type(pred1))
        img_E = Image.fromarray(pred1).convert('P')
        
        img_E.putpalette(palette)

        img_E.save(os.path.join(test_path, jpg_filename+'.png'))  #### there to save the images
        del pred1
        #####
        
        E_last = E.unsqueeze(2)
        F_last = F_
    Ms = np.concatenate(all_Ms,axis=2)
    return pred,Ms

def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    ##print('J_metric: ', j_metrics_res.shape)
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res

def evaluate(model,Testloader,metric):

        
    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for V in tqdm.tqdm(Testloader):
        num_objects, info = V
        seq_name = info['name']
        num_frames = info['num_frames']
        ##print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects.numpy()[0]))

        pred,Ms = Run_video(Testloader, seq_name, num_frames, num_objects,model,Mem_every=None, Mem_number=2) ##  THIS IS UPDATEEEEEEEDDDD
        # all_res_masks = Es[0].cpu().numpy()[1:1+num_objects]
        all_res_masks = np.zeros((num_objects,pred.shape[0],pred.shape[1],pred.shape[2]))
        for i in range(1,num_objects+1):
            all_res_masks[i-1,:,:,:] = (pred == i).astype(np.uint8)
        all_res_masks = all_res_masks[:, 1:, :, :]
        all_gt_masks = Ms[0][1:1+num_objects]
        all_gt_masks = all_gt_masks[:, :, :, :]
        j_metrics_res, f_metrics_res = evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                #print('>> J : ',j_metrics_res[ii])
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
            if 'F' in metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)

    J, F = metrics_res['J'], metrics_res['F']
    #print(len(J["M"]))
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])

    print("scores: [J&F-Mean, J-Mean, J-Recall, J-Decay, F-Mean, F-Recall, F-Decay] are: ",g_res)
    return g_res
	    



if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
        parser.add_argument("-s", type=str, help="set", required=True)
        parser.add_argument("-y", type=int, help="year", required=True)
        parser.add_argument("-D", type=str, help="path to data",default='/smart/haochen/cvpr/data/visor/')
        parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnest101']",default='resnet50')
        parser.add_argument("-p", type=str, help="path to weights",default='../visor_weights/coco_lr_fix_skip_0_1_release_resnet50_400000_32_399999.pth')
        return parser.parse_args()

    args = get_arguments()

    GPU = args.g
    YEAR = args.y
    SET = args.s
    DATA_ROOT = args.D

    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on VISOR')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    Testloader = VISOR_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    model = nn.DataParallel(STM(args.backbone))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    pth = args.p

    model.load_state_dict(torch.load(pth))
    metric = ['J','F']

    evaluate(model,Testloader,metric)
