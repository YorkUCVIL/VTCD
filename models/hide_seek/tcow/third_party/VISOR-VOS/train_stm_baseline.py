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
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import random


### My libs
from dataset.dataset import VISOR_MO_Test
from dataset.visor import VISOR_MO_Train
from model.model import STM
#from eval_custome import evaluate
from eval import evaluate
from utils.helpers import overlay_davis



def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-Dvisor", type=str, help="path to data",default='../data/VISOR/')
    parser.add_argument("-resolution", type=str, help="resolution of the dataset",default='480p')
    parser.add_argument("-train_set_txt", type=str, help="name of the text file that contains the traning sequences",default='train')
    parser.add_argument("-val_set_txt", type=str, help="name of the text file that contains the validation sequences(for evaluation)",default='val')    
    parser.add_argument("-year", type=int, help="last 2 digits of the year of the dataset release",default=22)
    parser.add_argument("-batch", type=int, help="batch size",default=4)
    parser.add_argument("-wandb_logs", type=int, help="if you want the wandb logs to be saved",default=0)
    parser.add_argument("-max_skip", type=int, help="max skip between training frames",default=25)
    parser.add_argument("-change_skip_step", type=int, help="change max skip per x iter",default=3000)
    parser.add_argument("-total_iter", type=int, help="total iter num",default=800000)
    parser.add_argument("-test_iter", type=int, help="evaluate per x iters",default=10000)
    parser.add_argument("-log_iter", type=int, help="log per x iters",default=500)
    parser.add_argument("-resume_path",type=str,default='')
    parser.add_argument("-save",type=str,default='../weights')
    parser.add_argument("-name",type=str,default='experiment')
    parser.add_argument("-sample_rate",type=float,default=0.08)
    parser.add_argument("-backbone", type=str, help="backbone ['resnet50', 'resnet18','resnet101]",default='resnet50')

    return parser.parse_args()

args = get_arguments()

rate = args.sample_rate
resolution = args.resolution
year = args.year
train_set_txt = args.train_set_txt
val_set_txt = args.val_set_txt
wandb_logs = args.wandb_logs

DATA_ROOT = args.Dvisor
palette = Image.open(os.path.join(DATA_ROOT + 'Annotations/480p/P01_01_seq_00001/P01_01_frame_0000000140.png')).getpalette()

torch.backends.cudnn.benchmark = True

Trainset_sparse = VISOR_MO_Train(DATA_ROOT, resolution=resolution, imset='20{}/{}.txt'.format(year,train_set_txt), single_object=False)
Trainloader_sparse = data.DataLoader(Trainset_sparse, batch_size=1, num_workers=1,shuffle = True, pin_memory=True)
loader_iter_sparse = iter(Trainloader_sparse)

Testloader = VISOR_MO_Test(DATA_ROOT, resolution=resolution, imset='20{}/{}.txt'.format(year,val_set_txt), single_object=False)

model = nn.DataParallel(STM(args.backbone))
pth_path = args.resume_path



print('Loading weights:', pth_path)
print(args)
if pth_path != '':
    model.load_state_dict(torch.load(pth_path))

if torch.cuda.is_available():
    model.cuda()
model.train()
for module in model.modules():
    if isinstance(module, torch.nn.modules.BatchNorm1d):
        module.eval()
    if isinstance(module, torch.nn.modules.BatchNorm2d):
        module.eval()
    if isinstance(module, torch.nn.modules.BatchNorm3d):
        module.eval()

criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5,eps=1e-8, betas=[0.9,0.999])

def adjust_learning_rate(iteration,power = 0.9):
    #lr = 1e-5 * pow((1 - 1.0 * iteration / args.total_iter), power)
    lr = 1e-5 #for fixed lr
    return lr


accumulation_step = args.batch
save_step = args.test_iter
log_iter = args.log_iter

loss_momentum = 0
max_skip = 1
#change_skip_step = args.change_skip_step
change_skip_step = args.total_iter/(max_skip+1)
skip_n = 0
max_jf = 0

if wandb_logs:
    import wandb
    os.environ['WANDB_MODE'] = "offline"
    wandb.init(project="official_release",name=f"{args.name}",config={
          "batch": accumulation_step,
          "initialization": "coco",
          "architecture": "CNN",
          "split": "train=>val",
          "iterations": args.total_iter,
          "save_step":save_step,
          "max_skip":max_skip,
          "Backbone_resume":args.resume_path,
          "Name":args.name,} )


for iter_ in range(args.total_iter):
    if (iter_ == 0):
        print('Evaluate at iter: ' + str(iter_))
        g = evaluate(model,Testloader,['J','F'])
        print("J&F:",g[0])
        if wandb_logs:
            wandb.log({'iteration':iter_,'J&F-Mean':g[0], 'J-Mean':g[1], 'J-Recall':g[2], 'J-Decay':g[3], 'F-Mean':g[4], 'F-Recall':g[5], 'F-Decay':g[6]})


    if (iter_ + 1) % 1000 == 0:
        lr = adjust_learning_rate(iter_)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    if (iter_ + 1) % change_skip_step == 0:
        if skip_n < max_skip:
            skip_n += 1
        Trainset_sparse.change_skip(skip_n)
        loader_iter_sparse = iter(Trainloader_sparse)


    try:
        Fs, Ms, num_objects, info = next(loader_iter_sparse)
    except:
        loader_iter_sparse = iter(Trainloader_sparse)
        Fs, Ms, num_objects, info = next(loader_iter_sparse)
    
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    num_frames = 3

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    n1_key, n1_value = model(Fs[:,:,0], Es[:,:,0], torch.tensor([num_objects]))
    n2_logit = model(Fs[:,:,1], n1_key, n1_value, torch.tensor([num_objects]))

    n2_label = torch.argmax(Ms[:,:,1],dim = 1).long().cuda()
    n2_loss = criterion(n2_logit,n2_label)

    Es[:,:,1] = F.softmax(n2_logit, dim=1).detach()

    n2_key, n2_value = model(Fs[:,:,1], Es[:,:,1], torch.tensor([num_objects]))
    n12_keys = torch.cat([n1_key, n2_key], dim=3)
    n12_values = torch.cat([n1_value, n2_value], dim=3)
    n3_logit = model(Fs[:,:,2], n12_keys, n12_values, torch.tensor([num_objects]))


    n3_label = torch.argmax(Ms[:,:,2],dim = 1).long().cuda()
    n3_loss = criterion(n3_logit,n3_label)

    Es[:,:,2] = F.softmax(n3_logit, dim=1)

    loss = n2_loss + n3_loss
    # loss = loss / accumulation_step
    loss.backward()
    loss_momentum += loss.cpu().data.numpy()

    if (iter_+1) % accumulation_step == 0:
        optimizer.step()
        optimizer.zero_grad()

    if ((iter_+1) % log_iter == 0):
        print('iteration:{}, loss:{},remaining iteration:{}'.format(iter_,loss_momentum/log_iter,args.total_iter - iter_))
        if wandb_logs:
            wandb.log({'loss_iteration':iter_,"train_loss": loss_momentum/log_iter})
        loss_momentum = 0



    if (iter_+1) % save_step == 0:
    #if ((iter_+1) % save_step == 0) and (iter_+1) >= 200000:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        torch.save(model.state_dict(), os.path.join(args.save,'{}_{}_{}_{}_{}.pth'.format(args.name,args.backbone,str(args.total_iter),str(args.batch),str(iter_))))
        
        model.eval()
        
        print('Evaluate at iter: ' + str(iter_))
        g = evaluate(model,Testloader,['J','F'])
        print("J&F:",g[0])
        if wandb_logs:
            wandb.log({'iteration':iter_,'J&F-Mean':g[0], 'J-Mean':g[1], 'J-Recall':g[2], 'J-Decay':g[3], 'F-Mean':g[4], 'F-Recall':g[5], 'F-Decay':g[6]})


        #evaluate(model,Testloader1,['J','F'],'train')
        
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
               module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
               module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
               module.eval()
if wandb_logs:
    # Mark the run as finished
    wandb.finish()
