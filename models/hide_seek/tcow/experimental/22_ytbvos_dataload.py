'''
BVH, Sep 2022.
'''

import os
import sys
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/aot-benchmark/'))

from __init__ import *

# Library imports.
import numpy as np
from torchvision import transforms
import dataloaders.video_transforms as tr

# External imports.
from configs.default import DefaultEngineConfig
from dataloaders.train_datasets import YOUTUBEVOS_Train


def main():

    exp_name = 'stub'
    model_name = 'aott'
    
    # NOTE: I verified that cfg here is exactly the same as the train_shapes command
    # (except for batch_size & some missing DIRs).
    cfg = DefaultEngineConfig(exp_name=exp_name, model=model_name)
    cfg.DIR_YTB = r'C:\Development\CVR Data\YouTube-VOS 2019'
    
    # See trainer.py for init args.
    composed_transforms = transforms.Compose([
        tr.RandomScale(cfg.DATA_MIN_SCALE_FACTOR,
                        cfg.DATA_MAX_SCALE_FACTOR, cfg.DATA_SHORT_EDGE_LEN),
        tr.BalancedRandomCrop(cfg.DATA_RANDOMCROP,
                                max_obj_num=cfg.MODEL_MAX_OBJ_NUM),
        tr.RandomHorizontalFlip(cfg.DATA_RANDOMFLIP),
        tr.Resize(cfg.DATA_RANDOMCROP, use_padding=True),
        tr.ToTensor()
    ])
    
    train_ytb_dataset = YOUTUBEVOS_Train(
        root=cfg.DIR_YTB,
        transform=composed_transforms,
        seq_len=cfg.DATA_SEQ_LEN,
        rand_gap=cfg.DATA_RANDOM_GAP_YTB,
        rand_reverse=cfg.DATA_RANDOM_REVERSE_SEQ,
        merge_prob=cfg.DATA_DYNAMIC_MERGE_PROB,
        enable_prev_frame=cfg.TRAIN_ENABLE_PREV_FRAME,
        max_obj_n=cfg.MODEL_MAX_OBJ_NUM)

    for i, example in enumerate(train_ytb_dataset):
        # keys: ['ref_img', 'prev_img', 'curr_img', 'ref_label', 'prev_label', 'curr_label', 'meta']
        # shapes: (3, 465, 465), (3, 465, 465), list-3, (1, 465, 465), (1, 465, 465), list-3, dict.
        # meta example: {'seq_name': '003234408d', 'frame_num': 36, 'obj_num': 5}
        print(i)
        print(example)
        if i >= 4:
            break

    # Conclusion: I will have to write my own dataloader for this dataset.

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()

    pass
