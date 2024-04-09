'''
BVH, Sep 2022.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'experimental/'))
sys.path.insert(0, os.path.join(os.getcwd(), 'third_party/aot-benchmark/'))
sys.path.insert(0, os.getcwd())

from __init__ import *

# Library imports.
import numpy as np
import torch

# External imports.
from configs.default import DefaultEngineConfig
from networks.engines.aot_engine import AOTEngine, AOTInferEngine
from networks.models.aot import AOT


def main():

    exp_name = 'stub'
    model_name = 'aotb'  # 'aott'
    gpu_id = 0
    # (H, W) = (256, 256)  # works
    (H, W) = (224, 288)  # works

    # ================================================================
    # See also HPOL C:\Users\Basile\Repos\aot-benchmark for debugging.
    
    # NOTE: I verified that at train time, cfg is the same here and from their train.py
    # if --model is aott.
    cfg = DefaultEngineConfig(exp_name=exp_name, model=model_name)
    model = AOT(cfg, encoder=cfg.MODEL_ENCODER)
    engine_train = AOTEngine(model, gpu_id=gpu_id)

    # ======================================
    # See trainer.py for model forward args.
    B = 1
    T = 10
    all_frames = torch.rand(T * B, 3, H, W)
    all_labels = torch.rand(T * B, 1, H, W)
    obj_nums = [1]

    engine_train.restart_engine(batch_size=B, enable_id_shuffle=False)
    (loss, all_pred_mask, all_frame_loss, boards) = \
        engine_train(all_frames, all_labels, B, obj_nums)
    
    # loss = (1) tensor.
    # all_pred_mask = list-T of (B, 1, H, W) tensors.
    # all_frame_los = (T) tensor.

    # ========================================
    # See evaluator.py for model forward args.
    # cfg and model can remain the same!
    engine_test = AOTInferEngine(model, gpu_id=gpu_id)

    first_image = torch.rand(1, 3, H, W)
    first_label = torch.rand(1, 1, H, W)
    next_image = torch.rand(1, 3, H, W)
    obj_nums = [1]

    engine_test.restart_engine()
    engine_test.add_reference_frame(first_image, first_label, obj_nums, frame_step=0)
    engine_test.match_propogate_one_frame(next_image)

    pred_logits = engine_test.decode_current_logits()  # (1, 11, H/4, W/4) tensor.
    pred_probs = torch.softmax(pred_logits, dim=1)  # (1, 11, H/4, W/4) tensor.

    pred_logits_os = engine_test.decode_current_logits(output_size=(H, W))  # (1, 11, H, W) tensor.
    pred_probs_os = torch.softmax(pred_logits_os, dim=1)  # (1, 11, H, W) tensor.

    pass


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()

    pass
