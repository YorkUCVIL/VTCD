# BVH, Aug 2022.

# https://github.com/facebookresearch/3detr

# NOTE: Have to install pointnet2 in third_party to get this to work.

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/fr-3detr/'))

from __init__ import *

# Library imports.

# Internal imports.
import datasets.scannet
import main
import models.model_3detr


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    device = torch.device('cuda:0')

    # See main.py.
    parser = main.make_args_parser()
    args = parser.parse_args(['--dataset_name', 'scannet'])
    dataset_config = datasets.scannet.ScannetDatasetConfig()

    # See model_3detr.py.
    model, _ = models.model_3detr.build_3detr(args, dataset_config)
    model = model.to(device)

    # See scannet.py.
    # NOTE: N must be > preenc_npoints = 2048.
    # NOTE: CPU not supported for FPS in pointnet2.
    N = 4096
    point_cloud = np.random.randn(N, 3).astype(np.float32) * 4.0
    point_cloud_dims_min = point_cloud.min(axis=0)[:3]
    point_cloud_dims_max = point_cloud.max(axis=0)[:3]
    inputs = {
        "point_clouds": torch.tensor(point_cloud).unsqueeze(0).to(device),
        "point_cloud_dims_min": torch.tensor(point_cloud_dims_min).unsqueeze(0).to(device),
        "point_cloud_dims_max": torch.tensor(point_cloud_dims_max).unsqueeze(0).to(device),
    }
    
    # See model_3detr.py.
    outputs = model(inputs)

    print('inputs:', stmmm(inputs))
    print('results:', stmmm(outputs))

    pass
