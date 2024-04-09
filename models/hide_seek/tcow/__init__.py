'''
These imports are shared across all files.
Created by Basile Van Hoorick, Jun 2022.
'''

# Library imports.
import argparse
import collections
import collections.abc
import copy
import cv2
import imageio
import itertools
import matplotlib.colors
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import platform
# import pytorch3d  # problematic to install on windows.
import random
import scipy
import seaborn as sns
import shutil
import sklearn
import sklearn.decomposition
import sys
import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.io
import torchvision.models
import torchvision.transforms
import torchvision.utils
import tqdm
import warnings
from collections import defaultdict
from einops import rearrange, repeat

PROJECT_NAME = 'hide-seek2'

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'data/'))
sys.path.append(os.path.join(os.getcwd(), 'eval/'))
sys.path.append(os.path.join(os.getcwd(), 'hider/'))
sys.path.append(os.path.join(os.getcwd(), 'model/'))
sys.path.append(os.path.join(os.getcwd(), 'seeker/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/'))
# sys.path.append(os.path.join(os.getcwd(), 'utilities/'))
sys.path.insert(0, os.path.join(os.getcwd(), 'third_party/aot-benchmark/'))

# Quick functions for usage during debugging:

def mmm(x):
    return (x.min(), x.mean(), x.max())


def st(x):
    return (x.dtype, x.shape)


def stmmm(x):
    return (*st(x), *mmm(x))
