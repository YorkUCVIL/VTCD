# BVH, Aug 2022.

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))

from __init__ import *

import PIL


def main():

    src_dp = '/home/basilevanhoorick/Documents/Local Hide & Seek/Figures/Constituents/kb_depth/'
    src_fns = ['f1_depth_00000.tiff', 'f1_depth_00004.tiff', 'f3_depth_00018.tiff']

    for src_fn in src_fns:

        src_fp = os.path.join(src_dp, src_fn)
        dst_fp = os.path.join(src_dp, src_fn.replace('.tiff', '.png'))

        depth = np.array(PIL.Image.open(src_fp))  # (H, W) floats.
        min_val = np.min(depth)
        max_val = np.max(depth)
        depth = (depth - min_val) / (max_val - min_val + 1e-6)
        depth = 1.0 / (depth * 2.0 + 1.0)

        cmap = plt.get_cmap('plasma')
        depth_rgb = cmap(depth)[..., :3]  # (H, W, 3) floats.
        
        plt.imsave(dst_fp, depth_rgb)


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()

    pass
