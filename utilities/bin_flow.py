import numpy as np
from torchvision.utils import flow_to_image
import torch
from PIL import Image
import matplotlib.pyplot as plt

def get_flow_names(magnitude_bins=1, angle_bins=4):
    if magnitude_bins == 3 and angle_bins == 8:
        # english names for the 25 directions of optical flow
        flow_names = {
            11: 'rightupslow',
            12: 'rightupmedium',
            13: 'rightupfast',
            21: 'uprightslow',
            22: 'uprightmedium',
            23: 'uprightfast',
            31: 'upleftslow',
            32: 'upleftmedium',
            33: 'upleftfast',
            41: 'leftupslow',
            42: 'leftupmedium',
            43: 'leftupfast',
            51: 'leftdownslow',
            52: 'leftdownmedium',
            53: 'leftdownfast',
            61: 'downleftslow',
            62: 'downleftmedium',
            63: 'downleftfast',
            71: 'downrightslow',
            72: 'downrightmedium',
            73: 'downrightfast',
            81: 'rightdownslow',
            82: 'rightdownmedium',
            83: 'rightdownfast',
        }
    elif magnitude_bins == 1 and angle_bins == 4:
        # english names for the 25 directions of optical flow
        flow_names = {
            0: 'still',
            1: 'right',
            11: 'up',
            21: 'left',
            31: 'down'
        }

    elif magnitude_bins == 2 and angle_bins == 4:
        # english names for the 25 directions of optical flow
        flow_names = {
            11: 'up-slow',
            21: 'left-slow',
            31: 'down-slow',
            41: 'right-slow',
            12: 'up-fast',
            22: 'left-fast',
            32: 'down-fast',
            42: 'right-fast'
        }

    else:
        print('Number of flow bins and mags not implemented')
        exit()

    return flow_names

def bin_flow(flow, magnitude_bins=1, angle_bins=4):

    '''
    Input: path to optical flow saved as npy file with shape 2xHxW (x-y pixel-wise flow)
    Output: HxW binned flow map where 1x,2x,3x,...8x corresponds to 45 angle partitions (starting from +ve x axis)
            and the ones column corresponds to magnitudes of flow with bins [0, 0.5, 1, 2]
    '''

    if magnitude_bins == 3 and angle_bins == 8:
        magnitude_bins = [0.0, 0.5, 1.0, 2]
        angle_bins = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 6 * np.pi / 4,
                  7 * np.pi / 4, 2 * np.pi]

    elif magnitude_bins == 1 and angle_bins == 4:
        clip_factor = 4
        magnitude_bins = [0.0, 10]
        angle_bins = [45, 135, 225]

    magnitude_bins = np.array(magnitude_bins)
    angle_bins = np.array(angle_bins)

    # 1) convert to magnitude and angle
    x = flow[0]
    y = flow[1]
    r = np.sqrt(x ** 2 + y ** 2)

    # get angle in range [0, 2pi]
    theta_rad = np.arctan2(y, x)
    theta = (theta_rad * 180 / np.pi) + 135
    # theta = np.where(theta < 0, 2 * np.pi + theta, theta)

    # if theta.min() < 0:
    # theta = (np.pi + theta)

    # 2) bin based on angle and magnitude
    binned_magnitude = np.digitize(r, magnitude_bins) - 1
    binned_angle = np.digitize(theta, angle_bins).clip(0,clip_factor)

    # Compute binned flow by base 10 for each angle, +1 for each magnitude level. e.g., dir-mag Output values: [0,11,12,13,21,22,...,81,82,83]
    binned_flow = np.where(binned_magnitude == 0, 0, binned_angle * 10 + binned_magnitude)
    return binned_flow



def debug_flow_bin(verbose=True):
    flow_paths = [
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c1/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c2/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c3/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c4/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c5/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c7/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c10/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c11/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c12/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c13/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c14/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c15/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c16/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c17/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c1/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c2/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c3/000050_flow.npy',
        '/home/m2kowal/data/DTDB/frames/Wavy_motion_g1_c4/000050_flow.npy',
                  ]
    for flow_path in flow_paths:
        flow = np.load(flow_path)



        # 1) convert to magnitude and angle
        x = flow[0]
        y = flow[1]

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, 2 * np.pi + theta, theta)

        # 2) bin based on angle (8 bins) and magnitude (3 bins)
        magnitude_bins = np.array([0.0, 0.5, 1.0, 2])
        angle_bins = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 6*np.pi/4, 7*np.pi/4, 2*np.pi])
        binned_magnitude = np.digitize(r, magnitude_bins)-1
        binned_angle = np.digitize(theta, angle_bins)

        # Compute binned flow by base 10 for each angle, +1 for each magnitude level. Output values: [0,11,12,13,21,22,...,81,82,83]
        binned_flow = np.where(binned_magnitude == 0, 0, binned_angle * 10 + binned_magnitude)
        print(binned_flow.shape)

        if verbose:
            torch_flow = torch.tensor(flow)
            flow_imgs = flow_to_image(torch_flow)

            img_path = flow_path.replace('_flow.npy', '.png')
            img = Image.open(img_path)

            plt.imshow(img)
            plt.show()
            plt.imshow(flow_imgs.permute(1,2,0))
            plt.show()


# debug_flow_bin()

# flow = bin_flow('/home/m2kowal/data/DTDB/frames/Rotary_motion_g5_c8/000050_flow.npy')
# print(flow)