# BVH, May 2022.
# Adapted from kubric/examples/helloworld.py.

# =================================
# Add this for every Kubric script:
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'third_party/kubric/'))

output_dp = 'experimental/kubric_output/'
os.makedirs(output_dp, exist_ok=True)

import matplotlib.pyplot as plt
import time

# =================================

import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

# --- create scene and attach a renderer to it
# NOTE: height and width must be same for some reason, crashes otherwise.
scene = kb.Scene(resolution=(480, 480))
renderer = KubricRenderer(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(10, 10, 0.1), position=(0, 0, -0.1))
scene += kb.Sphere(name="ball", scale=1, position=(0, 0, 1.))
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
scene += kb.PerspectiveCamera(name="camera", position=(3, -1, 4),
                              look_at=(0, 0, 1))

# --- render (and save the blender file)
renderer.save_state(output_dp + "/helloworld.blend")
frame = renderer.render_still()

# --- save the output as pngs
kb.write_png(frame["rgba"], output_dp + "/helloworld.png")
kb.write_palette_png(frame["segmentation"], output_dp + "/helloworld_segmentation.png")
kb.write_png(frame["object_coordinates"], output_dp + "/helloworld_objcoord.png")
scale = kb.write_scaled_png(frame["depth"], output_dp + "/helloworld_depth.png")
logging.info("Depth scale: %s", scale)

# NOTE:
# frame keys:
# ['rgba', 'backward_flow', 'forward_flow', 'depth', 'normal', 'object_coordinates', 'segmentation']
# frame values shapes:
# [(480, 480, 4), (480, 480, 2), (480, 480, 2), (480, 480, 1), (480, 480, 3), (480, 480, 3), (480, 480, 1)]
# frame values dtypes:
# uint8, float32, float32, float32, uint16, uint16, uint32.

if 1:
    logging.info('Showing figure...')
    plt.figure()
    plt.imshow(frame['rgba'] / 255.0)
    plt.show()

logging.info('Starting infinite sleep loop...')
while True:
    time.sleep(1.0)

