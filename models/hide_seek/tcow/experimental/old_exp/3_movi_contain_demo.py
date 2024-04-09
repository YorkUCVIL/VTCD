# BVH, Jun 2022.
'''
This is mostly just a copy of movi_def_worker.py.

Modifications:
- Force one object to be container.
- Force another object to fall above the container.

conda activate kubric
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_1/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_2/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_3/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_4/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_5/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_6/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_7/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_8/
python experimental/3_movi_contain_demo.py --output_dir tk_output/e3_v24_9/
'''

import imageio
import logging
import time

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np

# MOD:
import os


def main():

    os.environ["KUBRIC_USE_GPU"] = "1"
    os.environ['NO_GCE_CHECK'] = 'true'

    # --- CLI arguments
    parser = kb.ArgumentParser()

    # BVH MOD
    parser.add_argument("--output_dir", type=str, default='tk_output/e3_v25/')
    parser.add_argument("--samples_per_pixel", type=int, default=64)

    parser.add_argument("--objects_split", choices=["train", "test"],
                        default="train")
    # Configuration for the objects of the scene
    parser.add_argument("--min_num_static_objects", type=int, default=8,
                        help="minimum number of static (distractor) objects")
    parser.add_argument("--max_num_static_objects", type=int, default=12,
                        help="maximum number of static (distractor) objects")
    parser.add_argument("--min_num_dynamic_objects", type=int, default=4,
                        help="minimum number of dynamic (tossed) objects")
    parser.add_argument("--max_num_dynamic_objects", type=int, default=6,
                        help="maximum number of dynamic (tossed) objects")

    # Configuration for the floor and background
    parser.add_argument("--floor_friction", type=float, default=0.3)
    parser.add_argument("--floor_restitution", type=float, default=0.5)
    parser.add_argument("--backgrounds_split", choices=["train", "test"],
                        default="train")

    parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                        default="linear_movement")
    parser.add_argument("--max_camera_movement", type=float, default=4.0)

    # BVH MOD
    parser.add_argument("--min_motion_blur", type=float, default=1.4)
    parser.add_argument("--max_motion_blur", type=float, default=1.8)

    # Configuration for the source of the assets
    parser.add_argument("--kubasic_assets", type=str,
                        default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--hdri_assets", type=str,
                        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.add_argument("--gso_assets", type=str,
                        default="gs://kubric-public/assets/GSO/GSO.json")
    parser.add_argument("--save_state", dest="save_state", action="store_true")
    parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                        resolution=384)
    FLAGS = parser.parse_args()

    runtimes = []
    num_trials = 3
    output_dir_root = FLAGS.output_dir
    simulator = None

    # Start actual main process without any flags modification.
    (cur_runtime, simulator) = main_with_flags(FLAGS, simulator)
    runtimes.append(cur_runtime)

    # ====================================================
    # Write my own for loops here if measurements desired.
    # Calculate relative slopes with respect to default flags (middle out of 3).

    # Sweep over number of rays per pixel for rendering.
    # 2.679589416980743408e+02 2.670840220451354980e+02 2.707265882492065430e+02
    # 2.961382904052734375e+02 2.977207012176513672e+02 2.967928049564361572e+02
    # 3.554069900512695312e+02 3.498012995719909668e+02 3.301886632442474365e+02
    # Averages: 269, 297, 345.
    # => Relative slope: ~0.26.
    
    # FLAGS = parser.parse_args()
    # for samples_per_pixel in [32, 64, 96]:
    #     FLAGS.frame_end = 24
    #     FLAGS.resolution = 384
    #     FLAGS.min_num_static_objects = 10
    #     FLAGS.max_num_static_objects = 10
    #     FLAGS.min_num_dynamic_objects = 5
    #     FLAGS.max_num_dynamic_objects = 5
    #     FLAGS.samples_per_pixel = samples_per_pixel

    #     for trial in range(num_trials):
    #         FLAGS.output_dir = os.path.join(
    #             output_dir_root, f'spp{samples_per_pixel}_{trial}')

    #         (cur_runtime, simulator) = main_with_flags(FLAGS, simulator)
    #         runtimes.append(cur_runtime)


    # Sweep over number of objects.
    # 2.558057360649108887e+02 2.272159004211425781e+02 2.323519270420074463e+02
    # 2.788464887142181396e+02 2.929441375732421875e+02 2.701889410018920898e+02
    # 3.255843584537506104e+02 3.084642598628997803e+02 3.114447762966156006e+02
    # Averages (horizontal, seconds): 239, 281, 316.
    # => Relative slope: ~0.27.

    # FLAGS = parser.parse_args()
    # for num_static, num_dynamic in [(4, 2), (8, 4), (12, 6)]:
    #     FLAGS.frame_end = 24
    #     FLAGS.resolution = 384
    #     FLAGS.min_num_static_objects = num_static
    #     FLAGS.max_num_static_objects = num_static
    #     FLAGS.min_num_dynamic_objects = num_dynamic
    #     FLAGS.max_num_dynamic_objects = num_dynamic
    #     FLAGS.samples_per_pixel = 64

    #     for trial in range(num_trials):
    #         FLAGS.output_dir = os.path.join(
    #             output_dir_root, f's{num_static}_d{num_dynamic}_{trial}')

    #         (cur_runtime, simulator) = main_with_flags(FLAGS, simulator)
    #         runtimes.append(cur_runtime)


    # Sweep over video clip duration.
    # 1.827879734039306641e+02 1.899323494434356689e+02 1.847229351997375488e+02
    # 3.103555872440338135e+02 3.075283157825469971e+02 3.165390470027923584e+02
    # 4.130508532524108887e+02 4.343436617851257324e+02 4.063926098346710205e+02
    # Averages: 186, 312, 418.
    # => Relative slope: ~0.74.

    # FLAGS = parser.parse_args()
    # for frame_end in [12, 24, 36]:
    #     FLAGS.frame_end = frame_end
    #     FLAGS.resolution = 384
    #     FLAGS.min_num_static_objects = 10
    #     FLAGS.max_num_static_objects = 10
    #     FLAGS.min_num_dynamic_objects = 5
    #     FLAGS.max_num_dynamic_objects = 5
    #     FLAGS.samples_per_pixel = 64

    #     for trial in range(num_trials):
    #         FLAGS.output_dir = os.path.join(output_dir_root, f'e{frame_end}_{trial}')

    #         (cur_runtime, simulator) = main_with_flags(FLAGS, simulator)
    #         runtimes.append(cur_runtime)


    # Sweep over resolution.
    # 2.389830605983734131e+02 2.593999691009521484e+02 2.345184943675994873e+02
    # 2.845242037773132324e+02 3.070004894733428955e+02 2.965885193347930908e+02
    # 3.700811185836791992e+02 3.943258976936340332e+02 3.708914804458618164e+02
    # Averages: 244, 296, 378.
    # => Relative slope: ~0.45.

    # FLAGS = parser.parse_args()
    # for resolution in [256, 384, 512]:
    #     FLAGS.frame_end = 24
    #     FLAGS.resolution = resolution
    #     FLAGS.min_num_static_objects = 10
    #     FLAGS.max_num_static_objects = 10
    #     FLAGS.min_num_dynamic_objects = 5
    #     FLAGS.max_num_dynamic_objects = 5
    #     FLAGS.samples_per_pixel = 64

    #     for trial in range(num_trials):
    #         FLAGS.output_dir = os.path.join(output_dir_root, f'r{resolution}_{trial}')

    #         (cur_runtime, simulator) = main_with_flags(FLAGS, simulator)
    #         runtimes.append(cur_runtime)

    runtimes = np.array(runtimes).reshape(-1, num_trials)

    np.savetxt('experimental/3_kubric_runtimes_v3.txt', runtimes)

    kb.done()

    pass


def main_with_flags(FLAGS, simulator):

    main_start_time = time.time()

    # --- Some configuration values
    # the region in which to place objects [(min), (max)]
    STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 10)]
    DYNAMIC_SPAWN_REGION = [(-4, -4, 2), (4, 4, 6)]
    VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]

    # --- Common setups & resources
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

    # BVH MOD
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)

    motion_blur = rng.uniform(FLAGS.min_motion_blur, FLAGS.max_motion_blur)
    if motion_blur > 0.0:
        logging.info(f"Using motion blur strength {motion_blur}")

    # simulator_first = PyBullet(scene, scratch_dir)
    if simulator is None:
        simulator = PyBullet(scene, scratch_dir)
    else:
        logging.info(f'Swapping old with new scene in existing PyBullet simulator instance...')
        simulator.scene = scene
        simulator.scratch_dir = scratch_dir

    renderer = Blender(
        scene, scratch_dir, use_denoising=True, samples_per_pixel=FLAGS.samples_per_pixel,
        motion_blur=motion_blur)
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    # --- Populate the scene
    # background HDRI
    start_time = time.time()
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    if FLAGS.backgrounds_split == "train":
        logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
        hdri_id = rng.choice(train_backgrounds)
    else:
        logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
        hdri_id = rng.choice(test_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    #assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                          friction=1.0,
                          restitution=0.0,
                          static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    def get_linear_camera_motion_start_end(
            movement_speed: float,
            inner_radius: float = 10.,
            outer_radius: float = 12.,
            z_offset: float = 0.1,
    ):
        """Sample a linear path which starts and ends within a half-sphere shell."""
        while True:
            camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                         outer_radius,
                                                                         z_offset))
            direction = rng.rand(3) - 0.5
            movement = direction / np.linalg.norm(direction) * movement_speed
            camera_end = camera_start + movement
            if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                    camera_end[2] > z_offset):
                return camera_start, camera_end

    # Camera
    logging.info("Setting up the Camera...")
    start_time = time.time()
    scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
    if FLAGS.camera == "fixed_random":
        scene.camera.position = kb.sample_point_in_half_sphere_shell(
            inner_radius=7., outer_radius=9., offset=0.1)
        scene.camera.look_at((0, 0, 0))
    elif FLAGS.camera == "linear_movement":
        camera_start, camera_end = get_linear_camera_motion_start_end(
            movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement)
        )
        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames
        for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
            interp = ((frame - FLAGS.frame_start + 1) /
                      (FLAGS.frame_end - FLAGS.frame_start + 3))
            scene.camera.position = (interp * np.array(camera_start) +
                                     (1 - interp) * np.array(camera_end))
            scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # ---- Object placement ----
    train_split, test_split = gso.get_test_split(fraction=0.1)
    if FLAGS.objects_split == "train":
        logging.info("Choosing one of the %d training objects...", len(train_split))
        active_split = train_split
    else:
        logging.info("Choosing one of the %d held-out objects...", len(test_split))
        active_split = test_split

    # BVH MOD
    # force basket first
    logging.info('Inserting pre-chosen container...')
    start_time = time.time()

    # Used for e3_v1 - e3_v15:
    # This takes ~1.5 seconds.
    my_obj = gso.create(asset_id="Target_Basket_Medium")
    scale = rng.uniform(8.0, 12.0)
    # Used for e3_v16 - e3_v19:
    # my_obj = gso.create(asset_id="Utana_5_Porcelain_Ramekin_Large")
    # scale = rng.uniform(16.0, 24.0)
    # Used for e3_v20 - e3_v23:
    # my_obj = gso.create(asset_id="Sapota_Threshold_4_Ceramic_Round_Planter_Red")
    # scale = rng.uniform(12.0, 18.0)

    my_obj.scale = scale
    my_obj.metadata['scale'] = scale
    my_obj.position = np.array([0.0, 0.0, 0.6], dtype=np.float32)
    scene += my_obj  # This takes ~4 seconds.
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')
    # kb.move_until_no_overlap(my_obj, simulator, spawn_region=STATIC_SPAWN_REGION,
    #                         rng=rng)

    # add STATIC objects
    # This takes up to a minute.
    num_static_objects = rng.randint(FLAGS.min_num_static_objects,
                                     FLAGS.max_num_static_objects+1)
    logging.info("Randomly placing %d static objects:", num_static_objects)
    start_time = time.time()
    for i in range(num_static_objects):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 3.0)
        # make all objects of similar size
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale"] = scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=STATIC_SPAWN_REGION,
                                 rng=rng)
        obj.friction = 1.0
        obj.restitution = 0.0
        obj.metadata["is_dynamic"] = False
        logging.info("    Added %s at %s", obj.asset_id, obj.position)
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # This takes ~6 seconds.
    logging.info("Running 50 frames of simulation to let static objects settle...")
    start_time = time.time()
    _, _ = simulator.run(frame_start=-48, frame_end=0)
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # stop any objects that are still moving and reset friction / restitution
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0., 0., 0.)
            obj.friction = 0.5
            obj.restitution = 0.5

    dome.friction = FLAGS.floor_friction
    dome.restitution = FLAGS.floor_restitution

    # BVH MOD
    logging.info("Running 2 frames of simulation to gather hider input...")
    start_time = time.time()
    _, _ = simulator.run(frame_start=0, frame_end=2)
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # BVH MOD
    logging.info("Rendering initial scene for hider input...")
    start_time = time.time()
    scene.frame_start = 0
    scene.frame_end = 1
    data_stack_hider = renderer.render()
    scene.frame_start = 2
    scene.frame_end = FLAGS.frame_end
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # TODX our framework should utilize data_stack_hider when proceeding

    # Add DYNAMIC objects
    num_dynamic_objects = rng.randint(FLAGS.min_num_dynamic_objects,
                                      FLAGS.max_num_dynamic_objects+1)
    logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
    start_time = time.time()
    for i in range(num_dynamic_objects):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 2.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale"] = scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION,
                                 rng=rng)
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
                        [obj.position[0], obj.position[1], 0])
        obj.metadata["is_dynamic"] = True
        logging.info("    Added %s at %s", obj.asset_id, obj.position)
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    if FLAGS.save_state:
        logging.info("Saving the simulator state to '%s' prior to the simulation.",
                     output_dir / "scene.bullet")
        start_time = time.time()
        simulator.save_state(output_dir / "scene.bullet")
        logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # Run dynamic objects simulation
    logging.info("Running actual simulation...")
    start_time = time.time()
    animation, collisions = simulator.run(frame_start=2,
                                          frame_end=scene.frame_end+1)
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # --- Rendering
    if FLAGS.save_state:
        logging.info("Saving the renderer state to '%s' ",
                     output_dir / "scene.blend")
        renderer.save_state(output_dir / "scene.blend")

    logging.info("Rendering actual scene ...")
    start_time = time.time()
    data_stack_rest = renderer.render()
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # BVH MOD
    # Merge before and after hider policy action rendered data.
    # data_stack = {k: np.concatenate(
    #     [data_stack_hider[k], data_stack_rest[k]], axis=0)
    #     for k in data_stack_hider.keys()}

    # NOTE: Renderer seems to remember and prepend previous results already.
    data_stack = data_stack_rest

    # BVH MOD
    # Write video + gif first.
    # imageio.help()
    imageio.mimwrite(os.path.join(output_dir, '0_rgb.mp4'), data_stack['rgba'][..., :3],
                     format='ffmpeg', fps=FLAGS.frame_rate, quality=10)
    #  ffmpeg_params=["-pix_fmt", "yuv420p"])
    imageio.mimwrite(os.path.join(output_dir, '0_rgb.gif'), data_stack['rgba'][..., :3],
                     format='gif', fps=FLAGS.frame_rate)

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                 if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        visible_foreground_assets)
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    kb.write_image_dict(data_stack, output_dir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                      visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')
    kb.write_json(filename=os.path.join(output_dir, "metadata.json"), data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=os.path.join(output_dir, "events.json"), data={
        "collisions":  kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })
    logging.info(f'Took {(time.time() - start_time) * 1000.0:.2f}ms')

    # kb.done()

    total_runtime_s = time.time() - main_start_time
    logging.info(f'Total runtime {total_runtime_s:.2f}s')

    return total_runtime_s, simulator


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)

    main()
