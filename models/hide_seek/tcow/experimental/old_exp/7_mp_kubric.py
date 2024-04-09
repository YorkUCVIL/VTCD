# BVH, Jun 2022.

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))

from __init__ import *

# Library imports
import tempfile

# Internal imports.
# import kubric_sim
import logvisgen


num_workers = 2


class MyWorkerObject:

    def __init__(self, logger, worker_idx):

        logger.info(f'Before imports')
    
        # Library imports.
        # NOTE: We CANNOT import bpy outside of the actual thread using it!
        import bpy
        import kubric as kb
        import kubric.simulator
        import kubric.renderer
    
        logger.info(f'After imports')

        self.logger = logger
        self.worker_idx = worker_idx

        self.logger.info(f'Instantiating scene...')
        self.scene = kb.Scene(frame_start=0, frame_end=8, frame_rate=12, resolution=(320, 240))

        self.kb_module = kb

    def run(self):

        for i in range(5):
            time.sleep(1.0)
            self.logger.info(f'Debug sleep {i + 1} / 5')


        for i in range(5):
            for j in range(2000000):
                k = j * j
                l = i / (k + 1)
            self.logger.info(f'Debug calcs {i + 1} / 5')

        self.logger.info(f'Instantiating simulator...')
        scratch_dir = tempfile.mkdtemp()
        self.simulator = self.kb_module.simulator.PyBullet(self.scene, scratch_dir)

        self.logger.info(f'Instantiating renderer...')
        self.renderer = self.kb_module.renderer.Blender(self.scene, scratch_dir)

        focal_length = 30.0 + self.worker_idx * 20.0
        self.scene.camera = self.kb_module.PerspectiveCamera(focal_length=focal_length, sensor_width=32.0)

        self.logger.info(f'Running simulator...')
        self.simulator.run(frame_start=0, frame_end=8)

        self.logger.info(f'Running renderer...')
        self.renderer.render()

        self.logger.info(f'Done!')


def worker(worker_idx):

    logger = logvisgen.Logger(msg_prefix=f'exp7_worker{worker_idx}')
    logger.info(f'Start worker...')
    
    # sleep_s = worker_idx * 5.0
    # print(f'sleeping {sleep_s} seconds before proceeding...')
    # time.sleep(sleep_s)

    # NOTE: This instance must only be created once per process!
    # my_kubric = kubric_sim.MyKubricSimulatorRenderer(
    #     logger, frame_width=320, frame_height=240, num_frames=24, frame_rate=12)

    # gso_source = kb.AssetSource.from_manifest('gs://kubric-public/assets/GSO/GSO.json')

    worker_instance = MyWorkerObject(logger, worker_idx)

    worker_instance.run()

    pass


class MyPipeline(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inds):
        within_batch_idx = inds[0].item()
        time.sleep(within_batch_idx)
        worker(within_batch_idx)


def main():

    if num_workers <= 0:

        worker(0)

    else:

        if 0:

            # NOTE: These are NOT actually separate processes -- I have to augment DataParallel somehow!

            device = torch.device('cuda')
            net = MyPipeline()
            net = net.to(device)
            net = torch.nn.DataParallel(net, device_ids=[0, 1])

            inds = torch.arange(0, num_workers, dtype=torch.int32).to(device)
            y = net(inds)

        else:

            processes = [mp.Process(target=worker, args=(worker_idx, ))
                        for worker_idx in range(num_workers)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

        # with mp.Pool(num_workers) as pool:
        #     pool.map(worker, list(range(num_workers)))

    pass

if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()

    pass
