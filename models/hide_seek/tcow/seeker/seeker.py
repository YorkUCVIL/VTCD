'''
Neural network architecture description.
Created by Basile Van Hoorick, Jun 2022.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'seeker/'))

from models.hide_seek.tcow.__init__ import *

# Internal imports.
import models.hide_seek.tcow.seeker.aot_wrapper as      aot_wrapper
import models.hide_seek.tcow.seeker.mask_tracker  as    mask_tracker
import models.hide_seek.tcow.seeker.point_tracker as    point_tracker


class Seeker(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger, **kwargs):
        super().__init__()
        self.logger = logger
        try:
            self.which_seeker = kwargs['which_seeker']
            self.tracker_arch = kwargs['tracker_arch']
            del kwargs['which_seeker']
        except:
            self.which_seeker = 'mask_track_2d'
            self.tracker_arch = 'timesformer'

        if self.tracker_arch == 'aot':
            # We are using a baseline.
            self.seeker = aot_wrapper.AOTWrapper(
                self.logger, num_frames=kwargs['num_total_frames'],
                frame_height=kwargs['frame_height'], frame_width=kwargs['frame_width'],
                max_frame_gap=kwargs['aot_max_gap'], aot_arch=kwargs['aot_arch'],
                pretrain_path=kwargs['aot_pretrain_path'])

            if kwargs['concept_clustering']:
                self.seeker.cluster_subject = kwargs['cluster_subject']
                self.seeker.cluster_memory = kwargs['cluster_memory']

        else:
            # We are using our own model.
            if 'aot_max_gap' in kwargs:
                del kwargs['aot_max_gap']
            if 'aot_arch' in kwargs:
                del kwargs['aot_arch']
            if 'aot_pretrain_path' in kwargs:
                del kwargs['aot_pretrain_path']
            
            if self.which_seeker == 'point_track_3d':
                del kwargs['patch_size']
                del kwargs['track_map_stride']
                del kwargs['query_channels']
                del kwargs['output_channels']
                self.seeker = point_tracker.QueryPointTracker(logger, **kwargs)

            elif self.which_seeker == 'mask_track_2d':
                try:
                    del kwargs['query_format']
                    del kwargs['query_type']
                    del kwargs['output_format']
                    del kwargs['output_token']
                    del kwargs['output_type']
                except:
                    pass
                self.seeker = mask_tracker.QueryMaskTracker(logger, **kwargs)

    def forward(self, *args):
        return self.seeker(*args)

    def set_phase(self, phase):
        '''
        Must be called when switching between train / validation / test phases.
        '''
        if self.tracker_arch == 'aot':
            self.seeker.set_phase(phase)


# class PerfectBaseline(torch.nn.Module):
#     '''
#     X
#     '''

#     def __init__(self, logger, which_baseline, **kwargs):
#         super().__init__()
#         self.logger = logger
#         self.which_baseline = which_baseline

#     def forward(self, *args):
#         if self.which_baseline == 'static':
#             return perfect_baseline.run_static_mask(self.logger, *args)
#         if self.which_baseline == 'linear':
#             return perfect_baseline.run_linear_extrapolation(self.logger, *args)


if __name__ == '__main__':

    import logvisgen

    logger = logvisgen.Logger()

    pass
