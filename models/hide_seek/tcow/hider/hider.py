'''
Neural network architecture description.
Created by Basile Van Hoorick, Jun 2022.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'hider/'))

from __init__ import *

# Internal imports.
import hider_snitch
import my_utils
# import perceiver
import vision_tf


class Hider(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger, which_hider, **kwargs):
        '''
        X
        '''
        super().__init__()
        self.logger = logger
        self.which_hider = which_hider
        if which_hider == 'snitch':
            self.hider = hider_snitch.SnitchPolicy(logger, **kwargs)

    def forward(self, *args):
        '''
        X
        '''
        return self.hider(*args)


if __name__ == '__main__':

    import logvisgen

    logger = logvisgen.Logger()

    pass
