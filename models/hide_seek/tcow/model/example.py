'''
Neural network architecture description.
'''

from __init__ import *

# Internal imports.
import my_utils
# import perceiver
import vision_tf


class MySimpleModel(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger):
        '''
        X
        '''
        super().__init__()
        self.logger = logger

        self.net = torch.nn.Conv2d(3, 3, 1)

    def forward(self, rgb_input):
        '''
        :param rgb_input (B, 3, Hi, Wi) tensor.
        :return rgb_output (B, 3, Hi, Wi) tensor.
        '''
        rgb_output = self.net(rgb_input)

        return rgb_output


class MyDenseVitModel(torch.nn.Module):

    def __init__(self, logger, image_height, image_width, in_channels, out_channels):
        super().__init__()
        self.logger = logger
        self.image_height = image_height
        self.image_width = image_width
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.backbone = vision_tf.MyDenseVisionTransformerBackbone(
            logger, image_height, image_width, in_channels)

        self.post_proj = torch.nn.Linear(self.backbone.output_feature_dim,
            self.backbone.ho * self.backbone.wo * self.out_channels)

    def forward(self, rgb_input):
        '''
        :param rgb_input (B, Ci, Hi, Wi) tensor.
        :return rgb_output (B, Co, Hi, Wi) tensor.
        '''
        # rgb_input = (B, 3, 224, 288).
        
        embs_output = self.backbone(rgb_input)  # (B, D, H, W) = (B, 768, 14, 18).
        
        embs_output = rearrange(embs_output, 'B D H W -> B H W D')
        
        rgb_output = self.post_proj(embs_output)  # (B, H, W, D) = (B, 768, 14, 18).
        
        rgb_output = rearrange(rgb_output, 'B H W (h w C) -> B C (H h) (W w)',
                                h=self.backbone.ho, w=self.backbone.wo, C=self.out_channels)
        
        # rgb_output = (B, Co, Hi, Wi) = (B, 3, 224, 288).

        return rgb_output


class MyPerceiverModel(torch.nn.Module):

    def __init__(self, logger, image_height, image_width, in_channels, out_channels):
        super().__init__()
        self.logger = logger
        self.image_height = image_height
        self.image_width = image_width
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.backbone = perceiver.MyPerceiverBackbone(
            logger, (image_height, image_width), in_channels, (image_height, image_width),
            out_channels, -1, 'fourier')

    def forward(self, rgb_input):
        '''
        :param rgb_input (B, Ci, Hi, Wi) tensor.
        :return rgb_output (B, Co, Hi, Wi) tensor.
        '''
        # rgb_input = (B, 3, 224, 288).
        (rgb_output, last_hidden_state) = self.backbone(rgb_input)
        # rgb_output = (B, Co, Hi, Wi) = (B, 3, 224, 288).
        # last_hidden_state = (B, N, D) ???

        return rgb_output
