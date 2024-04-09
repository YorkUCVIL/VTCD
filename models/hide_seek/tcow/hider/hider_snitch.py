'''
Neural network architecture description.
Created by Basile Van Hoorick, Jun 2022.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'seeker/'))

from __init__ import *

# Internal imports.
import my_utils
# import perceiver
import resnet
import vision_tf


class SnitchPolicy(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger, num_frames=2, frame_height=224, frame_width=288,
                 action_space=16, policy_arch='timesformer', value_arch='resnet18',
                 policy_pretrained=False, value_pretrained=False):
        '''
        X
        '''
        super().__init__()
        self.logger = logger
        self.num_frames = num_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.action_space = action_space
        self.policy_arch = policy_arch
        self.value_arch = value_arch
        self.policy_pretrained = policy_pretrained
        self.value_pretrained = value_pretrained

        # Instantiate policy network.
        if policy_arch == 'timesformer':
            self.policy_backbone = vision_tf.MyDenseTimeSformerBackbone(
                logger, num_frames=num_frames, frame_height=frame_height, frame_width=frame_width,
                in_channels=3)
            self.patch_dim = 16
            self.use_feature_dim = self.policy_backbone.output_feature_dim

        elif policy_arch == 'resnet50':
            self.policy_backbone = resnet.MyDenseResNetBackbone(
                logger, frame_height=frame_height, frame_width=frame_width,
                in_channels=3 * num_frames)
            self.patch_dim = 16
            self.use_feature_dim = self.policy_backbone.output_feature_dim // self.num_frames
        
        self.policy_post_linear = torch.nn.Linear(self.use_feature_dim, action_space)

        # Instantiate value function network.
        if value_arch == 'resnet18':
            self.value_backbone = torchvision.models.resnet18(pretrained=value_pretrained)
            self.value_backbone.conv1 = torch.nn.Conv2d(
                in_channels=3 * num_frames, out_channels=64, kernel_size=7, stride=2, padding=3,
                bias=False)
            self.value_backbone.fc = torch.nn.Linear(512, 1)

    def forward(self, input_rgb):
        '''
        :param input_rgb (B, 3, T, Hf, Wf) tensor.
        :return (outpol_logits, outpol_probits, output_value).
            outpol_logits (B, A) tensor.
            outpol_probits (B, A) tensor.
            output_value (B, 1) tensor.
        '''
        # Run policy network.
        if self.policy_arch == 'timesformer':
            outpol_features = self.policy_backbone(input_rgb)  # (B, D, T, H, W).

        elif self.policy_arch == 'resnet50':
            # TODX DRY
            input_rgb_stack = rearrange(input_rgb, 'B C T H W -> B (C T) H W')
            outpol_features_stack = self.policy_backbone(input_rgb_stack)
            outpol_features_stack = \
                outpol_features_stack[:, :self.use_feature_dim * self.num_frames]
            outpol_features = rearrange(outpol_features_stack, 'B (D T) H W -> B D T H W',
                                        T=self.num_frames, D=self.use_feature_dim)

        outpol_features = rearrange(outpol_features, 'B D T H W -> B T H W D')

        outpol_logits = self.policy_post_linear(outpol_features)  # (B, T, H, W, A).

        outpol_logits = torch.mean(outpol_logits, dim=(1, 2, 3))  # (B, A).

        outpol_probits = torch.nn.functional.softmax(outpol_logits, dim=-1)  # (B, A).

        # Run value estimation network.
        input_rgb_stack = rearrange(input_rgb, 'B C T H W -> B (C T) H W')
        output_value = self.value_backbone(input_rgb_stack)  # (B, 1).

        return (outpol_logits, outpol_probits, output_value)


if __name__ == '__main__':

    import logvisgen
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    logger = logvisgen.Logger()

    (B, T, H, W, C, A) = (2, 4, 224, 288, 3, 36)

    hider_net = SnitchPolicy(logger, T, H, W, A, 'timesformer', 'resnet18', True, True)

    x = torch.randn(B, C, T, H, W)
    print('x:', x.shape, x.min().item(), x.mean().item(), x.max().item())

    (logits, probits, value) = hider_net(x)
    print('logits:', logits.shape, logits.min().item(), logits.mean().item(), logits.max().item())
    print('probits:', probits.shape, probits.min().item(),
          probits.mean().item(), probits.max().item())
    print('value:', value.shape, value.min().item(), value.mean().item(), value.max().item())
    assert logits.shape == (B, A)
    assert probits.shape == (B, A)
    assert value.shape == (B, 1)

    pass
