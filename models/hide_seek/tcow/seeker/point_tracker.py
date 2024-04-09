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
import models.hide_seek.tcow.utils.my_utils as my_utils
import models.hide_seek.tcow.model.resnet as resnet
import models.hide_seek.tcow.model.vision_tf as vision_tf


class QueryPointTracker(torch.nn.Module):
    '''
    X
    '''

    def __init__(self, logger, num_total_frames=24, num_visible_frames=16, frame_height=224,
                 frame_width=288, tracker_arch='timesformer', tracker_pretrained=False,
                 input_format='rgb', query_format='uvt', query_type='append',
                 output_format='uv', output_token='direct', output_type='regress'):
        '''
        X
        '''
        super().__init__()
        self.logger = logger
        self.num_total_frames = num_total_frames
        self.num_visible_frames = num_visible_frames
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.tracker_arch = tracker_arch
        self.tracker_pretrained = tracker_pretrained
        self.input_format = input_format
        self.query_format = query_format
        self.query_type = query_type
        self.output_format = output_format
        self.output_token = output_token
        self.output_type = output_type

        # Determine precise input shapes.
        input_channels = len(self.input_format)
        if 'append' in self.query_type:
            input_channels += len(self.query_format)
        if 'mask' in self.query_type:
            input_channels += 1
        if 'token' in self.query_type:
            pass  # This does not directly affect the number of input channels.
        self.input_channels = input_channels

        # Determine precise output shapes.
        if output_type == 'regress':
            output_squash = True
            output_channels = len(self.output_format)
        elif output_type == 'heatmap':
            output_squash = False
            output_channels = 1 + (1 if 'o' in self.output_format else 0)
        elif output_type == 'gaussian':
            output_squash = True
            output_channels = len(self.output_format) * 2 - (1 if 'o' in self.output_format else 0)
        self.output_squash = output_squash
        self.output_channels = output_channels

        # Instantiate actual network components.
        # Instantiate tracker backbone.
        if self.tracker_arch == 'timesformer':
            self.tracker_backbone = vision_tf.MyDenseTimeSformerBackbone(
                self.logger, num_frames=self.num_total_frames, frame_height=self.frame_height,
                frame_width=self.frame_width, in_channels=self.input_channels)
            self.patch_dim = 16
            self.use_feature_dim = self.tracker_backbone.output_feature_dim

        elif self.tracker_arch == 'resnet50':
            assert not('token' in self.query_type)
            self.tracker_backbone = resnet.MyDenseResNetBackbone(
                self.logger, frame_height=self.frame_height, frame_width=self.frame_width,
                in_channels=self.input_channels * self.num_visible_frames)
            self.patch_dim = 16
            self.use_feature_dim = self.tracker_backbone.output_feature_dim // self.num_total_frames

        # Instantiate query positional encoding & embedding & projection layer.
        if 'token' in self.query_type:
            assert self.tracker_arch == 'timesformer'
            self.query_num_freqs = 30
            self.query_pos_enc_size = my_utils.get_fourier_positional_encoding_size(
                len(self.query_format), self.query_num_freqs)
            assert self.query_pos_enc_size < self.use_feature_dim
            # Fill the remaining part of the query token with a fixed, learned embedding.
            self.query_prefix_emb = torch.nn.Parameter(
                torch.zeros(1, self.use_feature_dim - self.query_pos_enc_size))
            self.query_proj = torch.nn.Linear(self.use_feature_dim, self.use_feature_dim)

        # Instantiate head that turns final embeddings to trajectories.
        if self.output_token == 'query':
            # We have a single extra output token that must contain the entire trajectory.
            self.tracker_post_linear = torch.nn.Linear(
                self.use_feature_dim, self.output_channels * self.num_total_frames)
        else:
            # We have all spatiotemporal output tokens that are averaged spatially, so they must
            # each contain predictions for single frames only.
            self.tracker_post_linear = torch.nn.Linear(self.use_feature_dim, self.output_channels)

        assert self.frame_height % self.patch_dim == 0
        assert self.frame_width % self.patch_dim == 0

    def forward(self, input_frames, query_channels, query_mask):
        '''
        Assumes input frames are already blacked out as appropriate.
        :param input_frames (B, 3-7, Tt, Hf, Wf) tensor.
        :param query_channels (B, 3-7) tensor.
        :param query_mask (B, 1, Tt, Hf, Wf) tensor.
        :return output_traject (B, T, 2-7) tensor.
        '''
        # Append query information in desired way.
        (B, _, T, H, W) = input_frames.shape
        if query_channels is not None and len(query_channels.shape) >= 2:
            query_channels = query_channels.type(torch.float32)
            assert query_channels.shape[-1] == len(self.query_format)
        if query_mask is not None and len(query_mask.shape) >= 2:
            query_mask = query_mask.type(torch.float32)
            assert query_mask.shape[1] == 1

        input_with_query = input_frames.clone()
        if 'append' in self.query_type:
            # Concatenate along input channel axis.
            query_channels_bc = query_channels[..., None, None, None].broadcast_to(B, -1, T, H, W)
            input_with_query = torch.cat([input_with_query, query_channels_bc], dim=1)
        if 'mask' in self.query_type:
            input_with_query = torch.cat([input_with_query, query_mask], dim=1)
        # NOTE: input_with_query remains the same as input_frames if query_type is only token.

        if self.tracker_arch == 'timesformer':

            if 'token' in self.query_type:
                # Preprocess query with Fourier positional encoding and projection.
                query_pos_enc = my_utils.apply_fourier_positional_encoding(
                    query_channels, self.query_num_freqs)  # (B, <D).
                assert query_pos_enc.shape[-1] == self.query_pos_enc_size
                query_prefix_emb_bc = self.query_prefix_emb.repeat(B, 1)  # (B, <D).
                query_token = torch.cat([query_prefix_emb_bc, query_pos_enc], dim=-1)  # (B, D).
                extra_token_in = query_token.unsqueeze(-1)  # (B, D, 1).

            else:
                # Assume query is already passed via append or mask.
                extra_token_in = None

            (output_features, extra_token_out) = self.tracker_backbone(
                input_with_query, extra_token_in)  # (B, D, T, H, W), (B, D, 1).
            extra_embedding = extra_token_out.squeeze(-1)  # (B, D).

        elif self.tracker_arch == 'resnet50':
            input_with_query = input_with_query[:, :, :self.num_visible_frames, :, :]
            input_with_query_stack = rearrange(input_with_query, 'B C T H W -> B (C T) H W')
            output_features_stack = self.tracker_backbone(input_with_query_stack)
            output_features_stack = \
                output_features_stack[:, :self.use_feature_dim * self.num_total_frames]
            output_features = rearrange(output_features_stack, 'B (D T) H W -> B D T H W',
                                        T=self.num_total_frames, D=self.use_feature_dim)
            extra_embedding = None

        if self.output_token == 'query':
            # New way: use extra output token (= similar to classification or distillation).
            assert self.output_squash
            output_traject = self.tracker_post_linear(extra_embedding)  # (B, T * 2-7).
            output_traject = rearrange(output_traject, 'B (T C) -> B T C',
                                       T=self.num_total_frames, C=self.output_channels)

        elif self.output_token == 'direct':
            # Old way: average output feature map.
            assert self.output_squash
            output_features = rearrange(output_features, 'B D T H W -> B T H W D')
            output_patches = self.tracker_post_linear(output_features)  # (B, T, Hm, Wm, 2-7).
            output_traject = output_patches.mean(dim=(2, 3))
            # (B, T, 2-7) with (u, v) or (u, v, d) or (x, y, z) or Gaussian (mean + std) thereof
            # + optionally (o).

        return output_traject  # (B, T, 2-7).


if __name__ == '__main__':

    # All four-way for-loop tests (i.e. assert y.shape) passed on 18 Aug 2022.

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    (B, Tt, Tv, Hf, Wf) = (2, 16, 12, 224, 288)

    for input_format in ['rgb', 'rgbd', 'xyzrgb', 'xyzrgbd']:

        print()
        print('input_format:', input_format)
        Ci = len(input_format)

        x = torch.randn(B, Ci, Tt, Hf, Wf)
        # x = x.cuda()
        print('x:', x.shape, x.min().item(), x.mean().item(), x.max().item())

        for query_format in ['uvt', 'uvdt', 'xyzt', 'uvxyzt', 'uvdxyzt', 'mask']:

            print()
            print('query_format:', query_format)
            Cq = len(query_format)

            query_channels = torch.randn(B, Cq)
            query_mask = torch.zeros(B, 1, Tt, Hf, Wf)
            query_mask[:, :, 2, 3, 4] = 1.0
            # query_channels = query_channels.cuda()
            # query_mask = query_mask.cuda()

            for query_type in ['append', 'mask', 'token']:

                print()
                print('query_type:', query_type)

                for output_format in ['uv', 'uvd', 'uvo', 'uvdo', 'xyz', 'xyzo']:

                    print()
                    print('output_format:', output_format)
                    Co = len(output_format) * 2 - (1 if 'o' in output_format else 0)

                    seeker_net = QueryPointTracker(
                        None, Tt, Tv, Hf, Wf, 'timesformer', False,
                        input_format, query_format, query_type, output_format, 'gaussian')

                    y = seeker_net(x, query_channels, query_mask)
                    print('y:', y.shape, y.min().item(), y.mean().item(), y.max().item())
                    print()

                    assert y.shape == (B, Tt, Co)

    pass
