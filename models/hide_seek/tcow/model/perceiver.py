'''
Model constituents / network modules.
Created by Basile Van Hoorick, Jun 2022.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'model/'))

from __init__ import *

# Library imports.
# https://huggingface.co/docs/transformers/quicktour
import transformers
from transformers.models.perceiver import modeling_perceiver

# Internal imports.
import my_utils


class MyPerceiverBackbone(torch.nn.Module):

    def __init__(self, logger, input_shape, input_channels, output_shape, output_channels,
                 samples_per_frame, output_pos_enc):
        '''
        :param input_shape (tuple of int): Input index dimensionality (up to 3D).
        :param input_channels (int): Input embedding dimensionality; only relevant if input is not 1D.
        :param output_shape (tuple of int): Output index dimensionality (up to 3D).
        :param output_channels (int): Output embedding dimensionality; only relevant if output is not 1D.
        :param samples_per_frame (int): If input and/or output is 1D, patch length (i.e. bundling factor).
        :param output_pos_enc (str): Positional encoding format for output waveforms / images / videos
            (fourier / trainable).
        '''
        super().__init__()
        self.logger = logger
        self.input_shape = input_shape
        self.input_channels = input_channels
        self.output_shape = output_shape
        self.output_channels = output_channels
        self.samples_per_frame = samples_per_frame
        self.output_pos_enc = output_pos_enc

        # NOTE: This is heavily inspired by huggingface modeling_perceiver.py

        # For most parameters, use something similar to standard config.
        self.config = transformers.PerceiverConfig(
            num_latents=256,
            d_latents=1024,
            d_model=512,
            num_blocks=1,
            num_self_attends_per_block=12,
            num_self_attention_heads=8,
            num_cross_attention_heads=1,
            samples_per_patch=samples_per_frame,
        )

        # Number of frequencies for fourier positional encoding should be proportional to data dimensions.
        max_input_dim = input_shape[0] // samples_per_frame if len(
            input_shape) == 1 else max(input_shape)
        max_output_dim = output_shape[0] // samples_per_frame if len(
            output_shape) == 1 else max(output_shape)
        input_freqs = max_input_dim // 4
        output_freqs = max_output_dim // 4

        # Instantiate preprocessor (before encoder).
        if len(input_shape) == 1:
            # Assume input is waveform.
            self.input_preprocessor = modeling_perceiver.PerceiverAudioPreprocessor(
                self.config,
                prep_type='patches',
                samples_per_patch=samples_per_frame,
                out_channels=samples_per_frame * 2,
                position_encoding_type='fourier',
                fourier_position_encoding_kwargs=dict(
                    max_resolution=(input_shape[0],),
                    num_bands=input_freqs,
                    concat_pos=True,
                    sine_only=False,
                ),
            )

        elif len(input_shape) in [2, 3]:
            # Assume input is image or video.
            self.input_preprocessor = modeling_perceiver.PerceiverImagePreprocessor(
                self.config,
                prep_type='conv',
                spatial_downsample=1,
                position_encoding_type='fourier',
                in_channels=input_channels,
                out_channels=256,
                fourier_position_encoding_kwargs=dict(
                    max_resolution=input_shape,
                    num_bands=input_freqs,
                    concat_pos=True,
                    sine_only=False,
                ),
            )

        else:
            raise ValueError(input_shape)

        # Instantiate postprocessor (after decoder).
        if len(output_shape) == 1:
            # Assume output is waveform.
            decoder_output_num_channels = samples_per_frame
            decoder_output_index_dims = (output_shape[0] // samples_per_frame,)
            self.output_postprocessor = modeling_perceiver.PerceiverAudioPostprocessor(
                self.config,
                in_channels=decoder_output_num_channels,
            )

        elif len(output_shape) in [2, 3]:
            # Assume output is image or video.
            decoder_output_num_channels = 128
            decoder_output_index_dims = output_shape
            self.output_postprocessor = modeling_perceiver.PerceiverProjectionPostprocessor(
                in_channels=decoder_output_num_channels,
                out_channels=output_channels,
            )

        else:
            raise ValueError(output_shape)

        # Instantiate query embeddings.
        if output_pos_enc == 'fourier':
            # This is why huggingface's implementation is awkward; doesn't automatically infer this.
            # NOTE: There is only one cross attention (PerceiverLayer) step, so it is not that bad.
            query_num_channels = (output_freqs * 2 + 1) * len(output_shape)
            position_encoding_kwargs = dict(
                max_resolution=decoder_output_index_dims,
                num_bands=output_freqs,
                concat_pos=True,
                sine_only=False,
            )

        elif output_pos_enc == 'trainable':
            query_num_channels = 256
            position_encoding_kwargs = dict(
                index_dims=decoder_output_index_dims,
                num_channels=query_num_channels,
            )

        else:
            raise ValueError(output_pos_enc)

        # Instantiate decoder.
        self.decoder = modeling_perceiver.PerceiverBasicDecoder(
            self.config,
            output_num_channels=decoder_output_num_channels,
            output_index_dims=decoder_output_index_dims,
            num_channels=query_num_channels,
            position_encoding_type=output_pos_enc,
            concat_preprocessed_input=False,
            fourier_position_encoding_kwargs=position_encoding_kwargs,
            trainable_position_encoding_kwargs=position_encoding_kwargs,
        )

        self.perceiver = transformers.PerceiverModel(
            self.config,
            input_preprocessor=self.input_preprocessor,
            decoder=self.decoder,
            output_postprocessor=self.output_postprocessor,
        )

        pass

    def forward(self, input):
        '''
        :param input (B, S) or (B, C, H, W) or (B, C, T, H, W) tensor.
        :return (output, last_hidden_state)
            output (B, S) or (B, C, H, W) or (B, C, T, H, W) tensor.
            last_hidden_state (B, L, D) tensor.
        '''
        assert input.shape[-len(self.input_shape):] == self.input_shape

        # NOTE: It is crucial to specify the output shape as a flat array of flat indices here.
        # This is just how huggingface perceiver works.
        if len(self.output_shape) == 1:
            # Assume output is waveform.
            subsampling = torch.arange(
                self.output_shape[0] // self.samples_per_frame)
        elif len(self.output_shape) in [2, 3]:
            # Assume output is image or video.
            subsampling = torch.arange(np.prod(self.output_shape))

        perceiver_output = self.perceiver(
            inputs=input, subsampled_output_points=subsampling)
        output = perceiver_output.logits
        last_hidden_state = perceiver_output.last_hidden_state

        if len(self.output_shape) == 1:
            # Assume output is waveform.
            pass
        elif len(self.output_shape) == 2:
            # Assume output is image.
            # output = rearrange(output, 'B (H W) C -> B C (T Z)',
            #                    C=self.output_shape[0], T=self.output_shape[1], Z=1)
            output = rearrange(
                output, 'B (H W) C -> B C H W',
                H=self.output_shape[0], W=self.output_shape[1])
        elif len(self.output_shape) == 3:
            # Assume output is video.
            output = rearrange(
                output, 'B (T H W) C -> B C T H W',
                T=self.output_shape[0], H=self.output_shape[1], W=self.output_shape[2])

        assert output.shape[-len(self.output_shape):] == self.output_shape

        return (output, last_hidden_state)


if __name__ == '__main__':

    (S, T, H, W, C) = (6144, 16, 112, 112, 3)
    num_channels = 3
    samples_per_frame = 64

    # TODX: Video input does not work yet because not really supported by huggingface.
    # Instead, my recommendation is to concatenate frames along the channel dimension.

    for input_shape in [(S,), (H, W)]: # , (T, H, W)]:

        for output_shape in [(S,), (H, W), (T, H, W)]:

            for output_pos_enc in ['fourier', 'trainable']:

                print()
                print('input_shape:', input_shape)
                print('output_shape:', output_shape)
                print('output_pos_enc:', output_pos_enc)

                my_backbone = MyPerceiverBackbone(
                    None, input_shape, num_channels, output_shape,
                    num_channels, samples_per_frame, output_pos_enc)

                B = 2
                if len(input_shape) == 1:
                    my_input = torch.randn(B, S)
                else:
                    my_input = torch.randn(B, num_channels, *input_shape)

                print('my_input:', my_input.shape)
                (my_output, last_hidden_state) = my_backbone(my_input)
                print('my_output:', my_output.shape)
                print('last_hidden_state:', last_hidden_state.shape)

                if len(output_shape) == 1:
                    assert my_output.shape == (B, S)
                else:
                    assert my_output.shape == (B, num_channels, *output_shape)

                print()

    pass
