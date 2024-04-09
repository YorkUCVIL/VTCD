
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/constants.py
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DenseVisionTransformer(torch.nn.Module):
    '''
    Based on https://github.com/rwightman/pytorch-image-models.
    '''

    def __init__(self, logger, timm_name, pretrained_frozen, frame_height, frame_width, patch_dim,
                 in_channels):
        super().__init__()
        self.logger = logger
        self.timm_name = timm_name
        self.pretrained = pretrained_frozen
        # Frame size.
        self.Hf = frame_height
        self.Wf = frame_width
        # Number of patches.
        self.Ho = frame_height // patch_dim
        self.Wo = frame_width // patch_dim
        # Patch size.
        self.ho = patch_dim
        self.wo = patch_dim
        # Number of channels.
        self.Ci = in_channels

        # NOTE: We are usually modifying the image size which results in a different token sequence
        # length and set of positional embeddings. Not sure what the effect is.
        self.vit = timm.create_model(timm_name, pretrained=pretrained_frozen,
                                     img_size=(self.Hf, self.Wf))
        assert self.ho == 16 and self.wo == 16
        self.output_feature_dim = 768

        if pretrained_frozen:
            # Disable gradients for target features.
            # NOTE: used_model must always be set to eval, regardless of this model phase.
            for param in self.vit.parameters():
                param.requires_grad_(False)

        # Replace first convolutional layer to accommodate non-standard inputs.
        if in_channels != 3:
            assert not(pretrained_frozen)
            self.vit.patch_embed.proj = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=768, kernel_size=(16, 16), stride=(16, 16))

    def forward(self, input_pixels):
        '''
        :param input_pixels (B, C, Hf, Wf) tensor.
        :return output_features (B, D, Ho, Wo) tensor.
        '''

        # Normalize if pretrained.
        if self.pretrained:
            mean = torch.tensor(IMAGENET_DEFAULT_MEAN, dtype=input_pixels.dtype,
                                device=input_pixels.device)
            mean = mean[:, None, None].expand_as(input_pixels[0])
            std = torch.tensor(IMAGENET_DEFAULT_STD, dtype=input_pixels.dtype,
                               device=input_pixels.device)
            std = std[:, None, None].expand_as(input_pixels[0])
            input_pixels = input_pixels - mean
            input_pixels = input_pixels / std

        # Adapted from
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py.
        x = self.vit.patch_embed(input_pixels)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)  # (B, N, D), where N = 1 (cls_token) + Ho * Wo.

        # Discard cls_token altogether, and skip norm, pre_logits, head.
        x = x[:, 1:]  # (B, Ho * Wo, D).

        # Refer to
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py.
        # Here, we undo x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC.
        x = rearrange(x, 'B (H W) D -> B D H W', H=self.Ho, W=self.Wo)
        output_features = x

        assert output_features.shape[1] == self.output_feature_dim

        return output_features


class MyDenseVisionTransformerBackbone(DenseVisionTransformer):
    '''
    Trainable variant of the DenseVisionTransformer.
    '''

    def __init__(self, logger, frame_height=224, frame_width=288, in_channels=3):
        # TODO: Currently hardcoded to vit_base_patch16_224.
        super().__init__(logger, 'vit_base_patch16_224', False, frame_height, frame_width,
                         16, in_channels)
