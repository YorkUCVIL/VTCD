# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

from models.hide_seek.tcow.TimeSformer.timesformer.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models.hide_seek.tcow.TimeSformer.timesformer.models.helpers import load_pretrained
from models.hide_seek.tcow.TimeSformer.timesformer.models.vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,  # misleading?
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),  # misleading?
    ),
    # BVH MOD:
    'catchall': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),  # misleading?
    ),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True, causal_attention=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        self.causal_attention = causal_attention  # BVH MOD
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, cluster_subject=None):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn_pre_soft = (q @ k.transpose(-2, -1)) * self.scale


        attn = attn_pre_soft
        # BVH MOD: Apply temporal causal attention mask by setting lower triangle values to zero.
        # attn = (B, H, T, T) where H is multihead, first T is query time, second T is key time.
        # NOTE: We must do this before softmax to ensure probabilities keep summing to one.
        if self.causal_attention > 0:
            causal_mask = torch.ones(attn.shape, dtype=torch.bool, device=attn.device)
            if self.causal_attention <= 2:
                causal_mask = causal_mask.tril()
            else:
                causal_mask = causal_mask.tril(diagonal=self.causal_attention - 2)
            attn[~causal_mask] = -1e10
        # TIME: # (h w) n (b t) (b t) # SPACE: (b t) n (h w +1) (h w +1)
        attn_caus = attn
        attn = attn.softmax(dim=-1)

        # plt.imshow(rearrange(attn[:, :, 0, 1:].mean(1), 't (h w) -> t h w', h=15, w=20).cpu().numpy()[0]);
        # plt.show()
        
        # Uncommenting this causes crash:
        # if self.causal_attention > 0:
        #     attn[~causal_mask] = 0.0
        
        attn = self.attn_drop(attn)

        tokens = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x2 = self.proj(tokens)
            x2 = self.proj_drop(x2)

        # shape(x) = shape(x2) = (300, 30, 768) = (h*w, T, D).
        '''
BVH NOTES: To check if online operation is correct, place breakpoint and run:
x.retain_grad(), x2.retain_grad()
torch.autograd.backward(x2[0, 10, 100], retain_graph=True)
print(x2.grad[0, :, 100])  # should be one-hot
print(x.grad[0, :, 100])  # should be a bunch of non-zero values + all-zero values.
        '''
        if cluster_subject == 'tokens':
            return x2, rearrange(x2, 't s (nh m) -> t s nh m', m=C // self.num_heads, nh=self.num_heads)
            # return x2, tokens
        if cluster_subject == 'values':
            return x2, v.transpose(1, 2)
        if cluster_subject == 'keys':
            return x2, k.transpose(1, 2)
        if cluster_subject == 'queries':
            return x2, q.transpose(1, 2)
        if cluster_subject == 'attn':
            return x2, attn
        if cluster_subject == 'attn_sft':
            return x2, attn_pre_soft.softmax(dim=-1)
        if cluster_subject == 'attn_caus':
            return x2, attn_caus
        return x2, None


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time', causal_attention=False):
        super().__init__()
        self.attention_type = attention_type
        self.causal_attention = causal_attention  # BVH MOD
        assert(attention_type in ['divided_space_time', 'space_only', 'joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)  # NOTE: causal_attention must definitely remain False here.

        # Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, causal_attention=self.causal_attention)
            self.temporal_fc = nn.Linear(dim, dim)

        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, B, T, W, cluster_subject=None, use_temporal_attn=False, attn_head=-1):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            assert self.causal_attention == 0

            x = self.norm1(x)
            x, cluster_feature_s = self.attn(x, cluster_subject)
            x = x + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            # x = x + self.drop_path(self.attn(self.norm1(x)))
            # x = x + self.drop_path(self.mlp(self.norm2(x)))

            if cluster_feature_s is not None:
                # Remove the CLS token.
                if 'attn' in cluster_subject:
                    # (b t) n (h w +1) (h w +1)
                    if not attn_head == [-1]:
                        cluster_feature = cluster_feature_s[:, attn_head, 1:, 1:]
                    else:
                        cluster_feature = cluster_feature_s[:, :, 1:, 1:].mean(1)
                    # t h w hxw (spatial attention map
                    cluster_feature = rearrange(cluster_feature, '(b t) (h1 w1) (h2 w2) -> b (h1 w1 t) (h2 w2)', b=B, h1=H, w1=W, h2=H,
                                                w2=W, t=T)
                    # cluster_feature = rearrange(cluster_feature, '(b t) (h1 w1) (h2 w2) -> b (h2 w2 t) (h1 w1)', b=B, h1=H, w1=W, h2=H, w2=W, t=T)
                else:
                    # cluster_feature_s = cluster_feature_s[:, 1:, :]
                    # cluster_feature = rearrange(cluster_feature_s, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)

                    if not attn_head == [-1]:
                        cluster_feature = cluster_feature_s[:, 1:, attn_head, :]
                    else:
                        cluster_feature = cluster_feature_s[:, 1:, :, :].mean(2, keepdim=True)
                    # cluster_feature = rearrange(cluster_feature, 'b (h w t) nh m -> b (h w t) nh m', b=B, h=H, w=W, t=T)
            return x, cluster_feature

        elif self.attention_type == 'divided_space_time':
            # Temporal
            # BVH MOD: Apply causal_attention (see Attention module).
            # x = (1, 1945, 768) = (B, T*H*W+1, D).
            xt = x[:, 1:, :]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            # xt = (108, 18, 768) = (B*H*W, T, D).
            xt_norm = self.temporal_norm1(xt)
            xt_token, cluster_feature_t = self.temporal_attn(xt_norm, cluster_subject)
            res_temporal = self.drop_path(xt_token)
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            # res_temporal = (1, 1944, 768) = (B, T*H*W, D).
            xt = x[:, 1:, :] + res_temporal

            # Spatial
            if self.causal_attention in [0, 1]:
                init_cls_token = x[:, 0, :].unsqueeze(1)
                cls_token = init_cls_token.repeat(1, T, 1)
                cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
                xs = xt
                xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
                xs = torch.cat((cls_token, xs), 1)
                xs_norm = self.norm1(xs)
                xs_token, cluster_feature_s = self.attn(xs_norm, cluster_subject)
                res_spatial = self.drop_path(xs_token)

                # Taking care of CLS token
                cls_token = res_spatial[:, 0, :]
                cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
                
                if self.causal_attention == 0:
                    # BVH MOD: Very important former bug! This step indirectly used to cause temporal
                    # non-causal information leakage happening over >= 2 subsequent attention blocks:
                    cls_token = torch.mean(cls_token, 1, True)  # averaging for every frame
                
                else:
                    cls_token = cls_token[:, 0:1, :]  # Just copy the one from the first frame.
            
                res_spatial = res_spatial[:, 1:, :]
            
            elif self.causal_attention >= 2 or self.causal_attention == -1:
                # New: Avoid cls_token altogether.
                init_cls_token = x[:, 0, :].unsqueeze(1)  # (B, 1, D).
                cls_token = torch.zeros_like(init_cls_token)  # (B, 1, D).
                xs = xt
                xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
                res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial
            x = xt

            if cluster_feature_s is not None:
                 # Remove the CLS token.
                if not use_temporal_attn:
                    if 'attn' in cluster_subject:
                        # (b t) n (h w +1) (h w +1)
                        if not attn_head == [-1]:
                            cluster_feature = cluster_feature_s[:, attn_head, 1:, 1:]
                        else:
                            cluster_feature = cluster_feature_s[:, :, 1:, 1:].mean(1).unsqueeze(1)
                        # t h w hxw (spatial attention map
                        rearrange(cluster_feature, '(b t) n (h1 w1) (h2 w2) -> b (h1 w1 t) n (h2 w2)', b=B,
                                  n=len(attn_head), h1=H, w1=W, h2=H, w2=W, t=T)
                        # cluster_feature = rearrange(cluster_feature, '(b t) (h1 w1) (h2 w2) -> b (h2 w2 t) (h1 w1)', b=B, h1=H, w1=W, h2=H, w2=W, t=T)
                    else:
                        # cluster_feature_s = cluster_feature_s[:, 1:, :]
                        # cluster_feature = rearrange(cluster_feature_s, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)

                        if not attn_head == [-1]:
                            cluster_feature = cluster_feature_s[:, 1:, attn_head, :]
                        else:
                            cluster_feature = cluster_feature_s[:, 1:, :, :].mean(2, keepdim=True)
                        cluster_feature = rearrange(cluster_feature, '(b t) (h w) nh m -> b (h w t) nh m', b=B, h=H, w=W, t=T)
                else:
                    if 'attn' in cluster_subject:
                        if not attn_head == [-1]:
                            cluster_feature = cluster_feature_t[:, attn_head]
                        else:
                            cluster_feature = cluster_feature_t.mean(1)
                        # cluster_feature = rearrange(cluster_feature, '(b h w) t1 t2 -> b (h w t1) t2', h=H, w=W, t1=T, t2=T)
                        cluster_feature = rearrange(cluster_feature, '(b h w) n t1 t2 -> b (h w t1) n t2', h=H, n=len(attn_head), w=W, t1=T, t2=T)
                    else:
                        cluster_feature = rearrange(cluster_feature_t, '(h w) (b t) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            else:
                cluster_feature = None
                # stack along head dimension
            # Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            # x = x + self.drop_path(self.mlp(self.norm2(x)))
            x1 = self.norm2(x)
            x1 = self.mlp(x1)
            x = x + self.drop_path(x1)
            if cluster_subject == 'block_token':
                cluster_feature = x[:,1:,:].unsqueeze(2)
            return x, cluster_feature


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformere
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8,
                 attention_type='divided_space_time', causal_attention=0, dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.causal_attention = causal_attention  # BVH MOD
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type, causal_attention=self.causal_attention)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # BVH NOTE: This is irrelevant and replaced by vision_tf.py DenseTimeSformer forward().

        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            # Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        # Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        # Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
            x = torch.mean(x, 1)  # averaging predictions for every frame

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        # BVH NOTE: This is irrelevant and replaced by vision_tf.py DenseTimeSformer forward().

        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# @MODEL_REGISTRY.register()
# class vit_base_patch16_224(nn.Module):
#     def __init__(self, cfg, **kwargs):
#         super(vit_base_patch16_224, self).__init__()
#         self.pretrained = True
#         patch_size = 16
#         self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(
#             nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

#         self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
#         self.model.default_cfg = default_cfgs['vit_base_patch16_224']
#         self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * \
#             (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
#         pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
#         if self.pretrained:
#             load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter,
#                             img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

#     def forward(self, x):
#         x = self.model(x)
#         return x


@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8,
                 attention_type='divided_space_time', causal_attention=0, drop_path_rate=0.1,
                 network_depth=12, pretrained=False, pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained = pretrained

        if network_depth == 12:
            self.model = VisionTransformer(
                img_size=img_size, num_classes=num_classes,
                patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=drop_path_rate, num_frames=num_frames,
                attention_type=attention_type, causal_attention=causal_attention, **kwargs)

        elif network_depth == 18:
            self.model = VisionTransformer(
                img_size=img_size, num_classes=num_classes,
                patch_size=patch_size, embed_dim=896, depth=18, num_heads=14, mlp_ratio=4,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=drop_path_rate, num_frames=num_frames,
                attention_type=attention_type, causal_attention=causal_attention, **kwargs)

        elif network_depth == 24:
            self.model = VisionTransformer(
                img_size=img_size, num_classes=num_classes,
                patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                attn_drop_rate=0., drop_path_rate=drop_path_rate, num_frames=num_frames,
                attention_type=attention_type, causal_attention=causal_attention, **kwargs)

        else:
            raise ValueError(f'Invalid network depth {network_depth}, must be one of 12, 18, 24.')

        self.attention_type = attention_type

        # BVH MOD:
        # self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        cfg_key = 'vit_base_patch' + str(patch_size) + '_224'
        if cfg_key in default_cfgs:
            self.model.default_cfg = default_cfgs[cfg_key]
        else:
            self.model.default_cfg = default_cfgs['catchall']

        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size,
                            num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x
