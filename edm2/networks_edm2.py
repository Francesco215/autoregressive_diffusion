# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import numpy as np
import torch
from torch import nn, Tensor
import einops

from .utils import normalize, resample, mp_silu, mp_sum, mp_cat, MPFourier
from .conv import MPCausal3DConv, MPConv, EfficientWeight, Weight
from .attention import FrameAttention, VideoAttention



#---------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        # if attention:
        self.conv_res0 = MPCausal3DConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3,3])
        self.conv_res1 = MPCausal3DConv(out_channels, out_channels, kernel=[3,3,3])
        # else:
        # self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        # self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])

        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn = VideoAttention(out_channels, self.num_heads, attn_balance)
        # self.attn = FrameAttention(out_channels, self.num_heads, attn_balance)

        if self.num_heads > 0: 
            assert (out_channels & (out_channels - 1) == 0) and out_channels != 0, f"out_channels must be a power of 2, got {out_channels}"

    def forward(self, x, emb, batch_size):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x),batch_size=batch_size)
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = einops.einsum(y, c.to(y.dtype),'b c h w, b c -> b c h w') 
        y = mp_silu(y)
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y, batch_size=batch_size)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        x = self.attn(x, batch_size)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.tensor([1.], requires_grad=True))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[]) # this are actually linear layers with normalized weights. the kernel is empty
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level #bitwise right shift. it divides by 2 (img_resolution // 2)
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels, kernel=[3,3])

    def forward(self, x, noise_labels, text_embeddings = None):
        # x.shape = b t c h w
        batch_size, time_dimention = x.shape[:2]
        x = einops.rearrange(x, 'b t ... -> (b t) ...')
        noise_labels = einops.rearrange(noise_labels, 'b t ... -> (b t) ...')

        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None and text_embeddings is not None:
            # TODO: might need to change this for when the class label is not none
            text_embeddings = self.emb_label(text_embeddings * np.sqrt(text_embeddings.shape[1]))
            if text_embeddings.shape[0]!=1:
                text_embeddings = torch.repeat_interleave(text_embeddings, time_dimention, dim = 0)
            emb = mp_sum(emb, text_embeddings, t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb, batch_size=batch_size)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb, batch_size=batch_size)
        x = self.out_conv(x, gain=self.out_gain)
        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=batch_size)
        return x

#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

class Precond(torch.nn.Module):
    def __init__(self,
        unet,                   # UNet model.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.unet = unet
        self.img_resolution = unet.img_resolution
        self.img_channels = unet.img_channels
        self.label_dim = unet.label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data

    def forward(self, x:Tensor, sigma:Tensor, text_embeddings:Tensor=None, force_fp32=False, **unet_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        sigma = einops.rearrange(sigma, '... -> ... 1 1 1')

        text_embeddings = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if text_embeddings is None else text_embeddings.to(torch.float32)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.view(sigma.shape[:2]).log() / 4
 
        # Run the model.
        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, c_noise, text_embeddings, **unet_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        return D_x

    def normalize_all_weights(self):
        for module in self.modules():
            if isinstance(module, (EfficientWeight, Weight)):
                module.normalize_weight()
#----------------------------------------------------------------------------
