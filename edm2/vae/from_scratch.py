import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import einops
from typing import Optional

from ..utils import mp_sum, normalize, mp_silu
from ..conv import MPConv


class MPCausal3DConvVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        # TODO: make sure that the window of interaction is quite big
        self.out_channels = out_channels
        assert len(kernel)==3
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1, cache=None):
        batch_size, channels, time, height, width = x.shape

        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization. for the gradients 
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)

        image_padding = (0, w.shape[-2]//2, w.shape[-1]//2)
        causal_pad = torch.ones(batch_size, channels, w.shape[2]-1, height, width, device=x.device, dtype=x.dtype)

        if cache is None:
            cache = causal_pad

        # during inference is much simpler
        x = torch.cat((cache, x), dim=-3)
        cache = x[:,:,-(w.shape[2]-1):].clone()

        x = F.conv3d(x, w, padding=image_padding)

        return x, cache.detach()


def downsample(x):
    return einops.rearrange(x, 'b c (t ts) (h hs) (w ws) -> b (c ts hs ws) t h w', ts=4, hs=2, ws=2)

def upsample(x):
    return einops.rearrange(x, 'b (c ts hs ws) t h w -> b c (t ts) (h hs) (w ws)', ts=4, hs=2, ws=2)

class VAEAttention(nn.Module):
    def __init__(self, channels, num_heads, image_size, attn_balance = 0.3):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attn_balance = attn_balance
        self.image_size = image_size
        if num_heads == 0:
            return

        effective_size = channels * image_size
        self.attn_qkv = MPConv(effective_size, effective_size, kernel=[]) 
        self.attn_proj = MPConv(effective_size, effective_size, kernel=[]) 

    def forward(self, x):
        if self.num_heads==0:
            return x
        batch_size, channels, time, height, width = x.shape
        x = einops.rearrange(x, 'b c t h w -> b t (c h w)') # this is going to lead to huge latents

        y = self.attn_qkv(x)

        y = einops.rearrange(y, 'b t (s m chw) -> s b m t chw', s=3, m=self.num_heads)
        q, k, v =y.unbind(0)

        y = F.scaled_dot_product_attention(q, k, v)
        y = einops.rearrange(y, 'b m t (c h w) -> b c t h w', h=height, w=width)

        y = self.attn_proj(y)
        return mp_sum(x, y, t=self.attn_balance)

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.conv_res0 = MPCausal3DConvVAE(channels, channels, kernel_size=(3, 3, 3))
        self.conv_res1 = MPCausal3DConvVAE(channels, channels, kernel_size=(3, 3, 3))

        self.attn_block = VAEAttention(channels, num_heads=4)
    
    def forward(self, x):
        y = self.conv_res0(x)
        y = mp_silu(y)
        y = self.conv_res1(y)
        y = mp_silu(y)

        x = mp_sum(x,y)

        return self.attn_block(x)