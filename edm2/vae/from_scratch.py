import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import einops
from typing import Optional

from ..utils import mp_sum, normalize, mp_silu
from ..conv import MPConv


class MPGroupCausal3DConvVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, group_size, kernel, dilation = (1,1,1)):
        super().__init__()
        self.out_channels = out_channels
        self.group_size = group_size
        self.dilation = dilation
        self.weight = torch.nn.Parameter(torch.randn(out_channels*group_size, in_channels, *kernel))

        kt, kw, kh = kernel
        dt, dw, dh = dilation
        self.image_padding = (0, dh * (kh-1)//2, dw * (kw-1)//2)
        self.time_padding_size = kt*dt-self.group_size

    def forward(self, x, gain=1, cache=None):
        batch_size, channels, time, height, width = x.shape

        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization. for the gradients 
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)


        if cache is None:
            cache = torch.ones(batch_size, channels, self.time_padding_size, height, width, device=x.device, dtype=x.dtype)

        x = torch.cat((cache, x), dim=-3)
        cache = x[:,:,-self.time_padding_size:].clone().detach()

        x = F.conv3d(x, w, padding=self.image_padding, stride = (self.group_size, 1, 1), dilation=self.dilation)

        x = einops.rearrange(x, 'b (c g) t h w -> b c (t g) h w', g=self.group_size)

        return x, cache

def downsample(x):
    return einops.rearrange(x, 'b c (t ts) (h hs) (w ws) -> b (c ts hs ws) t h w', ts=4, hs=2, ws=2)

def upsample(x):
    return einops.rearrange(x, 'b (c ts hs ws) t h w -> b c (t ts) (h hs) (w ws)', ts=4, hs=2, ws=2)

class ResBlock(nn.Module):
    def __init__(self, channels: int, group_size:int, kernel_size=(8,3,3)):
        super().__init__()
        self.channels = channels

        self.conv_res0 = MPGroupCausal3DConvVAE(channels, channels, group_size, kernel_size, dilation = (1,1,1))
        self.conv_res1 = MPGroupCausal3DConvVAE(channels, channels, group_size, kernel_size, dilation = (3,3,3))
                                                                   
        # self.attn_block = VAEAttention(channels, num_heads=4)
    
    def forward(self, x):
        y = self.conv_res0(x)
        y = mp_silu(y)
        y = self.conv_res1(y)
        y = mp_silu(y)

        x = mp_sum(x,y)

        return x















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

