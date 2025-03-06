import torch
from torch import nn, Tensor
from torch.nn import functional as F


import einops
import numpy as np

from ..utils import mp_sum, mp_silu
from ..conv import MPConv, NormalizedWeight


class GroupCausal3DConvVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, dilation = (1,1,1)):
        super().__init__()
        self.out_channels = out_channels
        self.group_size = group_size
        self.dilation = dilation
        self.weight = NormalizedWeight(in_channels, out_channels*group_size, kernel)

        kt, kw, kh = kernel
        dt, dw, dh = dilation
        self.image_padding = (0, dh * (kh-1)//2, dw * (kw-1)//2)
        self.time_padding_size = kt+(kt-1)*(dt-1)-self.group_size

    def forward(self, x, gain=1, cache=None):

        if cache is None:
            cache = torch.ones(*x.shape[:2], self.time_padding_size, *x.shape[3:], device=x.device, dtype=x.dtype)

        x = torch.cat((cache, x), dim=-3)
        cache = x[:,:,-self.time_padding_size:].clone().detach()

        w = self.weight(gain).to(x.dtype)
        x = F.conv3d(x, w, padding=self.image_padding, stride = (self.group_size, 1, 1), dilation=self.dilation)

        x = einops.rearrange(x, 'b (c g) t h w -> b c (t g) h w', g=self.group_size)

        if self.training: cache = None
        return x, cache





class ConvVAE(MPConv):
    def forward(self,x, gain=1):
        batch_size = x.shape[0]
        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        super().forward(x,gain)
        return einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)






class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel=(8,3,3), group_size=1):
        super().__init__()

        self.conv3d0 = GroupCausal3DConvVAE(channels, channels,  kernel, group_size, dilation = (1,1,1))
        self.conv3d1 = GroupCausal3DConvVAE(channels, channels, kernel, group_size, dilation = (3,3,3))

        self.conv2d0 = ConvVAE(channels, channels,  kernel[1:], group_size)
        self.conv2d1 = ConvVAE(channels, channels, kernel[1:], group_size)
                                                                   
        self.weight_sum0 = nn.Parameter(torch.tensor(0.))
        self.weight_sum1 = nn.Parameter(torch.tensor(0.))

        # self.attn_block = VAEAttention(channels, num_heads=4)
    
    def forward(self, x, cache = None):
        if cache is None: cache = {}

        y, cache['conv3d_res0'] = self.conv3d0(x, cache=cache.get('conv3d_res0', None))
        y = mp_sum(y, self.conv2d0(x), F.sigmoid(self.weight_sum0))
        y = mp_silu(y)

        t = y.clone()
        
        y, cache['conv3d_res1'] = self.conv3d1(t, cache=cache.get('conv3d_res1', None))
        y = mp_sum(y, self.conv2d1(t), F.sigmoid(self.weight_sum1))
        y = mp_silu(y)

        x = mp_sum(x,y)

        return x, cache




class Encoder(nn.Module):

    def __init__(self, latent_channels, time_compressions = (2,2), spatial_compressions = (2,2)):
        super().__init__()

        time_compression = np.prod(time_compressions)

        self.time_compressions = time_compressions
        self.spatial_compressions = spatial_compressions

        #assuming the input is always rgb
        self.initial_conv = GroupCausal3DConvVAE(in_channels = 3, out_channels = 4, kernel = (8,3,3), group_size = time_compression)
        self.res_block1 = ResBlock(channels = 4, kernel=(8,3,3), group_size=time_compression)

        self.compression_block2 = GroupCausal3DConvVAE(in_channels = 4 * time_compressions[0]*spatial_compressions[0]**2, out_channels=4, kernel = (8,3,3), group_size = time_compressions[1])
        self.res_block2 = ResBlock(channels = 4, kernel=(8,3,3), group_size=time_compressions[1])

        self.compression_block3 = GroupCausal3DConvVAE(in_channels = 4 * time_compressions[1]*spatial_compressions[1]**2, out_channels=latent_channels*2, kernel = (8,3,3), group_size = 1)
        self.res_block3 = ResBlock(channels = latent_channels*2, kernel=(8,3,3), group_size= 1)


    def forward(self, x:Tensor, cache = None):
        if cache is None: cache = {}

        x, cache['initial_conv'] = self.initial_conv(x, cache = cache.get('initial_conv', None))
        x, cache['res_block1'] = self.res_block1(x, cache.get('res_block1', None))

        x = downsample(x, self.time_compressions[0], self.spatial_compressions[0])

        x, cache['compression_block2']= self.compression_block2(x, cache = cache.get('compression_block2',None))
        x, cache['res_block2'] = self.res_block2(x, cache.get('res_block2',None))

        x = downsample(x, self.time_compressions[1], self.spatial_compressions[1])

        x, cache['compression_block3']= self.compression_block3(x, cache = cache.get('compression_block3',None))
        x, cache['res_block3'] = self.res_block3(x, cache.get('res_block3',None))

        mean, logvar = x.split(split_size=x.shape[1]//2, dim = 1)

        return mean, logvar





def downsample(x, time_compression, spatial_compression):
    return einops.rearrange(x, 'b c (t ts) (h hs) (w ws) -> b (c ts hs ws) t h w', ts=time_compression, hs=spatial_compression, ws=spatial_compression)

def upsample(x, time_compression, spatial_compression):
    return einops.rearrange(x, 'b (c ts hs ws) t h w -> b c (t ts) (h hs) (w ws)', ts=time_compression, hs=spatial_compression, ws=spatial_compression)






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

