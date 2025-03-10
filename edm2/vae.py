import torch
from torch import nn, Tensor
from torch.nn import functional as F


import einops
import numpy as np

from .utils import mp_sum, mp_silu
from .conv import MPConv, NormalizedWeight


class GroupCausal3DConvVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, dilation = (1,1,1)):
        super().__init__()
        self.out_channels = out_channels
        self.group_size = group_size
        self.dilation = dilation
        self.weight = NormalizedWeight(in_channels, out_channels*group_size, kernel)

        kt, kw, kh = kernel
        dt, dw, dh = dilation
        self.image_padding = (dh * (kh//2), dh * (kh//2), dw * (kw//2), dw * (kw//2))
        self.time_padding_size = kt+(kt-1)*(dt-1)-self.group_size

    def forward(self, x, gain=1, cache=None):
        x = F.pad(x, pad = self.image_padding, mode="constant", value = 1)
        w = self.weight(gain).to(x.dtype)

        multiplicative = 1.
        if cache is None:
            cache = torch.zeros(*x.shape[:2], self.time_padding_size, *x.shape[3:], device=x.device, dtype=x.dtype)

            kernel_size = torch.tensor(w.shape[2], device="cuda")
            group_index = torch.arange(x.shape[2], device=x.device)//self.group_size+1
            n_inputs_inside = torch.clip((group_index*self.group_size-1)//self.dilation[0]+1, max=kernel_size)
            multiplicative = (torch.sqrt(kernel_size/n_inputs_inside)[:,None,None]).detach()



        x = torch.cat((cache, x), dim=-3)
        cache =  None if self.training else x[:,:,-self.time_padding_size:].clone().detach()

        x = F.conv3d(x, w, stride = (self.group_size, 1, 1), dilation=self.dilation)

        x = einops.rearrange(x, 'b (c g) t h w -> b c (t g) h w', g=self.group_size)
        x = x * multiplicative

        return x, cache


class ConvVAE(MPConv):
    def forward(self,x, gain=1):
        batch_size = x.shape[0]
        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        x = super().forward(x,gain)
        return einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel=(8,3,3), group_size=1):
        super().__init__()

        self.conv3d0 = GroupCausal3DConvVAE(channels, channels,  kernel, group_size, dilation = (1,1,1))
        self.conv3d1 = GroupCausal3DConvVAE(channels, channels, kernel, group_size, dilation = (3,3,3))

        self.conv2d0 = ConvVAE(channels, channels,  kernel[1:], dilation = 1)
        self.conv2d1 = ConvVAE(channels, channels, kernel[1:], dilation = 3)
                                                                   
        self.weight_sum0 = nn.Parameter(torch.tensor(0.))
        self.weight_sum1 = nn.Parameter(torch.tensor(0.))

        # TODO: maybe add attention later
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

class EncoderDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_compression, spatial_compression, kernel, group_size, n_res_blocks, type='encoder'):
        super().__init__()
        self.downsample_block = UpDownBlock(time_compression, spatial_compression, 'up' if type=='decoder' else 'down')
        total_compression = self.downsample_block.total_compression

        self.decompression_block = GroupCausal3DConvVAE(in_channels, out_channels*total_compression, kernel, group_size//time_compression) if type=='decoder' else None
        self.compression_block  =  GroupCausal3DConvVAE(in_channels*total_compression, out_channels, kernel, group_size) if type in ['encoder', 'discriminator'] else None

        self.res_blocks = nn.ModuleList([ResBlock(out_channels, kernel, group_size) for _ in range(n_res_blocks)])

    def forward(self,x, cache=None):
        if cache is None: cache = {}

        if self.decompression_block is not None:
            x, cache['decompression_block'] = self.decompression_block(x, cache = cache.get('decompression_block', None))

        x = self.downsample_block(x)

        if self.compression_block is not None:
            x, cache['compression_block'] = self.compression_block(x, cache = cache.get('compression_block', None))

        for i, res_block in enumerate(self.res_blocks):
            x, cache[f'res_block_{i}'] = res_block(x, cache.get(f'res_block_{i}', None))

        return x, cache


class UpDownBlock:
    def __init__(self, time_compression, spatial_compression, direction):
        assert direction in ['up', 'down'], 'Invalid direction, expected up or down'

        self.direction = direction
        self.time_compression = time_compression
        self.spatial_compression = spatial_compression
        self.total_compression = time_compression*spatial_compression**2

    def __call__(self, x):
        if self.total_compression==1: return x

        if self.direction=='down':
            return einops.rearrange(x, 'b c (t tc) (h hc) (w wc) -> b (c tc hc wc) t h w', tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)

        return einops.rearrange(x, 'b (c tc hc wc) t h w -> b c (t tc) (h hc) (w wc)', tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)



class EncoderDecoder(nn.Module):
    def __init__(self, latent_channels, n_res_blocks, time_compressions = [1,2,2], spatial_compressions = [1,2,2], type='encoder'):
        super().__init__()
        assert type in ['encoder', 'decoder', 'discriminator'], 'Invalid type, expected encoder, decoder or discriminator'

        self.time_compressions = time_compressions
        self.spatial_compressions = spatial_compressions
        self.encoding_type = type

        group_sizes = np.cumprod(time_compressions)
        channels = [3, 4, 4, latent_channels] #assuming the input is always rgb
        kernel = (8,3,3)
        
        if type=='encoder':
            group_sizes = group_sizes[::-1]
            channels[-1]=channels[-1]*2
            self.logvar_multiplier = nn.Parameter(torch.tensor(0.))
        elif type=='decoder':
            channels = channels[::-1]
        elif type=='discriminator':
            group_sizes = group_sizes[::-1]
            assert latent_channels == 2, 'Discriminator should have 2 latent channels, one for each logit'

        in_channels, out_channels = channels[:-1], channels[1:]
        
        self.encoder_blocks = nn.ModuleList([EncoderDecoderBlock(in_channels[i], out_channels[i], time_compressions[i], spatial_compressions[i], kernel, group_sizes[i], n_res_blocks, type) for i in range(len(group_sizes))])

    def forward(self, x:Tensor, cache = None):
        if cache is None: cache = {}

        for i, block in enumerate(self.encoder_blocks):
            x, cache[f'encoder_block_{i}'] = block(x, cache.get(f'encoder_block_{i}', None))

        if self.encoding_type == 'decoder':
            return x, cache

        if self.encoding_type == 'discriminator':
            return x.mean(dim=[2,3,4]), cache

        mean, logvar = x.split(split_size=x.shape[1]//2, dim = 1)
        logvar = logvar*torch.exp(self.logvar_multiplier)

        return mean, logvar, cache



class VAE(nn.Module):
    def __init__(self, latent_channels, n_res_blocks, time_compressions=[1, 2, 2], spatial_compressions=[1, 2, 2]):
        super().__init__()
        self.encoder = EncoderDecoder(latent_channels, n_res_blocks, time_compressions, spatial_compressions, type='encoder')
        self.decoder = EncoderDecoder(latent_channels, n_res_blocks, time_compressions, spatial_compressions, type='decoder')

    def forward(self, x, cache=None):
        if cache is None:
            cache = {}
        
        # Encode input to get mean and log-variance
        mean, logvar, cache['encoder'] = self.encoder(x, cache.get('encoder', None))
        
        # Reparameterization trick: sample z from N(mean, std)
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        eps = torch.randn_like(std)    # Sample noise from standard normal
        z = mean + eps * std           # Latent vector
        
        # Decode latent vector to reconstruct input
        recon, cache['decoder'] = self.decoder(z, cache.get('decoder', None))
        
        return recon, mean, logvar, cache

        