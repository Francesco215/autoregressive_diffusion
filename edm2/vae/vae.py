
import inspect
from urllib.parse import urlparse # To parse the S3 URI

import torch
from torch import nn, Tensor
from torch.nn import functional as F


import einops
import numpy as np

from ..utils import BetterModule, mp_silu




class GroupCausal3DConvVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, dilation = (1,1,1)):
        super().__init__()
        self.out_channels = out_channels
        self.group_size = group_size
        self.dilation = dilation
        # self.weight = NormalizedWeight(in_channels, out_channels*group_size, kernel, bias = True)
        self.conv3d = nn.Conv3d(in_channels, out_channels*group_size, kernel, dilation=dilation, stride=(group_size, 1, 1), bias=True)
        with torch.no_grad():
            w = self.conv3d.weight
            w[:,:,:-group_size] = 0
            self.conv3d.weight.copy_(w)

        kt, kw, kh = kernel
        dt, dw, dh = dilation
        self.image_padding = (dh * (kh//2), dh * (kh//2), dw * (kw//2), dw * (kw//2))
        self.time_padding_size = kt+(kt-1)*(dt-1)-self.group_size

    def forward(self, x, gain=1, cache=None):
        x = F.pad(x, pad = self.image_padding, mode="constant", value = 0)

        if cache is None:
            cache = x[:,:,:self.time_padding_size].clone().detach()

        x = torch.cat((cache, x), dim=-3)
        cache =  None if self.training else x[:,:,-self.time_padding_size:].clone().detach()

        x = self.conv3d(x)

        x = einops.rearrange(x, 'b (c g) t h w -> b c (t g) h w', g=self.group_size)

        return x, cache




class GroupNorm3D(nn.GroupNorm):
    def forward(self, input):
        batch_size = input.shape[0]
        input = einops.rearrange(input, "b c t h w -> (b t) c h w")
        output = super().forward(input)
        output = einops.rearrange(output, "(b t) c h w -> b c t h w", b=batch_size)
        return output

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel=(8,3,3), group_size=1):
        super().__init__()
        # self.norm_0 = GroupNorm3D(num_groups=1,num_channels=channels,eps=1e-6,affine=True)
        # self.norm_1 = GroupNorm3D(num_groups=1,num_channels=channels,eps=1e-6,affine=True)

        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.conv3d0 = GroupCausal3DConvVAE(channels, channels,  kernel, group_size, dilation = (1,1,1))
        self.conv3d1 = nn.Conv3d(channels, channels, kernel_size=(1,3,3), padding = (0,1,1))
        self.conv3d2 = nn.Conv3d(channels, channels, kernel_size=(1,3,3), padding = (0,1,1))

        scaling_factor = 4 ** -.25
        nn.init.kaiming_uniform_(self.conv3d0.conv3d.weight)
        nn.init.zeros_(self.conv3d0.conv3d.bias)
        self.conv3d0.conv3d.weight.data *= scaling_factor

        nn.init.kaiming_uniform_(self.conv3d1.weight)
        nn.init.zeros_(self.conv3d1.bias)
        self.conv3d1.weight.data *= scaling_factor

        nn.init.zeros_(self.conv3d2.weight)

    def forward(self, x, cache = None):
        if cache is None: cache = {}

        y, cache['conv3d_res0'] = self.conv3d0(x, cache=cache.get('conv3d_res0', None))
        y = self.act1(y)

        y = self.conv3d1(y)
        y = self.act2(y)

        y = self.conv3d2(y)

        x = x + y

        return x, cache


class EncoderDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_compression, spatial_compression, kernel, group_size, n_res_blocks, type='encoder'):
        super().__init__()
        self.updown_block = UpDownBlock(time_compression, spatial_compression, 'up' if type=='decoder' else 'down')
        total_compression = self.updown_block.total_compression

        self.decompression_block = nn.Conv3d(in_channels, out_channels*total_compression, kernel_size=(1,1,1)) if type=='decoder' else None
        self.compression_block  =  nn.Conv3d(in_channels*total_compression, out_channels, kernel_size=(1,1,1)) if type in ['encoder', 'discriminator'] else None

        self.res_blocks = nn.ModuleList([ResBlock(out_channels, kernel, group_size) for _ in range(n_res_blocks)])

    def forward(self,x, cache=None):
        if cache is None: cache = {}
        res = x.clone()

        if self.decompression_block:
            x = self.decompression_block(x)
            res = interpolate_channels(res, x.shape[1])

        x, res = self.updown_block(x), self.updown_block(res)

        if self.compression_block:
            x = self.compression_block(x)
            res = interpolate_channels(res, x.shape[1])

        x = x + res

        
        for i, res_block in enumerate(self.res_blocks):
            x, cache[f'res_block_{i}'] = res_block(x, cache.get(f'res_block_{i}', None))

        return x, cache

def interpolate_channels(x, cf):
    b, c, t, h, w = x.shape
    x = einops.rearrange(x, 'b c t h w -> b (t h w) c')
    x = F.interpolate(x, cf, mode='area')
    x = einops.rearrange(x, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
    return x

        




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
            return einops.rearrange(x, 'b c (t tc) (h hc) (w wc) -> b (tc hc wc c) t h w', tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)

        return einops.rearrange(x, 'b (tc hc wc c) t h w -> b c (t tc) (h hc) (w wc)', tc=self.time_compression, hc=self.spatial_compression, wc=self.spatial_compression)



class EncoderDecoder(nn.Module):
    def __init__(self, channels, n_res_blocks, time_compressions , spatial_compressions, type, learned_logvar):
        super().__init__()
        assert type in ['encoder', 'decoder', 'discriminator'], 'Invalid type, expected encoder, decoder or discriminator'
        assert len(channels) -1 == len(time_compressions) == len(spatial_compressions)

        self.time_compressions = time_compressions
        self.spatial_compressions = spatial_compressions
        self.encoding_type = type
        self.learned_logvar = learned_logvar

        channels = channels.copy()
        group_sizes = np.cumprod(time_compressions)

        if type=='encoder':
            group_sizes = group_sizes[::-1]
        elif type=='decoder':
            channels = channels[::-1]

        if learned_logvar:
            self.logvar_multiplier = nn.Parameter(torch.tensor(-2.))
            channels[-1] = channels[-1] * 2  

        in_channels, out_channels = channels[:-1], channels[1:]
        kernels = [(int(group_size)*2,3,3) for group_size in group_sizes]
        
        self.encoder_blocks = nn.ModuleList([EncoderDecoderBlock(in_channels[i], out_channels[i], time_compressions[i], spatial_compressions[i], kernels[i], group_sizes[i], n_res_blocks, type) for i in range(len(group_sizes))])

    def forward(self, x:Tensor, cache = None):
        if cache is None: cache = {}

        for i, block in enumerate(self.encoder_blocks):
            x, cache[f'encoder_block_{i}'] = block(x, cache.get(f'encoder_block_{i}', None))

        if not self.learned_logvar:
            return x, cache

        mean, logvar = x.split(split_size=x.shape[1]//2, dim=1)
        logvar = logvar*torch.exp(self.logvar_multiplier)
        return mean, logvar, cache



class VAE(BetterModule):
    def __init__(self, channels, n_res_blocks, time_compressions=[1, 2, 2], spatial_compressions=[1, 2, 2], std=None):
        super().__init__()
        
        self.latent_channels = channels[-1]
        self.encoder = EncoderDecoder(channels, n_res_blocks, time_compressions, spatial_compressions, type='encoder', learned_logvar=False)
        self.decoder = EncoderDecoder(channels, n_res_blocks, time_compressions, spatial_compressions, type='decoder', learned_logvar=True)

        self.time_compression = np.prod(time_compressions)
        self.spatial_compression = np.prod(spatial_compressions)

        # self.std=1.68 # this is when i pass z
        self.std=std #TODO put this as an argument, when it's untrained it should be none, but it must be specified when loading_from_pretrained

        # is it possible to put this inside of the super() class and avoid having it here?
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.kwargs = {arg: values[arg] for arg in args if arg != "self"}

    def forward(self, x, cache=None):
        if cache is None: cache = {}

        mean, cache['encoder'] = self.encode(x, cache.get('encoder', None))
        # Decode latent vector to reconstruct input
        noise_level = 0.1
        z = mean*(1-noise_level) + torch.randn_like(mean)*noise_level
        r_mean, r_logvar, cache['decoder'] = self.decode(z, cache.get('decoder', None))

        return r_mean, r_logvar, mean, cache
    
    def encode(self, x, cache=None):
        mean, cache = self.encoder(x, cache)
        return mean, cache
    
    def decode(self, z, cache=None):
        r_mean, r_logvar, cache = self.decoder(z, cache)
        return r_mean, r_logvar, cache
    


    @torch.no_grad()
    def encode_long_sequence(self, frames, cache=None, split_size=256):
        assert frames.shape[0]==1
        assert frames.dim()==5
        mean, logvar = None, None
        while frames.shape[2]>0:
            f = frames[:,:,:split_size].to(self.device)
            _,m,l,cache = self.encode(f,cache)
            if mean is None:
                mean, logvar = m, l
            else:
                mean = torch.cat((mean, m), dim = 2)
                logvar = torch.cat((logvar, l), dim = 2)
            frames = frames[:,:,split_size:]

        return mean, logvar
            

    # TODO: substitute this with encode_long_sequence. make sure it's also efficient
    torch.no_grad()
    def frames_to_latents(self, frames)->Tensor:
        """
        frames.shape: (batch_size, time, height, width, rgb)
        latents.shape: (batch_size, time, latent_channels, latent_height, latent_width)
        """
        batch_size = frames.shape[0]

        frames = frames / 127.5 - 1  # Normalize from (0,255) to (-1,1)
        frames = einops.rearrange(frames, 'b t h w c -> b c t h w')

        #split the conversion to not overload the GPU RAM
        split_size = 64
        for i in range (0, frames.shape[0], split_size):
            _, l, _, _ = self.encode(frames[i:i+split_size].to(self.device))
            if i == 0:
                latents = l
            else:
                latents = torch.cat((latents, l), dim=0)

        latents = einops.rearrange(latents, 'b c t h w -> b t c h w', b=batch_size)
        return latents/self.std


    # TODO: substitute this with decode_long_sequence. make sure it's also efficient
    @torch.no_grad()        
    def latents_to_frames(self,latents):
        """
            Converts latent representations to frames.
            Args:
                latents (torch.Tensor): A tensor of shape (batch_size, time, latent_channels, latent_height, latent_width) 
                                        representing the latent representations.
            Returns:
                numpy.ndarray: A numpy array of shape (batch_size, height, width * time, rgb) representing the decoded frames.
            Note:
                - The method uses an autoencoder to decode the latent representations.
                - The frames are rearranged and clipped to the range [0, 255] before being converted to a numpy array.
        """
        batch_size = latents.shape[0]
        latents = einops.rearrange(latents, 'b t c h w -> b c t h w')

        latents = latents * self.std

        #split the conversion to not overload the GPU RAM
        split_size = 16
        for i in range (0, latents.shape[0], split_size):
            l, _ = self.decode(latents[i:i+split_size])
            if i == 0:
                frames = l
            else:
                frames = torch.cat((frames, l), dim=0)

        frames = einops.rearrange(frames, 'b c t h w -> b t h w c', b=batch_size) 
        frames = torch.clip((frames + 1) * 127.5, 0, 255).cpu().detach().numpy().astype(int)
        return frames

        
