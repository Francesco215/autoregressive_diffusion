import torch
from torch import nn, Tensor
from torch.nn import functional as F


import math
import einops
import numpy as np
from functools import lru_cache

from .utils import mp_sum, mp_silu
from .conv import MPConv, NormalizedWeight
from .attention import FrameAttention


@lru_cache(maxsize=8)
@torch.no_grad()
def compute_multiplicative_time_wise(x_shape, kernel_size, dilation, group_size, device):
    # kernel_size = torch.tensor(kernel_size, device=device)
    group_index = torch.arange(x_shape[2], device=device)//group_size+1
    n_inputs_inside = torch.clip((group_index*group_size-1)//dilation[0]+1, max=kernel_size)
    multiplicative = (torch.sqrt(kernel_size/n_inputs_inside)[:,None,None]).detach()

    return multiplicative


@lru_cache(maxsize=8)
@torch.no_grad()
def compute_multiplicative_space_wise(x_shape, kernel_shape, padding, dilation, device):
    if dilation is None: dilation = tuple([1]*len(kernel_shape))

    if len(kernel_shape)==3:
        kernel_shape=kernel_shape[1:]
        dilation = dilation[1:]
    if padding is None:
        padding = (kernel_shape[0]//2, kernel_shape[0]//2, kernel_shape[1]//2, kernel_shape[1]//2)

    if dilation != (1,1):
        raise NotImplementedError("Dilation not supported yet")
    

    kernel_h, kernel_w = kernel_shape
    pad_h, _, pad_w, _ = padding

    height_indices = torch.arange(x_shape[-2], device=device)
    n_inputs_inside_h = torch.clamp(height_indices + 1 + pad_h, max=kernel_h) 
    multiplicative_height = torch.sqrt(kernel_h / n_inputs_inside_h).view(1, 1, -1, 1)
    multiplicative_height = multiplicative_height * multiplicative_height.flip(-2)

    width_indices = torch.arange(x_shape[-1], device=device)
    n_inputs_inside_w = torch.clamp(width_indices + 1 + pad_w, max=kernel_w)
    multiplicative_width = torch.sqrt(kernel_w / n_inputs_inside_w).view(1, 1, 1, -1)
    multiplicative_width = multiplicative_width * multiplicative_width.flip(-1)

    multiplicative = multiplicative_height * multiplicative_width
    if len(x_shape) == 5:
        multiplicative = multiplicative.unsqueeze(0)

    return multiplicative

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
        # weight, bias = self.weight(gain)

        # multiplicative = 1.
        multiplicative = compute_multiplicative_space_wise(x.shape, self.conv3d.weight.shape[2:], self.image_padding, self.dilation, device=x.device)
        if cache is None:
            cache = torch.zeros(*x.shape[:2], self.time_padding_size, *x.shape[3:], device=x.device, dtype=x.dtype)
            multiplicative = compute_multiplicative_time_wise(x.shape, self.conv3d.weight.shape[2], self.dilation, self.group_size, device=x.device)

        x = torch.cat((cache, x), dim=-3)
        cache =  None if self.training else x[:,:,-self.time_padding_size:].clone().detach()

        # x = F.conv3d(x, weight, bias, stride = (self.group_size, 1, 1), dilation=self.dilation)
        x = self.conv3d(x)

        x = einops.rearrange(x, 'b (c g) t h w -> b c (t g) h w', g=self.group_size)
        x = x * multiplicative

        return x, cache



class Conv2DVAE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, dilation=dilation, padding=kernel[1]//2)
    
    def forward(self, x):
        batch_size = x.shape[0]
        multiplicative = compute_multiplicative_space_wise(x.shape, self.conv.weight.shape[2:], None, None, device = x.device)
        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv(x)
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)
        return x * multiplicative


class FrameAttentionVAE(FrameAttention):
    def forward(self,x):
        batch_size = x.shape[0]
        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        super().forward(x, batch_size)
        return einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel=(8,3,3), group_size=1):
        super().__init__()


        self.conv3d0 = GroupCausal3DConvVAE(channels, channels,  kernel, group_size, dilation = (1,1,1))
        self.conv3d1 = GroupCausal3DConvVAE(channels, channels, kernel, group_size, dilation = (1,1,1))

        self.conv2d0 = Conv2DVAE(channels, channels,  kernel[1:], dilation = 1)
        self.conv2d1 = Conv2DVAE(channels, channels,  kernel[1:], dilation = 1)
                                                                   
        # self.attn_block = FrameAttentionVAE(channels, num_heads=1) if group_size==1 else nn.Identity()
    
    def forward(self, x, cache = None):
        if cache is None: cache = {}

        t = x.clone()
        y, cache['conv3d_res0'] = self.conv3d0(x, cache=cache.get('conv3d_res0', None))
        y = y + self.conv2d0(x)
        y = mp_silu(y)

        t = y.clone()
        
        y, cache['conv3d_res1'] = self.conv3d1(t, cache=cache.get('conv3d_res1', None))
        y = y + self.conv2d1(t)
        y = mp_silu(y)

        x = x + y

        # x = self.attn_block(x)

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
        kernels = [(int(group_size)*2,3,3) for group_size in group_sizes]
        
        self.encoder_blocks = nn.ModuleList([EncoderDecoderBlock(in_channels[i], out_channels[i], time_compressions[i], spatial_compressions[i], kernels[i], group_sizes[i], n_res_blocks, type) for i in range(len(group_sizes))])

    def forward(self, x:Tensor, cache = None):
        if cache is None: cache = {}

        for i, block in enumerate(self.encoder_blocks):
            x, cache[f'encoder_block_{i}'] = block(x, cache.get(f'encoder_block_{i}', None))

        if self.encoding_type in ['decoder','discriminator']:
            return x, cache

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


# https://github.com/IamCreateAI/Ruyi-Models/blob/6c7b5972dc6e6b7128d6238bdbf6cc7fd56af2a4/ruyi/vae/ldm/modules/vaemodules/discriminator.py

class DiscriminatorBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.BatchNorm2d(in_channels)

        self.nonlinearity = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.downsampler = BlurPooling2D(out_channels, out_channels)
        else:
            self.downsampler = nn.Identity()

        self.norm2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.shortcut = nn.Sequential(
                BlurPooling2D(in_channels, in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = nn.Identity()

        self.spatial_downsample_factor = 2
        self.temporal_downsample_factor = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.downsampler(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class Discriminator2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels = (64,),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = out_channels
            is_final_block = i == len(block_out_channels) - 1

            self.blocks.append(
                DiscriminatorBlock2D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    output_scale_factor=math.sqrt(2),
                    add_downsample=not is_final_block,
                )
            )

        self.conv_norm_out = nn.BatchNorm2d(block_out_channels[-1])
        self.conv_act = nn.LeakyReLU(0.2)

        self.conv_out = nn.Conv2d(block_out_channels[-1], 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x

class Downsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_downsample_factor: int = 1,
        temporal_downsample_factor: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor

class BlurPooling3D(Downsampler):
    def __init__(self, in_channels: int, out_channels):
        if out_channels is None:
            out_channels = in_channels

        assert in_channels == out_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_downsample_factor=2,
            temporal_downsample_factor=2,
        )

        filt = torch.tensor([1, 2, 1], dtype=torch.float32)
        filt = torch.einsum("i,j,k -> ijk", filt, filt, filt)
        filt = filt / filt.sum()
        filt = filt[None, None].repeat(out_channels, 1, 1, 1, 1)

        self.register_buffer("filt", filt)
        self.filt: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        return F.conv3d(x, self.filt, stride=2, padding=1, groups=self.in_channels)

class BlurPooling2D(Downsampler):
    def __init__(self, in_channels: int, out_channels):
        if out_channels is None:
            out_channels = in_channels

        assert in_channels == out_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_downsample_factor=2,
            temporal_downsample_factor=1,
        )

        filt = torch.tensor([1, 2, 1], dtype=torch.float32)
        filt = torch.einsum("i,j -> ij", filt, filt)
        filt = filt / filt.sum()
        filt = filt[None, None].repeat(out_channels, 1, 1, 1)

        self.register_buffer("filt", filt)
        self.filt: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return F.conv2d(x, self.filt, stride=2, padding=1, groups=self.in_channels)    
        

class DiscriminatorBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.output_scale_factor = output_scale_factor

        self.norm1 = nn.GroupNorm(32, in_channels)

        self.nonlinearity = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.downsampler = BlurPooling3D(out_channels, out_channels)
        else:
            self.downsampler = nn.Identity()

        self.norm2 = nn.GroupNorm(32, out_channels)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        if add_downsample:
            self.shortcut = nn.Sequential(
                BlurPooling3D(in_channels, in_channels),
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )

        self.spatial_downsample_factor = 2
        self.temporal_downsample_factor = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)

        x = self.dropout(x)
        x = self.downsampler(x)
        x = self.conv2(x)

        return (x + shortcut) / self.output_scale_factor


class Discriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels = (64,),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1, stride=2)

        self.blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = out_channels
            is_final_block = i == len(block_out_channels) - 1

            self.blocks.append(
                DiscriminatorBlock3D(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    output_scale_factor=math.sqrt(2),
                    add_downsample=not is_final_block,
                )
            )

        self.conv_norm_out = nn.GroupNorm(32, block_out_channels[-1])
        self.conv_act = nn.LeakyReLU(0.2)

        self.conv_out = nn.Conv3d(block_out_channels[-1], 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.conv_in(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_out(x)

        return x

        
class MixedDiscriminator(nn.Module):
    def __init__(self, in_channels = 3, block_out_channels = (64,32)):
        super().__init__()
        self.discriminator2d = Discriminator2D(in_channels, (32,32,32))
        self.discriminator3d = Discriminator3D(in_channels, (32,32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3d = self.discriminator3d(x)
        
        x    = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        x    = self.discriminator2d(x)
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=x_3d.shape[0])

        # x_2d.shape = torch.Size([6, 2, 32, 256, 256])
        # x_3d.shape = torch.Size([6, 2, 16, 128, 128]) 
        x = torch.cat((x, x_3d), dim=2)

        return x
