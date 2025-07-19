import numpy as np
import math
import einops

import torch
from torch import nn
from torch.nn import functional as F

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
    def __init__(self, in_channels = 6, block_out_channels = (64,32)):
        super().__init__()
        self.discriminator2d = Discriminator2D(in_channels, (32,32,32))
        self.discriminator3d = Discriminator3D(in_channels, (32,32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # it should return 1 if he thinks that the frames are in the first 3 channels, else, 0
        x_3d = self.discriminator3d(x)
        
        x    = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        x    = self.discriminator2d(x)
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=x_3d.shape[0])

        x = torch.cat((x, x_3d), dim=2)

        return x

    def cross_entropy(self, frames, recon_g, flip):

        # Adversarial loss for VAE/generator
        frames_recon = torch.cat([frames, recon_g], dim=1)
        recon_frames = torch.cat([recon_g, frames], dim=1)

        if flip==True:
            inputs = torch.cat([frames_recon,recon_frames], dim = 0)
        else:
            inputs = torch.cat([recon_frames,frames_recon], dim = 0).detach()
            
        logits = self(inputs)

        dims = logits.shape[2:]
        ones =  torch.ones (frames.shape[0], *dims, device=frames.device, dtype=torch.long)
        zeros = torch.zeros(frames.shape[0], *dims, device=frames.device, dtype=torch.long)
        targets = torch.cat([zeros, ones], dim = 0)
            
        # G wants D to misclassify
        return F.cross_entropy(logits, targets)/ np.log(2)

    def vae_loss(self, frames, recon_g):
        return self.cross_entropy(frames, recon_g, flip=True)

    def discriminator_loss(self, frames, recon_g):
        return self.cross_entropy(frames, recon_g, flip=False)