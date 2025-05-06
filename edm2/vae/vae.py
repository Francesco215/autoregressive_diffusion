import inspect

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import einops
import numpy as np

from ..utils import BetterModule


class GroupCausal3DConvVAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, dilation = (1,1,1), stride = [1,1,1]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = group_size
        self.kernel = kernel
        self.stride = stride

        stride = stride.copy()
        stride[0]*=group_size

        self.conv3d = nn.Conv3d(in_channels, out_channels*group_size, kernel, stride, dilation=dilation, bias=True)
        with torch.no_grad():
            w = self.conv3d.weight
            # w[:,:,:-group_size] = 0
            self.conv3d.weight.copy_(w)

        # assert not (group_size==1 and stride[0]!=1)
        kt, kw, kh = kernel
        dt, dw, dh = dilation
        self.image_padding = (dh * (kh//2), dh * (kh//2), dw * (kw//2), dw * (kw//2))
        self.time_padding_size = kt+(kt-1)*(dt-1)-self.group_size

        # assert kt/group_size==2
        
    def forward(self, x, cache=None):
        x = F.pad(x, pad = self.image_padding, mode="constant", value = 0)

        if cache is None:
            cache = x[:,:,:self.time_padding_size].clone().detach()

        x = torch.cat((cache, x), dim=-3)
        cache =  None if self.training else x[:,:,-self.time_padding_size:].clone().detach()

        x = self.conv3d(x)

        x = einops.rearrange(x, 'b (g c) t h w -> b c (t g) h w', g=self.group_size)

        return x, cache

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, kernel={self.kernel}, group_size={self.group_size}, stride={self.stride})"

    @torch.no_grad()
    def _load_from_2D_state_dict(self, state_dict_2d: dict):
        w2d = state_dict_2d.get("weight", None)
        b2d = state_dict_2d.get("bias", None)

        w = torch.zeros_like(self.conv3d.weight)
        b = torch.zeros_like(self.conv3d.bias)

        kt = w.shape[2]              

        for g in range(self.group_size):
            w[self.out_channels*g : self.out_channels*(g+1), :, kt-self.group_size+g] = w2d.clone()
            if b2d is not None:
                b[self.out_channels*g : self.out_channels*(g+1)] = b2d

        self.conv3d.weight.copy_(w)
        self.conv3d.bias.copy_(b)



class GroupNorm3D(nn.GroupNorm):
    def forward(self, input):
        batch_size = input.shape[0]
        input = einops.rearrange(input, "b c t h w -> (b t) c h w")
        output = super().forward(input)
        output = einops.rearrange(output, "(b t) c h w -> b c t h w", b=batch_size)
        return output

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels:int=None , kernel=(8,3,3), group_size=1):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = GroupNorm3D(num_groups=8,num_channels=in_channels)
        self.conv1 = GroupCausal3DConvVAE(in_channels, out_channels,  kernel, group_size, dilation = (1,1,1))

        self.norm2 = GroupNorm3D(num_groups=8,num_channels=out_channels,eps=1e-6,affine=True)
        self.conv2 = GroupCausal3DConvVAE(out_channels, out_channels, kernel, group_size, dilation = (1,1,1))

        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = GroupCausal3DConvVAE(in_channels, out_channels, kernel, group_size, dilation = (1,1,1)) if in_channels!=out_channels else None

    def forward(self, x, cache = None):
        if cache is None: cache = {}

        y = self.norm1(x)
        y = self.nonlinearity(y)
        y, cache['conv3d_res0'] = self.conv1(y, cache=cache.get('conv3d_res0', None))

        y = self.norm2(y)
        y = self.nonlinearity(y)
        y, cache['conv3d_res1'] = self.conv2(y, cache=cache.get('conv3d_res1', None))

        if self.conv_shortcut is not None:
            x, cache['shortcut'] = self.conv_shortcut(x, cache = cache.get('conv_shortcut', None))
        x = x + y

        return x, cache

    def _load_from_2D_state_dict(self, state_dict_2D):
        self.norm1._load_from_state_dict(state_dict_2D['norm1'])
        self.norm2._load_from_state_dict(state_dict_2D['norm2'])

        self.conv1._load_from_2D_state_dict(state_dict_2D['conv1'])
        self.conv2._load_from_2D_state_dict(state_dict_2D['conv2'])

        self.conv_shortcut._load_from_2D_state_dict(state_dict_2D['conv_shortcut'])


class EncoderDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_compression, spatial_compression, kernel, group_size, n_res_blocks, type='encoder'):
        super().__init__()
        self.total_compression = time_compression*spatial_compression**2
        if self.total_compression!=1:
            self.updown_block = UpDownBlock(in_channels, in_channels, kernel, group_size, time_compression, spatial_compression, 'up' if type=='decoder' else 'down')

        self.res_blocks = nn.ModuleList([ResBlock(in_channels if i==0 else out_channels, out_channels, kernel, group_size) for i in range(n_res_blocks)])

    def forward(self,x, cache=None):
        if cache is None: cache = {}

        if self.total_compression !=1:
            x, cache['updown_block'] = self.updown_block(x, cache.get('updown_block', None))

        for i, res_block in enumerate(self.res_blocks):
            x, cache[f'res_block_{i}'] = res_block(x, cache.get(f'res_block_{i}', None))

        return x, cache


class UpDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, group_size, time_compression, spatial_compression, direction):
        super().__init__()
        assert direction in ['up', 'down'], 'Invalid direction, expected up or down'

        self.in_channels, self.out_channels, self.kernel, self.group_size = in_channels, out_channels, kernel, group_size
        self.direction = direction
        self.time_compression = time_compression
        self.spatial_compression = spatial_compression
        self.total_compression = time_compression*spatial_compression**2

        kernel = (kernel[0]//time_compression, kernel[1], kernel[2])
        if direction=='up':
            group_size = group_size//time_compression
            self.stride = [1,1,1]
        if direction=='down':
            self.stride = [time_compression, spatial_compression, spatial_compression]

        self.conv = GroupCausal3DConvVAE(in_channels, out_channels, kernel, group_size, stride = self.stride) 

    def __call__(self, x, cache=None):
        x, cache = self.conv(x, cache)

        if self.direction=='up' and self.total_compression !=1:
            x = F.interpolate(x, scale_factor=[self.time_compression, self.spatial_compression, self.spatial_compression], mode='nearest')

        return x, cache
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, time_compression={self.time_compression}, space_compression={self.spatial_compression}, kernel={self.kernel}, group_size={self.group_size}, stride={self.stride})"
    
    def _load_from_2D_state_dict(self,state_dict_2d):
        self.conv._load_from_2D_state_dict(state_dict_2d)


class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, block_channels = [128, 512, 1024, 1024, 1024], n_res_blocks = [3,3,3,3,2], time_compressions = [1,2,2,1,1], spatial_compressions = [1,2,2,2,1], type='encoder', logvar_mode='learned_constant'):
        super().__init__()
        assert type in ['encoder', 'decoder'], 'Invalid type, expected encoder or decoder'
        assert logvar_mode in ['learned_constant', 'learned'] or isinstance(logvar_mode, float), 'Invalid logvar_mode, expected learned_constant or learned of a float'
        assert len(block_channels) == len(time_compressions) == len(spatial_compressions)

        self.encoding_type = type
        self.logvar_mode = logvar_mode

        block_channels = [block_channels[0]]+block_channels
        group_sizes = np.cumprod(time_compressions).copy()

        if type=='encoder':
            group_sizes = group_sizes[-1]//group_sizes
            self.logvar_multiplier = nn.Parameter(torch.tensor(0.5))
            if logvar_mode == 'learned':
                out_channels = out_channels * 2
            elif isinstance(logvar_mode, float):
                self.logvar_multiplier = nn.Parameter(torch.tensor(logvar_mode), requires_grad=False)

        in_block_channels, out_block_channels = block_channels[:-1], block_channels[1:]
        kernels = [(int(group_size)*2,3,3) for group_size in group_sizes]


        self.conv_in = GroupCausal3DConvVAE(in_channels, in_block_channels[0], kernels[0], group_sizes[0])
        self.encoder_blocks = nn.ModuleList([EncoderDecoderBlock(in_block_channels[i], out_block_channels[i], time_compressions[i], spatial_compressions[i], kernels[i], group_sizes[i], n_res_blocks[i], type) for i in range(len(group_sizes))])
        self.conv_norm_out = nn.GroupNorm(num_groups=8, num_channels=block_channels[-1])
        self.conv_act = nn.SiLU()
        self.conv_out = GroupCausal3DConvVAE(block_channels[-1], out_channels, kernels[-1], group_sizes[-1])

    def forward(self, x:Tensor, cache = None):
        if cache is None: cache = {}

        x, cache['conv_in'] = self.conv_in.forward(x, cache = cache.get('conv_in', None))

        for i, block in enumerate(self.encoder_blocks):
            x, cache[f'encoder_block_{i}'] = block(x, cache.get(f'encoder_block_{i}', None))

        x = self.conv_norm_out(x)        
        x = self.conv_act(x)
        x, cache['conv_out'] = self.conv_out(x, cache=cache.get('conv_in', None))



        if self.encoding_type == 'decoder':
            return x, cache

        # Different logvar calculation methods
        if self.logvar_mode == 'learned':
            mean, logvar = x.split(split_size=x.shape[1]//2, dim=1)
        else:  
            mean, logvar = x, torch.ones_like(x)

        logvar = logvar*torch.exp(self.logvar_multiplier)
        return mean, logvar, cache



class VAE(BetterModule):
    def __init__(self, latent_channels, logvar_mode='learned', std=None):
        super().__init__()
        
        self.encoder = EncoderDecoder(in_channels=3, out_channels=latent_channels, block_channels = [128, 512, 1024, 1024, 1024], n_res_blocks = [3,3,3,3,2], spatial_compressions = [1,2,2,2,1], time_compressions=[1,2,2,1,1], type='encoder', logvar_mode=logvar_mode)
        self.decoder = EncoderDecoder(in_channels=latent_channels, out_channels=3, block_channels = [1024, 1024, 512, 128], n_res_blocks = [6,4,4,4], spatial_compressions = [1,2,2,2], time_compressions=[1,1,2,2], type='decoder')
        # self.encoder = EncoderDecoder(in_channels=3, out_channels=latent_channels, block_channels = [16, 32, 64, 64, 64], n_res_blocks = [2,2,2,2,1], spatial_compressions = [1,2,2,2,1], time_compressions=[1,2,2,1,1], type='encoder', logvar_mode=logvar_mode)
        # self.decoder = EncoderDecoder(in_channels=latent_channels, out_channels=3, block_channels = [64, 64, 64, 32, 16], n_res_blocks = [1,2,2,2,2], spatial_compressions = [1,1,2,2,2], time_compressions=[1,2,2,1,1], type='decoder')

        self.std=std
        # is it possible to put this inside of the super() class and avoid having it here?
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.kwargs = {arg: values[arg] for arg in args if arg != "self"}


    def forward(self, x, cache=None):
        if cache is None: cache = {}

        z, mean, logvar, cache['encoder'] = self.encode(x, cache.get('encoder', None))
        
        # Decode latent vector to reconstruct input
        recon, cache['decoder'] = self.decode(z, cache.get('decoder', None))

        return recon, mean, logvar, cache
    
    def encode(self, x, cache=None):
        mean, logvar, cache = self.encoder(x, cache)
        
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)    
        z = mean + eps * std           

        return z, mean, logvar, cache
    
    def decode(self, z, cache=None):
        recon, cache = self.decoder(z, cache)
        return recon, cache
    
    def set_std(self, std):
        self.kwargs['std']=std

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

        
