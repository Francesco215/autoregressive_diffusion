# Code adapted from Nvidia EDM2 repository 

import inspect
from contextlib import nullcontext
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import einops

from .loss_weight import MultiNoiseLoss
from .utils import BetterModule, normalize, resample, mp_silu, mp_sum, mp_cat, MPFourier, bmult
from .conv import  MPConv, MPCausal3DGatedConv, Gating
from .attention import FrameAttention, VideoAttention



#---------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor               = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter      = [1,1],    # Resampling filter.
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

        self.conv_res0 = MPCausal3DGatedConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3,3])
        self.conv_res1 = MPCausal3DGatedConv(out_channels, out_channels, kernel=[3,3,3])
        # self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        # self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])

        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        if attention == 'video':
            self.attn = VideoAttention(out_channels, self.num_heads, attn_balance)
        else:
            self.attn = FrameAttention(out_channels, self.num_heads, attn_balance)

        if self.num_heads > 0: 
            assert (out_channels & (out_channels - 1) == 0) and out_channels != 0, f"out_channels must be a power of 2, got {out_channels}"

    def forward(self, x, emb, batch_size, c_noise, cache=None, update_cache=False, just_2d=False):
        if cache is None: cache = {}

        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y, cache['conv_res0'] = self.conv_res0(mp_silu(x), emb, batch_size, c_noise, cache.get('conv_res0', None), update_cache, just_2d) 
        # y = self.conv_res0(mp_silu(x)) 
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = bmult(y, c.to(y.dtype)) 
        y = mp_silu(y)
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y, cache['conv_res1'] = self.conv_res1(y, emb, batch_size, c_noise, cache.get('conv_res1', None), update_cache, just_2d) 
        # y = self.conv_res1(y) 

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        x, cache['attn'] = self.attn(x, batch_size, cache.get('attn', None), update_cache, just_2d)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x, cache

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(BetterModule):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        model_channels         = 192,       # Base multiplier for the number of channels.
        channel_mult           = [1,2,2,4], # Per-resolution multipliers for the number of channels.
        channel_mult_noise     = None,      # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb       = None,      # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks             = 3,         # Number of residual blocks per resolution.
        video_attn_resolutions = [8],       # List of resolutions with VideoAttention.
        frame_attn_resolutions = [16],      # List of resulutions with FrameAttention.
        label_balance          = 0.5,       # Balance between noise embedding (0) and class embedding (1).
        concat_balance         = 0.5,       # Balance between skip connections (0) and main path (1).
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
        self.out_res = Gating()

        # Embedding.
        self.emb_fourier_sigma = MPFourier(cnoise)
        self.emb_sigma = MPConv(cnoise, cemb, kernel=[]) 
        self.emb_fourier_time = MPFourier(cnoise)
        self.emb_time = MPConv(cnoise, cemb, kernel=[]) 
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level #bitwise right shift. it divides by 2 (img_resolution // 2)
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPCausal3DGatedConv(cin, cout, kernel=[3,3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                attention = 'video' if res in video_attn_resolutions else 'frame' if res in frame_attn_resolutions else False
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=attention, **block_kwargs)

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
                attention = 'video' if res in video_attn_resolutions else 'frame' if res in frame_attn_resolutions else False
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=attention, **block_kwargs)
        self.out_conv = MPCausal3DGatedConv(cout, img_channels, kernel=[3,3,3])

        # Saves the kwargs
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self.kwargs = {arg: values[arg] for arg in args if arg != "self"}

    def forward(self, x, c_noise, conditioning = None, cache:dict=None, update_cache=False, just_2d=False):
        batch_size, time_dimention = x.shape[:2]
        n_context_frames = cache.get('n_context_frames', 0)

        res = x.clone()
        out_res, updated_n_context_frames = self.out_res(c_noise, n_context_frames)
        if update_cache: cache['n_context_frames']=updated_n_context_frames

        # Reshaping
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        c_noise = einops.rearrange(c_noise, 'b t -> (b t)')

        # Time embedding
        frame_labels = torch.arange(time_dimention, device=x.device).repeat(batch_size) + n_context_frames
        frame_labels = frame_labels.log1p().to(c_noise.dtype) / 4 
        frame_embeddings = self.emb_time(self.emb_fourier_time(frame_labels))

        emb = self.emb_sigma(self.emb_fourier_sigma(c_noise))
        emb = mp_sum(emb, frame_embeddings, t=0.5)
        if self.emb_label is not None and conditioning is not None:
            conditioning = einops.rearrange(conditioning, 'b t -> (b t)')
            conditioning = F.one_hot(conditioning, num_classes=self.label_dim).to(c_noise.dtype)*self.label_dim**(0.5)
            conditioning = self.emb_label(conditioning)
            emb = mp_sum(emb, conditioning, t=1/3)
        emb = mp_silu(emb)

        c_noise = einops.rearrange(c_noise, '(b t) -> b t', b=batch_size)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x, cache['enc', name] = block(x, emb, batch_size, c_noise, cache=cache.get(('enc',name), None), update_cache=update_cache, just_2d=just_2d)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x, cache['dec', name] = block(x, emb, batch_size, c_noise, cache=cache.get(('dec',name), None),  update_cache=update_cache, just_2d=just_2d)
        x, cache['out_conv'] = self.out_conv(x, emb, batch_size, c_noise, cache=cache.get('out_conv', None), update_cache=update_cache, just_2d=just_2d)

        x = einops.rearrange(x, '(b t) c h w -> b t c h w', b=batch_size)
        x = mp_sum(x, res, out_res)
        return x, cache

        
    def no_sync(self):
        return nullcontext()

#----------------------------------------------------------------------------
# Preconditioning and uncertainty estimation.

class Precond(BetterModule):
    def __init__(self,
        unet,                   # UNet model.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.unet = unet
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.noise_weight = MultiNoiseLoss()

    def forward(self, x:Tensor, sigma:Tensor, conditioning:Tensor=None, force_fp32:bool=False, cache:dict=None, update_cache=False, just_2d=False):
        if cache is None: cache = {}
        cache['shape'] = x.shape
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        sigma = einops.rearrange(sigma, 'b t -> b t 1 1 1')

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.view(sigma.shape[:2]).log() / 4
 
        # Run the model.
        x_in = (c_in * x).to(dtype)
        F_x, cache = self.unet(x_in, c_noise, conditioning, cache, update_cache, just_2d)
        F_x = c_skip * x + c_out * F_x.to(torch.float32)
        return F_x, cache
    


#----------------------------------------------------------------------------

