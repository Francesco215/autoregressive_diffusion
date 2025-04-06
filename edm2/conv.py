import einops
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np
from .utils import mp_sum, normalize, mp_cat

class NormalizedWeight(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization. for the gradients 
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        return w

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation=1):
        super().__init__()
        self.out_channels = out_channels
        self.weight = NormalizedWeight(in_channels, out_channels, kernel)

        self.dilation = dilation
        self.padding = [dilation*(kernel[-1]//2)]*4 if len(kernel)!=0 else None

    def forward(self, x, gain=1, batch_size=None):
        w = self.weight(gain).to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        x = F.conv2d(x, w, padding=(w.shape[-1]//2,))
        return x



#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).
class MPCausal3DConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        assert len(kernel)==3
        self.weight = NormalizedWeight(in_channels, out_channels, kernel)

    def forward(self, x, emb, batch_size, gain=1, cache=None):
        # x.shape = (batch_size, time), channels, height, width

        # x.shape = batch_size, channels, time, height, width 
        w = self.weight(gain).to(x.dtype)

        image_padding = (0, w.shape[-2]//2, w.shape[-1]//2)

        # TODO: there should be a multiplicative factor in the causal_pad. 
        # to understand the theory check out the variance-preserving concatenation
        # however variance preserving concatenatinon doesn't work because it will give different results depending if self.training is true
        # causal_pad = einops.rearrange(x, '(b t) c ... -> b c t ...', b = batch_size)[:,:,0].unsqueeze(2).repeat(1,1,w.shape[2]-1,1,1).clone()
        # causal_pad = torch.ones(batch_size, x.shape[1], w.shape[2]-1, *x.shape[2:], device=x.device, dtype=x.dtype)
        # causal_pad = torch.zeros(batch_size, x.shape[1], w.shape[2]-1, *x.shape[2:], device=x.device, dtype=x.dtype)
        causal_pad = torch.randn(batch_size, x.shape[1], w.shape[2]-1, *x.shape[2:], device=x.device, dtype=x.dtype)

        if self.training:
            # Warning: to understand this, read first how it works during inference

            # this convolution is hard to do because each frame to be denoised has to do the convolution with the previous frames of the context
            # so we need to either have a really large kernel with lots of zeros in between (bad and dumb)
            # or we exploit linearity of the conv layers (good and smart). 
            
            # we do the 2d convolutions over the last frames
            last_frame_conv = F.conv2d(x, w[:,:,-1],  padding=image_padding[1:])

            # we just take the context frames
            context, _ = einops.rearrange(x, '(b s t) c h w -> s b c t h w', b=batch_size, s=2).unbind(0)
            #pad context along the time dimention to make sure that it's causal
            context = torch.cat((causal_pad, context), dim=-3)

            # now we do the 3d convolutions over the previous frames of the context
            context = F.conv3d(context[:,:,:-1], w[:,:,:-1], padding=image_padding)

            # we concatenate the results and reshape them to sum them back to the 2d convolutions
            context = torch.stack((context, context), dim=0)
            context = einops.rearrange(context, 's b c t h w -> (b s t) c h w')

            # we use the fact that the convolution is linear to sum the results of the 2d and 3d convolutions
            x = context + last_frame_conv

            return x, None

        if cache is None:
            cache = causal_pad

        # during inference is much simpler
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)
        x = torch.cat((cache, x), dim=-3)
        cache = x[:,:,-(w.shape[2]-1):].clone()

        x = F.conv3d(x, w, padding=image_padding)

        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        return x, cache.detach()

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).
class MPCausal3DGatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        assert len(kernel)==3
        self.last_frame_conv = MPConv(in_channels, out_channels, kernel[1:])
        kernel[0]-=1
        self.weight = NormalizedWeight(in_channels, out_channels, kernel)
        self.gating = Gating()

    def forward(self, x, emb, batch_size, gain=1, cache=None):
        if cache is None: cache = {}
        w = self.weight(gain).to(x.dtype)

        image_padding = (0, w.shape[-2]//2, w.shape[-1]//2)

        # however variance preserving concatenatinon doesn't work because it will give different results depending if self.training is true
        causal_pad = torch.zeros(batch_size, x.shape[1], w.shape[2], *x.shape[2:], device=x.device, dtype=x.dtype)
        causal_pad = cache.get('activations', causal_pad)
        gating, cache['n_context_frames'] = self.gating((batch_size, x.shape[0]//batch_size), cache.get('n_context_frames', 0)) # Change the context frames for inference

        # we do the 2d convolutions over the last frames
        last_frame_conv = self.last_frame_conv(x)

        if self.training:
            # we just take the context frames
            x, _ = einops.rearrange(x, '(b s t) c h w -> s b c t h w', b=batch_size, s=2).unbind(0)
        else: 
            x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)

        #pad context along the time dimention to make sure that it's causal
        context = torch.cat((causal_pad, x), dim=-3)
        cache['activations'] = context[:,:,-w.shape[2]:].clone().detach()
        # now we do the 3d convolutions over the previous frames of the context
        context = F.conv3d(context[:,:,:-1], w, padding=image_padding)

        if self.training:
            # we concatenate the results and reshape them to sum them back to the 2d convolutions
            context = torch.stack((context, context), dim=0)
            context = einops.rearrange(context, 's b c t h w -> (b s t) c h w')
        else:
            context = einops.rearrange(context, 'b c t h w -> (b t) c h w')

        return mp_sum(context, last_frame_conv, gating), cache



class Gating(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([0.]))
        self.mult = nn.Parameter(torch.tensor([1.]))
        self.activation = nn.Sigmoid()

    def forward(self, sequence_shape, n_context_frames:int=0, dtype=torch.float16, device = "cuda" if torch.cuda.is_available() else "cpu"):
        batch_size, time_dimention = sequence_shape
        numel = batch_size * time_dimention
        if self.training: time_dimention = time_dimention//2

        positions = torch.arange(numel, device=device) % time_dimention
        positions = positions + n_context_frames

        positions = positions.to(dtype).log1p()
        state_vector = positions * self.mult + self.offset

        return self.activation(state_vector), n_context_frames+time_dimention # TODO:check this thing
