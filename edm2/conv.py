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

    def forward(self, x, *args, **kwargs):
        w = self.weight().to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        x = F.conv2d(x, w, padding=(w.shape[-1]//2,))
        return x


class MPCausal3DGatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        assert len(kernel)==3
        self.last_frame_conv = MPConv(in_channels, out_channels, kernel[1:])
        kernel = (kernel[0]-1,kernel[1],kernel[2])
        self.weight = NormalizedWeight(in_channels, out_channels, kernel)
        self.gating = Gating()

    def forward(self, x, emb, batch_size, c_noise, cache=None, update_cache=False):
        if cache is None: cache = {}
        w = self.weight().to(x.dtype)

        image_padding = (0, w.shape[-2]//2, w.shape[-1]//2)

        # however variance preserving concatenatinon doesn't work because it will give different results depending if self.training is true
        causal_pad = torch.ones(batch_size, x.shape[1], w.shape[2], *x.shape[2:], device=x.device, dtype=x.dtype)
        causal_pad = cache.get('activations', causal_pad).clone()
        gating, updated_n_context_frames = self.gating(c_noise, cache.get('n_context_frames', 0)) # Change the context frames for inference
        if update_cache: cache['n_context_frames']=updated_n_context_frames

        # we do the 2d convolutions over the last frames
        last_frame_conv = self.last_frame_conv(x)

        if self.training:
            # we just take the context frames
            x, _ = einops.rearrange(x, '(b s t) c h w -> s b c t h w', b=batch_size, s=2).unbind(0)
        else: 
            x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)

        #pad context along the time dimention to make sure that it's causal
        context = torch.cat((causal_pad, x), dim=-3)
        if update_cache: cache['activations'] = context[:,:,-w.shape[2]:].clone().detach()
        # now we do the 3d convolutions over the previous frames of the context
        context = F.conv3d(context[:,:,:-1], w, padding=image_padding)

        if self.training:
            # we concatenate the results and reshape them to sum them back to the 2d convolutions
            context = torch.stack((context, context), dim=0)
            context = einops.rearrange(context, 's b c t h w -> (b s t) c h w')
        else:
            context = einops.rearrange(context, 'b c t h w -> (b t) c h w')

        return mp_sum(context, last_frame_conv, gating.flatten()), cache



class Gating(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([0.,0.]))
        self.mult = nn.Parameter(torch.tensor([1.5,-0.5]))
        self.activation = nn.Sigmoid()

    def forward(self, c_noise:Tensor, n_context_frames:int=0):
        batch_size, time_dimention = c_noise.shape
        if self.training: time_dimention = time_dimention//2
        positions = torch.arange(c_noise.numel(), device=c_noise.device) % time_dimention
        positions = einops.rearrange(positions, '(b t) -> b t', b=batch_size) + n_context_frames

        positions = positions.to(c_noise.dtype).log1p()
        state_vector = torch.stack([c_noise, positions], dim=-1)
        state_vector = (state_vector * self.mult + self.offset).sum(dim=-1)
        return self.activation(state_vector), n_context_frames+time_dimention # TODO:check this thing
