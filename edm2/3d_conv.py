import einops
import torch
import numpy as np
from .networks_edm2 import normalize
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

class MPCausal3DConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, batch_size, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization. for the gradients 
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        assert w.ndim == 5

        padding = (0, w.shape[1]//2, w.shape[2]//2)
        causal_pad = torch.ones(batch_size, w.shape[2]-1, *context.shape[2:], device=x.device, dtype=x.dtype)

        if self.training:
            # Warning: to understand this, read first how it works during inference
            # Warning: This is probably inefficiend, but I don't think there is a better way with current hardware.

            # this convolution is hard to do because each frame to be denoised has to do the concolution with the previous frames of the context
            # so we need to either have a really large kernel with lots of zeros in between (bad)
            # or we use the fact that the conv layers are linear (good). 
            
            # split the context and the frames to denoise
            context, denoise = einops.rearrange(x, '(b s t) c h w -> s b c t h w', b=batch_size, s=2).unbind(0)

            # we do the 2d convolutions over the last frames
            last_component_context = torch.nn.functional.conv2d(context, w=w[:,:,-1],  padding=w.shape[-1]//2)
            denoise = torch.nn.functional.conv2d(denoise, w=w[:,:,-1],  padding=w.shape[-1]//2)

            #pad context along the time dimention to make sure that it's causal
            context = torch.cat(causal_pad, context, dim = 1)
            # now we do the 3d convolutions over the previous frames of the context
            context = torch.nn.functional.conv3d(context, w=w[:,:,:-1], padding=padding)[:,:,:-1]

            # we use the fact that convolution is linear to sum the results of the 2d and 3d convolutions
            denoise = context + denoise
            context = context + last_component_context
            
            x = torch.cat((context, denoise), dim = 0)
            x = einops.rearrange(x, 's b c t h w -> (b s t) c h w')
            return x

        # during inference is much simpler
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)
        x = torch.cat(causal_pad, x, dim = 1)
        x = torch.nn.functional.conv3d(x, w, padding=padding)

        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        return x

