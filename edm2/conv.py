import einops
import torch
import numpy as np
from .utils import normalize, mp_cat
from torch.autograd import Function

#----------------------------------------------------------------------------
# Magnitude-preserving weight (legacy)
class MPWeightFunction(Function):
    @staticmethod
    def forward(weight):
        return weight.data

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)  # Unpacking the tuple

    @staticmethod
    def backward(ctx, grad_output):
        (w,) = ctx.saved_tensors  # Unpack saved tensor

        # More efficient dot product computation
        dot = (grad_output * w).sum(dim=tuple(range(1, grad_output.dim())), keepdim=True)
        dot = dot / w.numel()

        # Efficiently subtract the weighted dot product
        grad_output = grad_output - w * dot

        return grad_output

#----------------------------------------------------------------------------
# Magnitude-preserving weight
class MPWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))
        self.weight_function = MPWeightFunction.apply
        self.normalize_weight()

    @torch.no_grad()
    def normalize_weight(self):
        w = normalize(self.weight.to(torch.float32))
        self.weight.copy_(w)

    def forward(self):
        return  self.weight_function(self.weight)



#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.mp_weight = MPWeight(in_channels, out_channels, kernel)

    def forward(self, x, gain=1, batch_size=None):
        w = self.mp_weight()
        x = x * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))



#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).
class MPCausal3DConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        assert len(kernel)==3
        self.mp_weight = MPWeight(in_channels, out_channels, kernel)

    def forward(self, x, batch_size, gain=1):
        w = self.mp_weight().to(x.dtype)
        image_padding = (0, w.shape[-2]//2, w.shape[-1]//2)

        # TODO: fix this
        # the multiplicative factor sqrt(0.5) is wrong. it is only an approximation of the correct analytical solution
        # to understand the theory check out the variance-preserving concatenation
        # however variance preserving concatenatinon doesn't work because it will give different results depending if self.training is true
        causal_pad = torch.ones(batch_size, x.shape[1], w.shape[2]-1, *x.shape[2:], device=x.device, dtype=x.dtype)*np.sqrt(0.5)

        if self.training:
            # Warning: to understand this, read first how it works during inference

            # this convolution is hard to do because each frame to be denoised has to do the convolution with the previous frames of the context
            # so we need to either have a really large kernel with lots of zeros in between (bad and dumb)
            # or we exploit linearity of the conv layers (good and smart). 
            
            # we do the 2d convolutions over the last frames
            last_frame_conv = torch.nn.functional.conv2d(x, w[:,:,-1],  padding=image_padding[1:])

            # we just take the context frames
            context, _ = einops.rearrange(x, '(b s t) c h w -> s b c t h w', b=batch_size, s=2).unbind(0)
            #pad context along the time dimention to make sure that it's causal
            context = torch.cat((causal_pad, context), dim=-3)

            # now we do the 3d convolutions over the previous frames of the context
            context = torch.nn.functional.conv3d(context[:,:,:-1], w[:,:,:-1], padding=image_padding)

            # we concatenate the results and reshape them to sum them back to the 2d convolutions
            context = torch.stack((context, context), dim=0)
            context = einops.rearrange(context, 's b c t h w -> (b s t) c h w')

            # we use the fact that the convolution is linear to sum the results of the 2d and 3d convolutions
            x = context + last_frame_conv
            x = x * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
            return x

        # during inference is much simpler
        x = einops.rearrange(x, '(b t) c h w -> b c t h w', b=batch_size)
        x = torch.cat((causal_pad, x), dim=-3)
        x = torch.nn.functional.conv3d(x, w, padding=image_padding)

        x = einops.rearrange(x, 'b c t h w -> (b t) c h w')
        return x


#----------------------------------------------------------------------------
# Magnitude-preserving weight (legacy)
# class Weight(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel):
#         super().__init__()
#         self.out_channels = out_channels
#         self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

#     def forward(self):
#         w = self.weight.to(torch.float32)
#         if self.training:
#             with torch.no_grad():
#                 self.weight.copy_(normalize(w)) # forced weight normalization
#         w = normalize(w) # traditional weight normalization. for the gradients 
#         return  w