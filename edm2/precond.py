# Code adapted from Nvidia EDM2 repository 

from contextlib import nullcontext
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import einops

from .loss_weight import MultiNoiseLoss
from .utils import BetterModule


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

    def forward(self, x:Tensor, sigma:Tensor, conditioning:Tensor=None, force_fp32:bool=False, cache:dict=None, update_cache=False):
        if cache is None: cache = {}
        cache['shape'] = x.shape # b t c h w
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
        F_x, cache = self.unet(x_in, c_noise, conditioning, cache, update_cache)
        F_x = c_skip * x + c_out * F_x.to(torch.float32)
        return F_x, cache
    


#----------------------------------------------------------------------------

