import numpy as np
import torch
import einops
from .loss_weight import MultiNoiseLoss

#----------------------------------------------------------------------------
# Adapted from Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".
class EDM2Loss:
    def __init__(self, P_mean=0.5, P_std=2., sigma_data=1., context_noise_reduction=0.1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.context_noise_reduction = context_noise_reduction
        assert context_noise_reduction >= 0 and context_noise_reduction <= 1, f"context_noise_reduction must be in [0,1], what are you doing? {context_noise_reduction}"

    def __call__(self, net, images, conditioning=None, sigma=None, just_2d=False):
        batch_size, n_frames, channels, height, width = images.shape    
        assert net.training, "The model should be in training mode"
        cat_images = images if just_2d else torch.cat((images,images),dim=1).clone()
        if conditioning is not None and not just_2d:
            conditioning = torch.cat((conditioning,conditioning),dim=1).clone()

        if sigma is None:
            sigma = (torch.randn(batch_size,n_frames,device=images.device) * self.P_std + self.P_mean).exp()
            if not just_2d:
                sigma_context = torch.rand(batch_size,1,device=images.device).expand(-1,n_frames).clone()*self.context_noise_reduction # reducing significantly the noise of the context tokens
                sigma = torch.cat((sigma_context,sigma),dim=1)
        
        noise = einops.einsum(sigma, torch.randn_like(cat_images), 'b t, b t ... -> b t ...') 
        out, _ = net(cat_images + noise, sigma, conditioning, just_2d)
        denoised = out[:,-n_frames:]
        errors = (denoised - images) ** 2
        losses = errors.mean(dim=(-1,-2,-3))
        # losses = top_losses(errors, fraction=3e-3)

        sigma = sigma[:,-n_frames:]
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 # the 0.5 factor is because the Karras paper is wrong
        losses = losses * weight

        un_weighted_avg_loss = losses.mean().detach().cpu().item()

        net.noise_weight.add_data(sigma, losses)
        mean_loss = net.noise_weight.calculate_mean_loss(sigma)
        # mean_loss = torch.clamp(mean_loss, min=1e-4, max=1)
        losses = losses / mean_loss
        return losses.mean(), un_weighted_avg_loss
#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def top_losses(errors:torch.Tensor, fraction:float):
    errors = errors.mean(dim=2)    
    errors = einops.rearrange(errors, 'b t h w -> b t (h w)')
    k = int(errors.shape[-1]*errors.shape[-2]*fraction)

    top_k = torch.topk(errors, k, dim =-1, sorted = False)
    return (top_k.values).mean(dim=-1) + errors.mean(dim=-1)




def learning_rate_schedule(current_step, ref_lr=1e-2, ref_step=7e4, rampup_steps=1e3):
    lr = ref_lr
    if ref_step > 0:
        lr /= np.sqrt(max(current_step / ref_step, 1))
    if rampup_steps > 0:
        lr *= min(current_step / rampup_steps, 1)
    return lr
