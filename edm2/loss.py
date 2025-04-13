# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import numpy as np
import torch
import einops
from .loss_weight import MultiNoiseLoss
# import dnnlib
# from torch_utils import distributed as dist
# from torch_utils import training_stats
# from torch_utils import misc

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

#respects time frames, loss per fram averaged, then per batch
# class EDM2Loss:
#     def __init__(self, P_mean=0.5, P_std=2., sigma_data=1., context_noise_reduction=0.1, topk_percentage=0.002):
#         self.P_mean = P_mean
#         self.P_std = P_std
#         self.sigma_data = sigma_data
#         self.context_noise_reduction = context_noise_reduction
#         self.topk_percentage = topk_percentage
#         assert context_noise_reduction >= 0 and context_noise_reduction <= 1, f"context_noise_reduction must be in [0,1], what are you doing? {context_noise_reduction}"
#         assert topk_percentage > 0 and topk_percentage <= 1, f"topk_percentage must be in (0,1], got {topk_percentage}"

#     def __call__(self, net, images, conditioning=None, sigma=None):
#         batch_size, n_frames, channels, height, width = images.shape    
#         assert net.training, "The model should be in training mode"
#         cat_images = torch.cat((images,images),dim=1).clone()
#         if conditioning is not None:
#             conditioning = torch.cat((conditioning,conditioning),dim=1).clone()

#         if sigma is None:
#             sigma_targets = (torch.randn(batch_size,n_frames,device=images.device) * self.P_std + self.P_mean).exp()
#             sigma_context = torch.rand(batch_size,1,device=images.device).expand(-1,n_frames).clone()*self.context_noise_reduction
#             sigma = torch.cat((sigma_context,sigma_targets),dim=1)
        
#         assert sigma.shape == (batch_size, n_frames*2), f"sigma shape is {sigma.shape} but should be {(batch_size, n_frames*2)}"

#         noise = einops.einsum(sigma, torch.randn_like(cat_images), 'b t, b t ... -> b t ...') 
#         out, _ = net(cat_images + noise, sigma, conditioning)
#         denoised = out[:,n_frames:]
        
#         # Calculate per-pixel MSE
#         pixel_losses = (denoised - images) ** 2
        
#         # Calculate the number of elements to keep in topk
#         elements_per_frame = channels * height * width
#         k = int(max(1, elements_per_frame * self.topk_percentage))
        
#         # Reshape for vectorized topk - reshape to [batch_size*n_frames, -1]
#         reshaped_losses = pixel_losses.reshape(batch_size * n_frames, -1)
        
#         # Get the top k% worst losses for each frame
#         topk_values, _ = torch.topk(reshaped_losses, k=k, dim=1)
        
#         # Calculate the mean of the top k% worst losses for each frame
#         frame_topk_losses = topk_values.mean(dim=1)
        
#         # Reshape back to [batch_size, n_frames]
#         frame_topk_losses = frame_topk_losses.reshape(batch_size, n_frames)
        
#         # Apply weighting with sigma
#         sigma = sigma[:,n_frames:]
#         weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
#         weighted_losses = frame_topk_losses * weight
        
#         # Average across frames first, then across batch
#         per_sample_loss = weighted_losses.mean(dim=1)
#         final_loss = per_sample_loss.mean()
        
#         # Store the unweighted average loss for tracking
#         un_weighted_avg_loss = frame_topk_losses.mean().detach().cpu().item()
        
#         # Update noise weight for adaptive weighting
#         net.noise_weight.add_data(sigma, frame_topk_losses)
#         mean_loss = net.noise_weight.calculate_mean_loss(sigma)
#         mean_loss = torch.clamp(mean_loss, min=1e-4, max=1)
#         final_loss = final_loss / mean_loss.mean()
        
#         return final_loss, un_weighted_avg_loss

# 
class EDM2Loss:
    def __init__(self, P_mean=0.5, P_std=2., sigma_data=1., context_noise_reduction=0.1, topk_percentage=0.002):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data  # Kept for backward compatibility
        self.context_noise_reduction = context_noise_reduction
        self.topk_percentage = topk_percentage
        assert context_noise_reduction >= 0 and context_noise_reduction <= 1, f"context_noise_reduction must be in [0,1], what are you doing? {context_noise_reduction}"
        assert topk_percentage > 0 and topk_percentage <= 1, f"topk_percentage must be in (0,1], got {topk_percentage}"

    def __call__(self, net, images, conditioning=None, sigma=None):
        batch_size, n_frames, channels, height, width = images.shape    
        assert net.training, "The model should be in training mode"
        cat_images = torch.cat((images,images),dim=1).clone()
        if conditioning is not None:
            conditioning = torch.cat((conditioning,conditioning),dim=1).clone()

        if sigma is None:
            sigma_targets = (torch.randn(batch_size,n_frames,device=images.device) * self.P_std + self.P_mean).exp()
            sigma_context = torch.rand(batch_size,1,device=images.device).expand(-1,n_frames).clone()*self.context_noise_reduction
            sigma = torch.cat((sigma_context,sigma_targets),dim=1)
        
        assert sigma.shape == (batch_size, n_frames*2), f"sigma shape is {sigma.shape} but should be {(batch_size, n_frames*2)}"

        noise = einops.einsum(sigma, torch.randn_like(cat_images), 'b t, b t ... -> b t ...') 
        out, _ = net(cat_images + noise, sigma, conditioning)
        denoised = out[:,n_frames:]
        
        # Calculate per-pixel MSE
        pixel_losses = (denoised - images) ** 2
        
        # Calculate the number of elements to keep in topk
        elements_per_frame = channels * height * width
        k = int(max(1, elements_per_frame * self.topk_percentage))
        
        # Reshape for vectorized topk - reshape to [batch_size*n_frames, -1]
        reshaped_losses = pixel_losses.reshape(batch_size * n_frames, -1)
        
        # Get the top k% worst losses for each frame
        topk_values, _ = torch.topk(reshaped_losses, k=k, dim=1)
        
        # Calculate the mean of the top k% worst losses for each frame
        frame_topk_losses = topk_values.mean(dim=1)
        
        # Reshape back to [batch_size, n_frames]
        frame_topk_losses = frame_topk_losses.reshape(batch_size, n_frames)
        
        # Average across frames first, then across batch
        per_sample_loss = frame_topk_losses.mean(dim=1)
        final_loss = per_sample_loss.mean()
        
        # Store the unweighted average loss for tracking
        un_weighted_avg_loss = frame_topk_losses.mean().detach().cpu().item()
        
        # Removed sigma weighting sections but kept same return format
        return final_loss, un_weighted_avg_loss
#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(current_step, ref_lr=1e-2, ref_step=7e4, rampup_steps=1e3):
    lr = ref_lr
    if ref_step > 0:
        lr /= np.sqrt(max(current_step / ref_step, 1))
    if rampup_steps > 0:
        lr *= min(current_step / rampup_steps, 1)
    return lr
