# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import pickle
import psutil
import numpy as np
import torch
from torch.nn import functional as F
import einops
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .loss_weight import MultiNoiseLoss
# import dnnlib
# from torch_utils import distributed as dist
# from torch_utils import training_stats
# from torch_utils import misc

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

class EDM2Loss:
    def __init__(self, P_mean=0.5, P_std=2., sigma_data=1., context_noise_reduction=0.1, noise_weight:MultiNoiseLoss = None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.context_noise_reduction = context_noise_reduction
        self.noise_weight = noise_weight
        assert context_noise_reduction >= 0 and context_noise_reduction <= 1, f"context_noise_reduction must be in [0,1], what are you doing? {context_noise_reduction}"

    def __call__(self, net, images, labels=None, use_loss_weight=True):
        batch_size, n_frames, channels, height, width = images.shape    
        images = torch.cat((images,images),dim=1)

        sigma_targets = (torch.randn(batch_size,n_frames,device=images.device) * self.P_std + self.P_mean).exp()
        sigma_context = torch.rand(batch_size,1,device=images.device).expand(-1,n_frames).clone() # reducing significantly the noise of the context tokens
        sigma = torch.cat((sigma_context,sigma_targets),dim=1)

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = einops.einsum(sigma, torch.randn_like(images), 'b t, b t ... -> b t ...') 
        denoised = net(images + noise, sigma, labels)
        losses = ((denoised[:,n_frames:] - images[:,n_frames:]) ** 2).mean(dim=(-1,-2,-3))
        losses = losses * weight[:,n_frames:]
        if self.noise_weight is not None:
            self.noise_weight.add_data(sigma[:,n_frames:], losses)
            if use_loss_weight:
                mean_loss = self.noise_weight.calculate_mean_loss(sigma[:,n_frames:])
                losses = losses / mean_loss**2 + 2*torch.log(mean_loss)
        return losses.mean()
#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr




#%%
import torch
a=torch.randn(1,10)