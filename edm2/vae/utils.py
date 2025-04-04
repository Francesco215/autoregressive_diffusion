
from functools import lru_cache

import torch
from torch.nn import functional as F


@lru_cache(maxsize=8)
@torch.no_grad()
def compute_multiplicative_time_wise(x_shape, kernel_size, dilation, group_size, device):
    # kernel_size = torch.tensor(kernel_size, device=device)
    group_index = torch.arange(x_shape[2], device=device)//group_size+1
    n_inputs_inside = torch.clip((group_index*group_size-1)//dilation[0]+1, max=kernel_size)
    multiplicative = torch.sqrt(kernel_size/n_inputs_inside)[:,None,None].detach()

    return multiplicative


@lru_cache(maxsize=8)
@torch.no_grad()
def compute_multiplicative_space_wise(x_shape, kernel_shape, padding, dilation, device):
    if dilation is None: dilation = tuple([1]*len(kernel_shape))

    if len(kernel_shape)==3:
        kernel_shape=kernel_shape[1:]
        dilation = dilation[1:]
    if padding is None:
        padding = (kernel_shape[0]//2, kernel_shape[0]//2, kernel_shape[1]//2, kernel_shape[1]//2)

    if dilation != (1,1):
        raise NotImplementedError("Dilation not supported yet")
    

    kernel_h, kernel_w = kernel_shape
    pad_h, _, pad_w, _ = padding

    height_indices = torch.arange(x_shape[-2], device=device)
    n_inputs_inside_h = torch.clamp(height_indices + 1 + pad_h, max=kernel_h) 
    multiplicative_height = torch.sqrt(kernel_h / n_inputs_inside_h).view(1, 1, -1, 1)
    multiplicative_height = multiplicative_height * multiplicative_height.flip(-2)

    width_indices = torch.arange(x_shape[-1], device=device)
    n_inputs_inside_w = torch.clamp(width_indices + 1 + pad_w, max=kernel_w)
    multiplicative_width = torch.sqrt(kernel_w / n_inputs_inside_w).view(1, 1, 1, -1)
    multiplicative_width = multiplicative_width * multiplicative_width.flip(-1)

    multiplicative = multiplicative_height * multiplicative_width
    if len(x_shape) == 5:
        multiplicative = multiplicative.unsqueeze(0)

    return multiplicative


def worst_k_percent_loss(recon, frames, percent=0.5):
        # Calculate element-wise MSE without reduction
        pixel_losses = F.mse_loss(recon, frames, reduction='none')
        
        # Flatten the tensor to easily find the top k values
        flat_losses = pixel_losses.flatten()
        
        # Calculate how many elements to keep (0.5% of total)
        k = max(1, int(flat_losses.numel() * (percent / 100.0)))
        
        # Get the worst k losses
        worst_k_losses, _ = torch.topk(flat_losses, k)
        
        # Compute the mean of the worst k losses
        return worst_k_losses.mean()