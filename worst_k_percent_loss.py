import torch
import torch.functional as F


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