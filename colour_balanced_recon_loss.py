import torch
from torch.nn import functional as F


def color_balanced_recon_loss(recon_x, x, purple_blue_weight=1.0, epsilon=1e-8):
    """
    Calculate color-balanced reconstruction loss similar to DiscreteColorBalancedVAELoss.
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        purple_blue_weight: Weight for purple/blue category
        epsilon: Small value to avoid division by zero
        
    Returns:
        Weighted reconstruction loss
    """
    # Match dimensions if needed
    if recon_x.size() != x.size():
        recon_x = F.interpolate(recon_x, size=(x.size(-2), x.size(-1)), 
                              mode='bilinear', align_corners=False)
    
    batch_size = x.size(0)
    
    # Calculate squared error
    squared_error = (recon_x - x) ** 2
    
    # Reshape for efficient processing - adapt to your tensor shape
    # For 'b c t h w' format:
    error_flat = squared_error.reshape(batch_size, squared_error.size(1), -1)  # [B, C, THW]
    x_flat = x.reshape(batch_size, x.size(1), -1)  # [B, C, THW]
    
    # Define color bin mask functions
    def _is_black(x_flat):
        # For [-1,1] range (adjust from [0,1] range in original code)
        return torch.max(x_flat, dim=1, keepdim=True)[0] < -0.765  # 30/255.0 in [-1,1] scale
    
    def _is_white(x_flat):
        # For [-1,1] range
        return torch.min(x_flat, dim=1, keepdim=True)[0] > 0.725  # 220/255.0 in [-1,1] scale
    
    def _is_gray(x_flat):
        # For [-1,1] range
        r, g, b = x_flat[:, 0:1], x_flat[:, 1:2], x_flat[:, 2:3]
        max_diff = torch.max(torch.max(torch.abs(r-g), torch.abs(r-b)), torch.abs(g-b))
        return (max_diff < 0.235) & (r > -0.765) & (r < 0.725)  # 30/255.0 and 220/255.0 in [-1,1] scale
    
    def _is_purple_blue(x_flat):
        # For [-1,1] range
        r, g, b = x_flat[:, 0:1], x_flat[:, 1:2], x_flat[:, 2:3]
        return (b > torch.max(r, g)) & (b > 0.176) & (r > -0.294) & (r < 0.176) & (g > -0.373) & (g < 0.020)
        
    def _is_other(x_flat):
        # Colors that don't fall into the above categories
        black_mask = _is_black(x_flat)
        white_mask = _is_white(x_flat)
        gray_mask = _is_gray(x_flat)
        purple_blue_mask = _is_purple_blue(x_flat)
        
        return ~(black_mask | white_mask | gray_mask | purple_blue_mask)
    
    # Define color bins with their weights
    color_bins = {
        "Black": {"mask_function": _is_black, "weight": 1.0},
        "White": {"mask_function": _is_white, "weight": 1.0},
        "Gray": {"mask_function": _is_gray, "weight": 1.0},
        "Purple/Blue": {"mask_function": _is_purple_blue, "weight": purple_blue_weight},
        "Other Colors": {"mask_function": _is_other, "weight": 1.0}
    }
    
    # Calculate per-bin losses with weights
    bin_losses = []
    bin_weights = []
    
    for bin_name, bin_info in color_bins.items():
        # Create mask for this color bin
        mask_function = bin_info["mask_function"]
        color_mask = mask_function(x_flat)  # [B, 1, THW]
        
        # Count pixels in this bin
        num_pixels = torch.sum(color_mask) + epsilon
        
        if num_pixels > epsilon:
            # Average error for this color bin
            bin_error = torch.sum(error_flat * color_mask) / num_pixels
            bin_losses.append(bin_error)
            bin_weights.append(bin_info["weight"])
            
    # Apply weighted average across bins
    if bin_losses:
        bin_weights_tensor = torch.tensor(bin_weights, device=x.device)
        bin_losses_tensor = torch.stack(bin_losses)
        recon_loss = torch.sum(bin_losses_tensor * bin_weights_tensor) / torch.sum(bin_weights_tensor)
    else:
        recon_loss = torch.tensor(0.0).to(x.device)
    
    return recon_loss

# to use:
# recon_loss = color_balanced_recon_loss(recon, frames, purple_blue_weight=1.0)