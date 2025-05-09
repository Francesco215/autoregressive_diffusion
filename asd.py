#%%

#%%

        
from diffusers import AutoencoderKL
import torch
from edm2.vae import VAE





vae2d = AutoencoderKL.from_pretrained("THUDM/CogView4-6B", subfolder="vae")
x = torch.randn(2,3,64,64)                       # (B, C, T, H, W)
downsampler =vae2d.decoder.up_blocks[1].upsamplers[0]

#%%


vae = VAE(16)

# %%
from edm2.vae import VAE


vae._load_from_2D_model(vae2d)

#%%
# Check if the models have loaded properly
print("2D VAE parameters:", sum(p.numel() for p in vae2d.parameters()))
print("3D VAE parameters:", sum(p.numel() for p in vae.parameters()))

# Test encoding/decoding with a simple example
x_2d = torch.randn(2, 3, 64, 64)  # Create a random 2D input
x_3d = torch.randn(2, 3, 8, 64, 64)  # Create a random 3D input with temporal dimension

# Test the 2D model
with torch.no_grad():
    latent_2d = vae2d.encode(x_2d).latent_dist.sample()
    recon_2d = vae2d.decode(latent_2d).sample
    print("2D model encode-decode shapes:", x_2d.shape, "->", latent_2d.shape, "->", recon_2d.shape)

# Test the 3D model
with torch.no_grad():
    try:
        z_3d, mean_3d, logvar_3d, cache_3d = vae.encode(x_3d)
        recon_3d, cache_dec_3d = vae.decode(z_3d)
        print("3D model encode-decode shapes:", x_3d.shape, "->", z_3d.shape, "->", recon_3d.shape)
        print("3D model successful encode-decode!")
    except Exception as e:
        print("Error testing 3D model:", str(e))
                
        
#%%
import numpy as np
from edm2.vae.vae import ResBlock

def check_weights_similarity(model2d, model3d):
    """Compare weights between 2D and 3D models at key points"""
    results = []
    
    # 1. Check encoder first conv layer
    w2d = model2d.encoder.conv_in.weight.data
    w3d = model3d.encoder.conv_in.conv3d.weight.data
    kt = w3d.shape[2]
    group_size = getattr(model3d.encoder.conv_in, 'group_size', 4)  # Assuming group_size=4 based on earlier output
    slice_3d = w3d[:, :, kt-group_size]  # Take the last temporal slice
    
    if slice_3d.shape[0] > w2d.shape[0]:
        # If 3D has more output channels, compare only the first matching ones
        mse_first_conv = torch.mean((w2d - slice_3d[:w2d.shape[0], :w2d.shape[1]]) ** 2).item()
    else:
        mse_first_conv = torch.mean((w2d[:slice_3d.shape[0], :slice_3d.shape[1]] - slice_3d) ** 2).item()
    
    results.append(("Encoder First Conv", mse_first_conv))
    
    # 2. Check encoder norm_out
    w2d = model2d.encoder.conv_norm_out.weight.data
    w3d = model3d.encoder.conv_norm_out.weight.data
    
    if w2d.shape == w3d.shape:
        mse_norm_out = torch.mean((w2d - w3d) ** 2).item()
    else:
        mse_norm_out = f"Shape mismatch: 2D {w2d.shape} vs 3D {w3d.shape}"
    
    results.append(("Encoder Norm Out", mse_norm_out))
    
    # 3. Check encoder last conv layer
    w2d = model2d.encoder.conv_out.weight.data
    w3d = model3d.encoder.conv_out.conv3d.weight.data
    kt = w3d.shape[2]
    group_size = getattr(model3d.encoder.conv_out, 'group_size', 4)
    slice_3d = w3d[:, :, kt-group_size]
    
    if slice_3d.shape[0] > w2d.shape[0]:
        # If 3D has more output channels, compare only the first matching ones
        mse_last_conv = torch.mean((w2d - slice_3d[:w2d.shape[0], :w2d.shape[1]]) ** 2).item()
    else:
        mse_last_conv = torch.mean((w2d[:slice_3d.shape[0], :slice_3d.shape[1]] - slice_3d) ** 2).item()
    
    results.append(("Encoder Last Conv", mse_last_conv))
    
    # 4. Check decoder first conv
    w2d = model2d.decoder.conv_in.weight.data
    w3d = model3d.decoder.conv_in.conv3d.weight.data
    kt = w3d.shape[2]
    group_size = getattr(model3d.decoder.conv_in, 'group_size', 4)
    slice_3d = w3d[:, :, kt-group_size]
    
    if slice_3d.shape[0] > w2d.shape[0]:
        mse_dec_first = torch.mean((w2d - slice_3d[:w2d.shape[0], :w2d.shape[1]]) ** 2).item()
    else:
        mse_dec_first = torch.mean((w2d[:slice_3d.shape[0], :slice_3d.shape[1]] - slice_3d) ** 2).item()
    
    results.append(("Decoder First Conv", mse_dec_first))
    
    # 5. Check decoder last conv
    w2d = model2d.decoder.conv_out.weight.data
    w3d = model3d.decoder.conv_out.conv3d.weight.data
    kt = w3d.shape[2]
    group_size = getattr(model3d.decoder.conv_out, 'group_size', 4)
    slice_3d = w3d[:, :, kt-group_size]
    
    if slice_3d.shape[0] > w2d.shape[0]:
        mse_dec_last = torch.mean((w2d - slice_3d[:w2d.shape[0], :w2d.shape[1]]) ** 2).item()
    else:
        mse_dec_last = torch.mean((w2d[:slice_3d.shape[0], :slice_3d.shape[1]] - slice_3d) ** 2).item()
    
    results.append(("Decoder Last Conv", mse_dec_last))
    
    # 6. Sample ResBlocks (first one in encoder and decoder)
    # Get a resblock from each model
    resblock2d_enc = model2d.encoder.down_blocks[0].resnets[0]
    resblock3d_enc = None
    for name, module in model3d.named_modules():
        if isinstance(module, ResBlock) and 'encoder' in name:
            resblock3d_enc = module
            break
    
    if resblock3d_enc:
        # Check a sample conv weight
        w2d = resblock2d_enc.conv1.weight.data
        w3d = resblock3d_enc.conv1.conv2d.weight.data
        
        if w2d.shape == w3d.shape:
            mse_resblock = torch.mean((w2d - w3d) ** 2).item()
        else:
            mse_resblock = f"Shape mismatch: 2D {w2d.shape} vs 3D {w3d.shape}"
        
        results.append(("Encoder ResBlock Conv", mse_resblock))
    
    # Print results
    print("Weight similarity check (MSE, lower is better):")
    print("-" * 50)
    for name, mse in results:
        print(f"{name}: {mse}")
    
    # Overall assessment
    numerical_results = [mse for _, mse in results if isinstance(mse, (int, float))]
    if numerical_results:
        avg_mse = np.mean(numerical_results)
        print(f"\nAverage MSE across checked layers: {avg_mse:.8f}")
        
        if avg_mse < 1e-5:
            print("Conclusion: Weight loading appears SUCCESSFUL across checked layers.")
        elif avg_mse < 1e-3:
            print("Conclusion: Weight loading appears MOSTLY SUCCESSFUL with minor differences.")
        else:
            print("Conclusion: Significant differences in weights detected.")
    
# Run the check
check_weights_similarity(vae2d, vae)
# %%

#%%
#%%
def check_weight_transfer_coverage(model2d, model3d):
    """
    Analyze both:
    1. What percentage of 3D model weights were transferred from 2D model
    2. What percentage of 2D model weights were utilized in the transfer
    """
    # Count parameters in 2D model
    total_2d_params = sum(p.numel() for p in model2d.parameters())
    
    # Track 3D model statistics
    total_3d_params = 0
    total_transferred_params = 0
    zero_params = 0
    
    # Track parameters by module type
    module_stats = {}
    
    # For mapping 2D to 3D parameters
    param_mapping = {}
    
    # Step 1: Identify key parameters in both models to track transfer
    for name_2d, param_2d in model2d.named_parameters():
        module_type = name_2d.split('.')[0]  # encoder/decoder
        param_type = name_2d.split('.')[-1]  # weight/bias
        layer_type = name_2d.split('.')[-2] if len(name_2d.split('.')) > 1 else "unknown"
        
        key = f"{module_type}_{layer_type}_{param_type}"
        if key not in param_mapping:
            param_mapping[key] = {'2d_params': 0, '3d_params': 0, '3d_nonzero': 0}
        
        param_mapping[key]['2d_params'] += param_2d.numel()
    
    # Step 2: Process 3D model parameters
    for name_3d, param_3d in model3d.named_parameters():
        total_3d_params += param_3d.numel()
        non_zero = (param_3d.abs() > 1e-10).sum().item()
        total_transferred_params += non_zero
        zero_params += (param_3d.numel() - non_zero)
        
        # Update module stats
        module_type = name_3d.split('.')[-2] if '.' in name_3d else name_3d
        if module_type not in module_stats:
            module_stats[module_type] = {'total': 0, 'transferred': 0}
        module_stats[module_type]['total'] += param_3d.numel()
        module_stats[module_type]['transferred'] += non_zero
        
        # Try to map to 2D parameter
        module_type = name_3d.split('.')[0]  # encoder/decoder
        param_type = name_3d.split('.')[-1]  # weight/bias
        layer_type = name_3d.split('.')[-2] if len(name_3d.split('.')) > 1 else "unknown"
        
        key = f"{module_type}_{layer_type}_{param_type}"
        if key in param_mapping:
            param_mapping[key]['3d_params'] += param_3d.numel()
            param_mapping[key]['3d_nonzero'] += non_zero
    
    # Print overall results
    print(f"Weight Transfer Coverage Analysis:")
    print("-" * 70)
    print(f"2D model total parameters: {total_2d_params:,}")
    print(f"3D model total parameters: {total_3d_params:,}")
    print(f"Size ratio (3D/2D): {total_3d_params/total_2d_params:.2f}x")
    print(f"Non-zero parameters in 3D: {total_transferred_params:,} ({total_transferred_params/total_3d_params:.2%})")
    
    # Calculate theoretical maximum transfer
    # Simple estimate: if weights were perfectly transferred
    theoretical_max = min(total_2d_params, total_3d_params)
    print(f"Theoretical maximum transferable: {theoretical_max:,} ({theoretical_max/total_3d_params:.2%} of 3D model)")
    print(f"Transfer efficiency: {total_transferred_params/theoretical_max:.2%} of theoretical maximum")
    
    # Print parameter mapping stats
    print("\nParameter mapping analysis:")
    print("-" * 80)
    print(f"{'Parameter Type':<25} {'2D Params':<12} {'3D Params':<12} {'3D Non-Zero':<12} {'Coverage':<10} {'Utilization':<10}")
    print("-" * 80)
    
    total_2d_mapped = 0
    total_3d_mapped = 0
    total_3d_nonzero_mapped = 0
    
    # Sort by utilization (2D → 3D transfer percentage)
    for key, stats in sorted(param_mapping.items(), 
                             key=lambda x: x[1]['3d_nonzero']/max(1, min(x[1]['2d_params'], x[1]['3d_params']))):
        
        params_2d = stats['2d_params']
        params_3d = stats['3d_params']
        nonzero_3d = stats['3d_nonzero']
        
        # Coverage: What percentage of potential 3D params received weights?
        coverage = nonzero_3d / params_3d if params_3d > 0 else 0
        
        # Utilization: What percentage of 2D params were transferred?
        # For perfect transfer, we would transfer min(2d_params, 3d_params)
        utilization = nonzero_3d / min(params_2d, params_3d) if min(params_2d, params_3d) > 0 else 0
        
        total_2d_mapped += params_2d
        total_3d_mapped += params_3d
        total_3d_nonzero_mapped += nonzero_3d
        
        print(f"{key:<25} {params_2d:,} {params_3d:,} {nonzero_3d:,} {coverage:.2%} {utilization:.2%}")
    
    # Add a total row
    print("-" * 80)
    overall_coverage = total_3d_nonzero_mapped / total_3d_mapped if total_3d_mapped > 0 else 0
    overall_utilization = total_3d_nonzero_mapped / min(total_2d_mapped, total_3d_mapped) if min(total_2d_mapped, total_3d_mapped) > 0 else 0
    print(f"{'TOTAL':<25} {total_2d_mapped:,} {total_3d_mapped:,} {total_3d_nonzero_mapped:,} {overall_coverage:.2%} {overall_utilization:.2%}")
    
    # Temporal analysis for conv3d weights specifically
    temporal_stats = {'total_capacity': 0, 'nonzero_by_time': {}}
    
    # Analyze which temporal slices got weights
    for name, module in model3d.named_modules():
        if hasattr(module, 'conv3d') and hasattr(module.conv3d, 'weight'):
            w = module.conv3d.weight.data
            kt = w.shape[2]  # temporal kernel size
            
            for t in range(kt):
                if t not in temporal_stats['nonzero_by_time']:
                    temporal_stats['nonzero_by_time'][t] = 0
                
                # Count non-zero weights at this time slice
                slice_capacity = w[:, :, t].numel()
                nonzero_count = (w[:, :, t].abs() > 1e-10).sum().item()
                
                temporal_stats['nonzero_by_time'][t] += nonzero_count
                temporal_stats['total_capacity'] += slice_capacity
    
    # Print temporal analysis
    if temporal_stats['total_capacity'] > 0:
        print("\nTemporal weight distribution in 3D convolutions:")
        print("-" * 60)
        for t in sorted(temporal_stats['nonzero_by_time'].keys()):
            nonzero = temporal_stats['nonzero_by_time'][t]
            total_this_slice = temporal_stats['total_capacity'] / kt  # Capacity per time slice
            percentage_of_slice = nonzero / total_this_slice
            percentage_of_total = nonzero / temporal_stats['total_capacity']
            
            print(f"Time slice {t}: {nonzero:,} non-zero weights "
                  f"({percentage_of_slice:.2%} of slice capacity, "
                  f"{percentage_of_total:.2%} of total)")
    
    return {
        'total_3d': total_3d_params,
        'total_2d': total_2d_params,
        'nonzero_3d': total_transferred_params,
        'coverage': total_transferred_params / total_3d_params,
        'utilization': total_transferred_params / theoretical_max
    }

#%%
def zero_out_all_weights(model):
    """Set all weights in the model to exact zero to create a clean baseline"""
    total_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            # Save original parameter count
            param_count = param.numel()
            total_params += param_count
            
            # Set to zero
            param.data.zero_()
            
            # Verify
            assert torch.all(param.data == 0), f"Failed to zero out {name}"
    
    print(f"Zeroed out all {total_params:,} parameters in model")
    return model

# Create a fresh model
vae_zeroed = VAE(16)

# Zero out all weights
zero_out_all_weights(vae_zeroed)

# Verify all weights are zero
non_zero_before = sum((p != 0).sum().item() for p in vae_zeroed.parameters())
print(f"Non-zero parameters before transfer: {non_zero_before}")

# Transfer weights from 2D model
vae_zeroed._load_from_2D_model(vae2d)

# Check how many parameters are now non-zero
non_zero_after = sum((p != 0).sum().item() for p in vae_zeroed.parameters())
print(f"Non-zero parameters after transfer: {non_zero_after:,} ({non_zero_after/sum(p.numel() for p in vae_zeroed.parameters()):.2%})")

# Analyze specific temporal slices for conv3d weights
temporal_non_zeros = {}
for name, module in vae_zeroed.named_modules():
    if hasattr(module, 'conv3d') and hasattr(module.conv3d, 'weight'):
        w = module.conv3d.weight.data
        kt = w.shape[2]  # temporal kernel size
        
        for t in range(kt):
            if t not in temporal_non_zeros:
                temporal_non_zeros[t] = 0
            
            # Count non-zero weights at this time slice
            non_zero_slice = (w[:, :, t].abs() > 0).sum().item()
            temporal_non_zeros[t] += non_zero_slice

# Print temporal distribution
print("\nNon-zero weights by temporal slice:")
print("-" * 40)
for t in sorted(temporal_non_zeros.keys()):
    print(f"Time slice {t}: {temporal_non_zeros[t]:,} non-zero weights")

# Run the full analysis on the zeroed model
transfer_stats_zeroed = check_weight_transfer_coverage(vae2d, vae_zeroed)