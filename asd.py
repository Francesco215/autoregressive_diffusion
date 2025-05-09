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
