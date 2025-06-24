#%%
from edm2.networks_edm2 import UNet, Precond
import torch
device = "cuda"
my_net = UNet(img_resolution=64, # Match your latent resolution
            img_channels=4, # Match your latent channels
            label_dim = 1000,
            model_channels=128,
            channel_mult=[1,2,3,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=3,
            video_attn_resolutions=[8],
            frame_attn_resolutions=[16],
            ).to("cuda")
            

# %%
model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/'
model_name = 'edm2-img512-xs-2147483-0.135.pkl'

import dnnlib
import pickle
with dnnlib.util.open_url(f'{model_name}') as f:
    data = pickle.load(f)
net= data['ema'].to("cuda")

my_net.load_from_2d(net.unet)
net.unet.emb_label=None

my_net.save_to_state_dict("edm2.pt")

#%% -- Helper: Compare Outputs and Gradients --
def compare_outputs_and_grads(module_name, input_args, just_2d=True):
    print(f"\n--- Comparing {module_name} ---")

    # Clone and prepare inputs
    args1 = [a.detach().clone().requires_grad_() for a in input_args]
    args2 = [a.detach().clone().requires_grad_() for a in input_args]

    x1, label1, sigma1, c_noise1 = args1
    x2, label2, sigma2, c_noise2 = args2

    # Forward passes
    out1 = my_net.enc[module_name](x1, label1, sigma1, c_noise1, just_2d=just_2d)[0]
    out2 = net.unet.enc[module_name](x2, label2, sigma2, c_noise2)

    output_diff = (out1 - out2).abs().mean().item()
    print(f"Output diff: {output_diff:.6f}")

    # Compute gradients w.r.t. input x
    grad1 = torch.autograd.grad(out1.sum(), x1, retain_graph=True)[0]
    grad2 = torch.autograd.grad(out2.sum(), x2)[0]
    grad_diff = (grad1 - grad2).abs().mean().item()
    print(f"Gradient diff: {grad_diff:.6f}")



#%% -- Test Low-Level Module: '64x64_conv' --
n = 4
# x = torch.randn(n, 5, 16, 16, device=device)
# label = torch.randn(n, 512, device=device)
# sigma = torch.randn(1, n, device=device).exp() + 0.1
# c_noise = torch.randn(1, n, device=device).exp() + 0.1

# compare_outputs_and_grads('64x64_conv', [x, label, sigma, c_noise])

# #%% -- Test Mid-Level Module: '16x16_block0' --
# x = torch.randn(n, 256, 16, 16, device=device)
# compare_outputs_and_grads('16x16_block0', [x, label, sigma])


#%% -- Full Forward Pass Comparison --
x = torch.randn(1, n, 4, 64, 64, device=device).requires_grad_()
sigma = torch.randn(1, n, device=device).exp() + 0.1
label = None

# Custom forward (2D-only)
y1 = my_net(x, sigma, label, cache=None, update_cache=False, just_2d=True)[0]
grad1 = torch.autograd.grad(y1.sum(), x, retain_graph=True)[0]

# Official network
x2 = x.detach().clone().requires_grad_()[0]
sigma2 = sigma.detach().clone()[0]
y2 = net.unet(x2, sigma2, label)
grad2 = torch.autograd.grad(y2.sum(), x2)[0]

# Comparison
print("\n--- Full Forward Pass ---")
print(f"Output diff: {(y1 - y2).abs().mean().item():.6f}")
print(f"Gradient diff: {(grad1[0] - grad2).abs().mean().item():.6f}")

# %%
