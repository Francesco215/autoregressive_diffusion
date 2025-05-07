# %%
from edm2.vae import VAE
from diffusers import AutoencoderKL
import torch


vae2d = AutoencoderKL.from_pretrained("THUDM/CogView4-6B", subfolder="vae")
vae=VAE(latent_channels=16, logvar_mode='learned')

#%%
vae._load_from_2D_model(vae2d)

#%%
for name, sub_module in vae.named_parameters():
    if "ResBlock" in name:
        print(sub_module)
    print(name)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from edm2.vae import GroupCausal3DConvVAE
                                                         

# ----- hyper‑params -----------------------------------------------------
in_ch        = 3
out_ch       = 5
group_size   = 2          # try other values (1, 4, …) if you like
k_spatial    = 3          # 3×3 spatial kernel
k3d          = (group_size * 2, k_spatial, k_spatial)   # causal = 2×group
padding2d    = k_spatial // 2                          # replicate 3D pad
device       = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------

# 2‑D reference conv
conv2d = nn.Conv2d(in_ch, out_ch,
                    kernel_size=k_spatial,
                    padding=padding2d,
                    bias=False).to(device)

# our 3‑D causal conv
conv3d = GroupCausal3DConvVAE(in_ch, out_ch, k3d, group_size).to(device)
conv3d._load_from_2D_module(conv2d.weight.data, None)

conv2d.eval();   conv3d.eval()

x = torch.randn(2,in_ch,8,16,16, device=device)                       # (B, C, T, H, W)

# --- forward -----------------------------------------------------------
with torch.no_grad():
    y3d, _ = conv3d(x)                                 # (B, C_out, T, H, W)

    # apply the 2‑D conv frame‑by‑frame
    y2d_frames = [conv2d(x[:, :, t]) for t in range(x.size(2))]
    y2d = torch.stack(y2d_frames, dim=2)               # (B, C_out, T, H, W)

# --- assertions --------------------------------------------------------
assert y3d.shape == y2d.shape, "Output shapes disagree"
assert torch.allclose(y3d, y2d, atol=1e-3), \
    "GroupCausal3DConvVAE is **not** frame‑wise equivalent to Conv2d"

# %%

# %%
