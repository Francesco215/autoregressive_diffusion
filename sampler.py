#%%
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import einops
from diffusers import AutoencoderKLMochi

import torch
from torch.utils.data import DataLoader

from edm2.gym_dataloader import GymDataGenerator, frames_to_latents, gym_collate_function, latents_to_frames
from edm2.networks_edm2 import UNet, Precond
from edm2.vae import VAE
from edm2.sampler import edm_sampler_with_mse


# import logging
# torch._logging.set_logs(dynamo=logging.INFO)
# torch._dynamo.config.recompile_limit = 100
# Example usage:
n_clips = 100_000
micro_batch_size = 4 
batch_size = 4
accumulation_steps = batch_size//micro_batch_size
total_number_of_batches = n_clips // batch_size
total_number_of_steps = total_number_of_batches * accumulation_steps
num_workers = 8 
original_env = "LunarLander-v3"
state_size = 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_channels=12
# autoencoder = VAE.load_from_pretrained("saved_models/vae_4000.pt").to("cuda")
autoencoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")
dataset = GymDataGenerator(state_size, original_env, total_number_of_steps, autoencoder_time_compression = 6, return_anyways=False)
dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=gym_collate_function, num_workers=16)

unet = UNet(img_resolution=64, # Match your latent resolution
            img_channels=latent_channels, # Match your latent channels
            label_dim = 4,
            model_channels=64,
            channel_mult=[1,2,2,4],
            channel_mult_noise=None,
            channel_mult_emb=None,
            num_blocks=3,
            attn_resolutions=[8,4]
            )
print(f"Number of UNet parameters: {sum(p.numel() for p in unet.parameters())//1e6}M")
sigma_data = 1.
precond = Precond(unet, use_fp16=True, sigma_data=sigma_data)
precond_state_dict = torch.load("saved_models/lunar_lander_67.0M.pt",map_location=device,weights_only=False)['model_state_dict']
precond.load_state_dict(precond_state_dict, strict=False)
precond.to(device)

g_net=None
#%%

# Prepare data
batch = next(iter(dataloader))

with torch.no_grad():
    frames, actions, reward = batch
    frames = frames.to(device)
    actions = actions.to(device)
    latents = frames_to_latents(autoencoder, frames)/1.3
latents = latents[:,:2].to(device)
actions = None #if i%4==0 else actions.to(device)
# latents = batch["latents"][start:start+num_samples].to(device)
# text_embeddings = batch["text_embeddings"][start:start+num_samples].to(device)
context = latents[:, :-1]  # First frames (context)
target = latents[:, -1:]    # Last frame (ground truth)
precond.eval()
sigma = torch.ones(context.shape[:2], device=device) * 0.05
x, cache = precond(context, sigma)

#%%
# Run sampler with sigma_max=0.5 for initial noise level
_, mse_steps, mse_pred_values, _ = edm_sampler_with_mse(
    net=precond,
    cache=cache,
    target=target,
    sigma_max=3,  # Initial noise level matches our test
    num_steps=32,
    conditioning=None,
    # gnet=g_net,
    rho = 7,
    guidance = 1,
)
# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(mse_steps, marker='o', linestyle='-', label="MSE")
plt.plot(mse_pred_values, marker='o', linestyle='-', label="MSE (Predicted)")
plt.xlabel("Denoising Step")
plt.ylabel("MSE")
plt.yscale("log")
plt.title("Denoising Progress (Lower is Better)")
plt.grid(True)
plt.legend()
plt.show()
print(mse_steps[-1])
# %%
for i in tqdm(range(4)):
    x, _, _, cache= edm_sampler_with_mse(precond, cache=cache, gnet=g_net, sigma_max = 80, num_steps=32, rho=7, guidance=1)
    latents = torch.cat((latents,x),dim=1)

print(latents.shape)
# %%
torch.save(latents, "x.pt")

# %%
import torch
from edm2.gym_dataloader import GymDataGenerator, frames_to_latents, gym_collate_function, latents_to_frames
import einops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.load("x.pt").to(device)
with torch.no_grad():
    frames = latents_to_frames(autoencoder, x[:2])

# %%

from matplotlib import pyplot as plt
# frames = frames
f = frames[:,:24]
x = einops.rearrange(f, 'b (t1 t2) h w c -> b (t1 h) (t2 w) c', t2=12)
#set high resolution
plt.imshow(x[1])
plt.axis('off')
plt.savefig("lunar_lander.png",bbox_inches='tight',pad_inches=0, dpi=1000)


# # %%
# losses = torch.load("lunar_lander_38.0M_trained.pt",map_location=device,weights_only=False)['losses']
# print(losses[-1])
# plt.yscale('log')
# plt.xscale('log')
# plt.plot(losses)
# # %%

# %%
