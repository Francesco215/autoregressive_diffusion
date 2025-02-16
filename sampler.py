#%%
import copy
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader


from edm2.networks_edm2 import UNet, Precond
from edm2.loss import EDM2Loss, learning_rate_schedule
from edm2.loss_weight import MultiNoiseLoss
from edm2.mars import MARS
from edm2.gym_dataloader import GymDataGenerator, frames_to_latents, gym_collate_function
from edm2.phema import PowerFunctionEMA


from diffusers import AutoencoderKL

# import logging
# torch._logging.set_logs(dynamo=logging.INFO)

torch._dynamo.config.recompile_limit = 100
# Example usage:
n_clips = 100_000
micro_batch_size = 8 
batch_size = 32
accumulation_steps = batch_size//micro_batch_size
total_number_of_batches = n_clips // batch_size
total_number_of_steps = total_number_of_batches * accumulation_steps

num_workers = 8 
device = "cuda" if torch.cuda.is_available() else "cpu"
state_size = 8
original_env = "LunarLander-v3"
latent_channels = 0

model_id="stabilityai/stable-diffusion-2-1"
autoencoder = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).requires_grad_(False)
dataset = GymDataGenerator(state_size, original_env)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=gym_collate_function, num_workers=2)
batch = next(iter(dataloader))
#%%

latent_channels = autoencoder.config.latent_channels

unet = UNet(img_resolution=32, # Match your latent resolution
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
precond_state_dict = torch.load("lunar_lander_68.0M_trained.pt",map_location=device,weights_only=False)['model_state_dict']
precond.load_state_dict(precond_state_dict)
precond.to(device)

# g_net_state_dict = torch.load("lunar_lander_68.0M_trained.pt",map_location=device,weights_only=False)['model_state_dict']
# gunet = UNet(img_resolution=32, # Match your latent resolution
#             img_channels=latent_channels, # Match your latent channels
#             label_dim = 4,
#             model_channels=64,
#             channel_mult=[1,2,2,4],
#             channel_mult_noise=None,
#             channel_mult_emb=None,
#             num_blocks=3,
#             attn_resolutions=[8,4]
#             )
# g_net = Precond(gunet, use_fp16=True, sigma_data=sigma_data)
# g_net.load_state_dict(g_net_state_dict)
# g_net.to(device)
g_net=None
# Modify the sampler to collect intermediate steps and compute MSE
#%%
@torch.no_grad()
def edm_sampler_with_mse(
    net, cache, target=None,  # Added target for MSE calculation
    gnet=None, conditioning=None, num_steps=32, sigma_min=0.002, sigma_max=80, 
    rho=7, guidance=1, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32,
):
    batch_size, n_frames, channels, height, width = cache.get('shape', (None, None, None, None, None)) # TODO: change this
    device = net.device
    
    # Guided denoiser (same as original)
    def denoise(x, t, cache):
        cache = cache.copy()
        t = torch.ones(batch_size, 1, device=device, dtype=dtype) * t
        
        # cache = cache.copy()
        Dx, cache = net(x, t, conditioning, cache=cache)
        if guidance == 1:
            return Dx, cache
        ref_Dx, _ = gnet(x, t, conditioning, cache = {}) # TODO: play with the cache
        return ref_Dx.lerp(Dx, guidance), cache

    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=dtype, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * 
              (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    # Main sampling loop with MSE tracking
    x_next = torch.randn(batch_size, 1, channels, height, width, device=device) * t_steps[0]
    mse_values = []
    mse_pred_values = []
    if target is not None:
        target = target.to(dtype)
        x_next = x_next + target

    net.eval()
    if gnet is not None: gnet.eval()

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        # Noise injection step
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step
        if i == num_steps:
            x_pred, cache=denoise(x_hat, t_hat, cache)
        else:
            x_pred, _ = denoise(x_hat, t_hat, cache)
        d_cur = (x_hat - x_pred) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction
        if i < num_steps - 1:
            x_pred, _ = denoise(x_next, t_next, cache)
            d_prime = (x_next - x_pred) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Calculate MSE after each step
        if target is not None:
            mse_pred = torch.mean((x_pred - target) ** 2).item()
            mse = torch.mean((x_next - target) ** 2).item()
            mse_values.append(mse)
            mse_pred_values.append(mse_pred)

    net.train()
    return x_next, mse_values, mse_pred_values, cache

#%%

# Testing procedure
# Initialize dataloader and model
# batch ={"latents": torch.randn(2, 8, 16, 64, 64)}

# Prepare data
start = 0
num_samples = 4
frames, action, reward = batch
action = action.to(device)
frames = frames.to(device)
latents = frames_to_latents(autoencoder, frames)
# latents = batch["latents"][start:start+num_samples].to(device)
# text_embeddings = batch["text_embeddings"][start:start+num_samples].to(device)
context = latents[:, :-1]  # First frames (context)
target = latents[:, -1:]    # Last frame (ground truth)
precond.eval()
sigma = torch.ones(context.shape[:2], device=device) * 0.05
x, cache = precond(context, sigma, action[:,:-1])

#%%
# Run sampler with sigma_max=0.5 for initial noise level
_, mse_steps, mse_pred_values, _ = edm_sampler_with_mse(
    net=precond,
    cache=cache,
    target=target,
    sigma_max=3,  # Initial noise level matches our test
    num_steps=32,
    conditioning=action[:,-1:],
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

context.shape
# %%
for i in tqdm(range(8)):
    x, _, _, cache= edm_sampler_with_mse(precond, cache=cache, gnet=g_net, sigma_max = 80, num_steps=32, rho=7, guidance=1)
    latents = torch.cat((latents,x),dim=1)

print(x.shape)
# %%
torch.save(latents, "x.pt")

# %%
import torch
from edm2.gym_dataloader import latents_to_frames
import einops
x = torch.load("x.pt").to(device)
frames = latents_to_frames(autoencoder, x)

# %%
from matplotlib.pyplot import imshow

x = einops.rearrange(frames, 'b (t1 t2) h w c -> b (t1 h) (t2 w) c', t1=4)
imshow(x[3])


# %%
losses = torch.load("lunar_lander_38.0M_trained.pt",map_location=device,weights_only=False)['losses']
print(losses[-1])
plt.yscale('log')
plt.xscale('log')
plt.plot(losses)
# %%
