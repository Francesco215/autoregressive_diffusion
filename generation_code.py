#%%
from tqdm import tqdm
from matplotlib import pyplot as plt
import einops

import torch
from torch.utils.data import DataLoader

from edm2.gym_dataloaders.gym_dataloader_acrobot import GymDataGenerator, gym_collate_function
from edm2.networks_edm2 import UNet, Precond
from edm2.vae import VAE
from edm2.sampler import edm_sampler_with_mse
import os

# import logging
# torch._logging.set_logs(dynamo=logging.INFO)
# Example usage:
n_clips = 100_000
micro_batch_size = 4 
batch_size = 4
accumulation_steps = batch_size//micro_batch_size
total_number_of_batches = n_clips // batch_size
total_number_of_steps = total_number_of_batches * accumulation_steps
num_workers = 8 
original_env = "Acrobot-v1"
state_size = 24

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_channels=8
autoencoder = VAE.from_pretrained("saved_models/vae_acrobot_16000.pt").to("cuda")
dataset = GymDataGenerator(state_size, original_env, total_number_of_steps, autoencoder_time_compression = 4, return_anyways=False)
dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=gym_collate_function, num_workers=0)

unet=UNet.from_pretrained('saved_models/unet_8.0M_acrobot.pt')
g_net=None
precond = Precond(unet, use_fp16=True, sigma_data=1.).to(device)
# Modify the sampler to collect intermediate steps and compute MSE

# Testing procedure
# Initialize dataloader and model
# batch ={"latents": torch.randn(2, 8, 16, 64, 64)}

# Prepare data
batch = None
for i, batch in enumerate(dataloader):
    if i==100: break


with torch.no_grad():
    frames, actions, reward = batch
    frames, actions = torch.tensor(frames, device=device), torch.tensor(actions, device=device)
    latents = autoencoder.frames_to_latents(frames)
latents = latents[:,:5].to(device)
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
_, mse_steps, mse_pred_values, cache = edm_sampler_with_mse(
    net=precond,
    cache=cache,
    target=target,
    sigma_max=1,  # Initial noise level matches our test
    num_steps=64,
    conditioning=None,
    # gnet=g_net,
    rho = 7,
    guidance = 1,
    S_churn=0,
    S_noise=1,
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
    x, _, _, cache= edm_sampler_with_mse(precond, cache=cache, gnet=g_net, sigma_max = 80, num_steps=32, rho=7, guidance=1, S_churn=20)
    context = torch.cat((context,x),dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# frames = autoencoder.latents_to_frames(context)



# x = einops.rearrange(frames, 'b (t1 t2) h w c -> b (t1 h) (t2 w) c', t2=8)
# #set high resolution
# #%%
# plt.imshow(x[2])
# plt.axis('off')
# plt.savefig("lunar_lander.png",bbox_inches='tight',pad_inches=0, dpi=1000)


# # # %%
# # losses = torch.load("lunar_lander_38.0M_trained.pt",map_location=device,weights_only=False)['losses']
# # print(losses[-1])
# # plt.yscale('log')
# # plt.xscale('log')
# # plt.plot(losses)
# # # %%

# %%
import os
import numpy as np
from tqdm import tqdm
import einops
import cv2  # OpenCV for video writing

# After generating frames with the autoencoder
frames = autoencoder.latents_to_frames(context)

# Create output directory if it doesn't exist
output_dir = "generated_videos"
os.makedirs(output_dir, exist_ok=True)

# Process each batch sample as a separate video
for i in range(frames.shape[0]):
    # Rearrange dimensions for video writing (t h w c) format
    v = einops.rearrange(frames[i], 't h w c -> t h w c')
    
    # Convert to uint8 if needed (video writers expect uint8)
    if v.dtype != np.uint8:
        v = (v * 255).clip(0, 255).astype(np.uint8)
    
    # Write video file using OpenCV
    height, width, channels = v[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for better compatibility
    video_writer = cv2.VideoWriter(
        os.path.join(output_dir, f"lunar_lander_video_{i}.mp4"),
        fourcc, 30, (width, height)
    )
    
    for frame in v:
        # OpenCV expects BGR format, so convert if your frames are in RGB
        if channels == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    
    video_writer.release()

print(f"Videos saved to {output_dir}/")