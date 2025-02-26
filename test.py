#%%
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from edm2.gym_dataloader import GymDataGenerator, frames_to_latents, gym_collate_function
from diffusers import AutoencoderKL, AutoencoderKLCogVideoX, AutoencoderKLMochi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# autoencoder = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(device).eval().requires_grad_(False)
autoencoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float32).to("cuda")

state_size = 96
env = "LunarLander-v3"
dataset = GymDataGenerator(state_size, env, training_examples=500, autoencoder_time_compression = 6)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=gym_collate_function, num_workers=16)
#%%
for i, batch in tqdm(enumerate(dataloader)):
    frames, actions, rewards = batch
    frames = frames.to(device)
    latents = frames_to_latents(autoencoder, frames)
    # latents = (latents-mean.to(latents))/std.to(latents)
    latents = latents.mean(dim=(0,1))
    if i==0:
        mean_latent= latents
    else:
        mean_latent += latents
    if i==100:
        break
mean_latent /= 100
#%%


torch.save(mean_latent, f'mean_{env}_latent.pt')

# %%

all_std = 0
axis_std = [0.,0.,0.,0.,0.]
for i, batch in tqdm(enumerate(dataloader)):
    frames, actions, rewards = batch
    frames = frames.to(device)
    latents = frames_to_latents(autoencoder, frames)
    # latents = (latents-mean.to(latents))/std.to(latents)
    for axis in range(5):
        axis_std[axis] += latents.std(dim=axis).mean().item()
    all_std += latents.std().item()
    
    if i==10:
        break
all_std /= 10
axis_std = [std/10 for std in axis_std]

print(f"Mean std: {all_std}, Axis std: {axis_std}")
# %%
axis_std

# %%
