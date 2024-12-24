#%%
import torch
import mediapy
import einops
import os
from tqdm import tqdm
from diffusers import AutoencoderKLMochi
from datasets import load_dataset
from utils import downsample_tensor

image_resolution=128
save_dir = f"encoded_latents/{image_resolution}x{image_resolution}"
dtype = torch.float32
# Assuming video_tensor is [num_frames, height, width, channels] and you want to
encoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=dtype).encoder.to("cuda").requires_grad_(False)
dataset = load_dataset("meta-ai-for-media-research/movie_gen_video_bench", split="test_with_generations", streaming=True)
# Directory to save encoded latents

os.makedirs(save_dir, exist_ok=True)
for i, example in tqdm(enumerate(dataset)):
    #%%
    # to display Movie Gen generated video and the prompt on jupyter notebook

    with open("tmp.mp4", "wb") as f:
        f.write(example["video"])
    # print(example["prompt"])

    video = mediapy.read_video("tmp.mp4")

    # %%
    v=torch.from_numpy(video)

    with torch.no_grad():
        v = downsample_tensor(v, image_resolution, image_resolution).to(dtype).cuda()
        v=v.unsqueeze(0)/255*2-1
        encoded=encoder(v)[0][0]
        encoded = einops.rearrange(encoded, 'c t h w -> t c h w')

    # Save encoded latents
    torch.save(encoded, os.path.join(save_dir, f"latent_{i+1}.pt"))

print("Encoding complete!")