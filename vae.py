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
dtype = torch.float16
# Assuming video_tensor is [num_frames, height, width, channels] and you want to
encoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=dtype).to(dtype).to("cuda").requires_grad_(False)
dataset = load_dataset("meta-ai-for-media-research/movie_gen_video_bench", split="test_with_generations")
# Directory to save encoded latents
os.environ['DISABLE_ADDMM_CUDA_LT'] = '1' 

torch.compile(encoder)
os.makedirs(save_dir, exist_ok=True)
#%%
for i, example in tqdm(enumerate(dataset)):
    # to display Movie Gen generated video and the prompt on jupyter notebook
    with open("tmp.mp4", "wb") as f:
        f.write(example["video"])
    # print(example["prompt"])

    video = mediapy.read_video("tmp.mp4")

    with torch.no_grad():
        v=torch.from_numpy(video)
        v = downsample_tensor(v, image_resolution, image_resolution).to(dtype).cuda()
        v = v[:,:200]
        v=v.unsqueeze(0)/255*2-1
        # encoded=encoder.encode(v)
        # encoded = einops.rearrange(encoded, 'c t h w -> t c h w')

        # # Save encoded latents
        # torch.save(encoded, os.path.join(save_dir, f"latent_{i+1}.pt"))
    break
print("Encoding complete!")

# %%
with torch.no_grad():
    encoded=encoder.encode(v)
# %%
encoded.latent_dist.logvar[0,-1]

# %%
with torch.no_grad():
    a = encoder.encoder(v)

# %%
a[0][0,-1]

# %%