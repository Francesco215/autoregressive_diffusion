#%%
import torch
import mediapy
from diffusers import AutoencoderKLMochi
from datasets import load_dataset
from utils import downsample_tensor
import os

# Assuming video_tensor is [num_frames, height, width, channels] and you want to
encoder = AutoencoderKLMochi.from_pretrained("genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.float16).encoder.to("cuda").requires_grad_(False)
dataset = load_dataset("meta-ai-for-media-research/movie_gen_video_bench", split="test_with_generations", streaming=True)
# Directory to save encoded latents
save_dir = "encoded_latents"
os.makedirs(save_dir, exist_ok=True)
for i, example in enumerate(dataset):
    print("data downloaded")
    #%%
    # to display Movie Gen generated video and the prompt on jupyter notebook

    with open("tmp.mp4", "wb") as f:
        f.write(example["video"])
    print(example["prompt"])

    video = mediapy.read_video("tmp.mp4")

    # %%
    import torch
    v=torch.from_numpy(video)
    v = downsample_tensor(v, 256, 256).requires_grad_(False).to(torch.float16).cuda()
    v=v/255*2-1
    v=v.unsqueeze(0)

    with torch.no_grad():

        encoded=encoder(v)[0]

    # Save encoded latents
    torch.save(encoded, os.path.join(save_dir, f"latent_{i+1}.pt"))
    print(f"Saved latent for video {i+1}")

    # Optional: Stop after encoding a certain number of videos for testing
    # if i == 4:
    #     break

print("Encoding complete!")