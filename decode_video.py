#%%
import os
from huggingface_hub import login, snapshot_download
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import torch
import numpy as np
import einops
from torchvision.io import write_video

x = torch.load("sampled_latents.pt", map_location="cuda")

def download_pretrained_ckpts(local_dir: str, model_name: str):
    """Download pretrained checkpoints from huggingface."""

    login()
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=f"nvidia/{model_name}", local_dir=local_dir)

def get_decoder(model_name: str  = "Cosmos-Tokenizer-DV4x8x8"):
    """Get the decoder for the given model name.
    model_name can be "Cosmos-Tokenizer-DV4x8x8", "Cosmos-Tokenizer-DV8x8x8", or "Cosmos-Tokenizer-DV8x16x16"."""

    local_dir = f"./pretrained_ckpts/{model_name}"
    if not os.path.exists(local_dir):
        download_pretrained_ckpts(local_dir, model_name)
    decoder = CausalVideoTokenizer(checkpoint_dec=f"{local_dir}/decoder.jit")
    return decoder


_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

def unclamp_video(input_tensor: torch.Tensor) -> torch.Tensor:
    """Unclamps tensor in [-1,1] to video(dtype=np.uint8) in range [0..255]."""
    tensor = (input_tensor.float() + 1.0) / 2.0
    tensor = tensor.clamp(0, 1).cpu().numpy()
    return (tensor * _UINT8_MAX_F + 0.5).astype(np.uint8)
# %%
decoder = get_decoder(model_name = "Cosmos-Tokenizer-CV4x8x8").cuda()
decoder_dtype = next(iter(decoder.parameters())).dtype

#%%
y = einops.rearrange(x,'b t c h w -> b c t h w').to(decoder_dtype)
# %%
video = unclamp_video(decoder.decode(y))
for i in range(video.shape[0]):
    output_dir = "output"
    v = einops.rearrange(video[i], 'c t h w -> t h w c')
    write_video(os.path.join(output_dir, f"video_{i}.mp4"), v, fps=30)


# %%
