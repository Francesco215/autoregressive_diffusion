# %%

import os
import io
import einops

import torch
from torch.utils.data import DataLoader, IterableDataset

from abc import abstractmethod
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Optional

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def deserialize_tensor(
    serialized_tensor: bytes, device: Optional[str] = "cpu"
) -> torch.Tensor:
    return torch.load(
        io.BytesIO(serialized_tensor),
        weights_only=True,
        map_location=torch.device(device) if device else None,
    )

class OpenVidDataset(IterableDataset):
    def __init__(self):
        self.dataset = load_dataset("fal/cosmos-openvid-1m", data_dir="continuous", split="train", streaming=True, cache_dir="/tmp/datasets_cache")

        # self.mean, self.std = 0.051, 0.434
        self.mean, self.std = 0, 1 
        self.mean = torch.tensor([ 0.1001,  0.0025,  0.0703, -0.0649,  0.0347, -0.0874,  0.0237,  0.0269,
         -0.1187, -0.0049,  0.0938, -0.0098, -0.0250,  0.1118,  0.0015,  0.1562])
        self.std = torch.tensor([0.3281, 0.5664, 0.2949, 0.2754, 0.3984, 0.3652, 0.6094, 0.2910, 0.5195,
         0.2734, 0.4219, 0.3574, 0.2314, 0.3086, 0.6133, 0.3125])
    def __iter__(self):
        for example in self.dataset:
            latent = deserialize_tensor(example['serialized_latent'])
            latent = (latent - self.mean[:,None,None,None]) / self.std[:,None,None,None]
            caption = example['caption']
            # if (latent.mean(dim =(1,2,3)).abs()+latent.std(dim=(1,2,3))).max().item() < 5:
            yield latent, caption

class OpenVidDataloader(DataLoader):
    def __init__(self, batch_size, num_workers, device, dataset):
        self.dataset = dataset
        self.device = device
        self.mean, self.std, self.channel_wise_std = -0.010, 2.08, 69
        model_name = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).cpu() # Keep on CPU to save VRAM
        self.text_embedding_dim = self.text_encoder.config.hidden_size
        
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn, prefetch_factor=16)
    
    @abstractmethod
    def collate_fn(self, batch):
        latents, caption = zip(*batch)
        latents = torch.stack(latents)
        latents = einops.rearrange(latents, 'b c t h w -> b t c h w')
        caption = list(caption)

        inputs = self.tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]

        return {"latents": latents/self.std, "captions": caption, "text_embeddings": text_embeddings}














# useful for debugging
class RandomDataset(IterableDataset):
    def __init__(self, clip_length= 16, clip_shape = (16,64,64)):
        self.clip_length = clip_length
        self.clip_shape = clip_shape
        self.mean, self.std = 0.051, 0.434
        self.channel_wise_std = 69

    def __iter__(self):
        while True:
            latents = torch.randn(self.clip_length, *self.clip_shape)
            yield latents, "random_caption"
# %%
