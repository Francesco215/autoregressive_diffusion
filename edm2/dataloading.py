# %%

import os
import io
import einops

import torch
from torch.utils.data import DataLoader, IterableDataset

from abc import abstractmethod
from tqdm import tqdm
from datasets import load_dataset
from typing import Optional

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
#%%
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

        self.mean, self.std = 0.051, 0.434
    def __iter__(self):
        for example in self.dataset:
            latent = (deserialize_tensor(example['serialized_latent'])-self.mean)/self.std
            caption = example['caption']
            yield latent, caption

class OpenVidDataloader(DataLoader):
    def __init__(self, batch_size, num_workers, device):
        self.dataset = OpenVidDataset()
        self.device = device
        self.mean, self.std, self.channel_wise_std = -0.010, 2.08, 69
        
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn, prefetch_factor=16)
    
    @abstractmethod
    def collate_fn(self, batch):
        latents, caption = zip(*batch)
        latents = torch.stack(latents)
        latents = einops.rearrange(latents, 'b c t h w -> b t c h w')
        caption = list(caption)
        return {"latents": latents/self.std, "captions": caption}














# useful for debugging
class RandomDataset(IterableDataset):
    def __init__(self, clip_length, clip_shape):
        self.clip_length = clip_length
        self.clip_shape = clip_shape
        self.mean, self.std = 0.051, 0.434
        self.channel_wise_std = 69

    def __iter__(self):
        while True:
            latents = torch.randn(self.clip_length, *self.clip_shape)
            yield latents, "random_caption"
class RandomDataloader(DataLoader):
    def __init__(self, batch_size, num_workers, device):
        self.dataset = RandomDataset(16, (16, 64, 64))
        self.device = device
        self.mean, self.std, self.channel_wise_std = 0, 2., 1.
        
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn, prefetch_factor=4)
    
    @abstractmethod
    def collate_fn(self, batch):
        latents, caption = zip(*batch)
        latents = torch.stack(latents)
        latents = einops.rearrange(latents, 'b c t h w -> b t c h w')
        caption = list(caption)
        return {"latents": latents/self.std, "captions": caption}
