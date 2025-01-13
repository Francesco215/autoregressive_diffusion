# %%

import os
import io

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
        self.dataset = load_dataset("fal/cosmos-openvid-1m", data_dir="continuous", split="train", streaming=True)
        self.device = "cuda"

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
        
        super().__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
    
    @abstractmethod
    def collate_fn(self, batch):
        latents, caption = zip(*batch)
        latents = torch.stack(latents).to(self.device)
        caption = list(caption)
        return {"latents": latents, "captions": caption}
