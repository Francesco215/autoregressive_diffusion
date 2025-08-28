# %%

import numpy as np
import os
import io
import einops

from streaming import StreamingDataset
import torch
from torch.utils.data import DataLoader, IterableDataset

from torchvision.transforms import v2 as transforms # Import the transforms module
from typing import Any


# This is used to train the VAE
class CsDataset(IterableDataset):
    def __init__(self, remote, clip_size, resolution=None, local='/tmp/streaming_dataset/counterstrike', shuffle=False, **kwargs):
        self.dataset = StreamingDataset(remote=remote, local=local, shuffle=shuffle, **kwargs)
        
        self.clip_size = clip_size
        
        # Define the resize transform once during initialization for efficiency
        self.resize_transform = None if resolution is None else transforms.Resize((resolution, resolution), antialias=True)

    def __iter__(self):
        for example in self.dataset:
            frames, actions = torch.tensor(example['frames']), torch.tensor(example['actions'])
            
            # Ensure frames are float, as transforms often expect float tensors
            frames = frames.float() / 127.5 - 1
            
            frames = einops.rearrange(frames, 't h w c -> t c h w')
            
            # --- RESIZE FRAMES HERE ---
            # Apply the resize transform to the batch of frames
            if self.resize_transform is not None:
                frames = self.resize_transform(frames)

            while(len(actions) >= self.clip_size):
                yield frames[:self.clip_size], actions[:self.clip_size]
                frames, actions = frames[self.clip_size:], actions[self.clip_size:]

    def __len__(self):
        return len(self.dataset) * (1000 // self.clip_size)


class CsCollate:

    def __call__(self, batch):
        frames, actions = zip(*batch)
        frames, actions = torch.stack(frames), torch.stack(actions)
        return frames, actions




# This is used to train the Diffusion model
# In i could do some factorization, but i think that for now it's more readable like this
class CsVaeDataset(StreamingDataset):
    def __init__(self, remote, clip_size, local, shuffle=False, **kwargs):
        super().__init__(remote=remote, local=local, shuffle=shuffle, **kwargs)
        
        self.clip_size = clip_size

    def __iter__(self) -> Any:

        for example in super().__iter__():
            means, actions = torch.tensor(example['mean']), torch.tensor(example['action'])
            means = einops.rearrange(means, 'c t h w -> t c h w')
            # logvars = einops.rearrange(logvars, 'c t h w -> t c h w')
            # actions = einops.rearrange(actions, 'c t -> t c')


            while(means.shape[0]>=self.clip_size):
                yield means[:self.clip_size], actions[:self.clip_size]
                means, actions = means[self.clip_size:], actions[self.clip_size:]
    
    def __len__(self):
        return super().__len__()*(1000//self.clip_size)



class CsVaeCollate:
    def __call__(self, batch):
        means, actions = zip(*batch)
        means, actions = torch.stack(means), torch.stack(actions)
        return means, actions