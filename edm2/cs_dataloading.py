# %%

import numpy as np
import os
import io
import einops

from streaming import StreamingDataset
import torch
from torch.utils.data import DataLoader, IterableDataset

from abc import abstractmethod
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Any


# This is used to train the VAE
class CsDataset(IterableDataset):
    def __init__(self, remote, clip_size, local='/tmp/streaming_dataset/counterstrike', shuffle=False, **kwargs):
        self.dataset = StreamingDataset(remote=remote, local=local, shuffle=shuffle, **kwargs)
        
        self.clip_size = clip_size

    def __iter__(self):
        for example in self.dataset:
            frames, actions = torch.tensor(example['frames']), torch.tensor(example['actions'])

            while(len(actions)>=self.clip_size):
                yield frames[:self.clip_size], actions[:self.clip_size]
                frames, actions = frames[self.clip_size:], actions[self.clip_size:]

    def __len__(self):
        return len(self.dataset)*(1000//self.clip_size)

class CsDataset(IterableDataset):
    def __init__(self, remote, clip_size, local='/tmp/streaming_dataset/counterstrike', shuffle=False, **kwargs):
        self.dataset = StreamingDataset(remote=remote, local=local, shuffle=shuffle, **kwargs)
        self.clip_size = clip_size
        
        # Safety check for the first shard
        samples_per_shard = getattr(self.dataset, 'samples_per_shard', None)
        if samples_per_shard is not None and len(samples_per_shard) > 0 and samples_per_shard[0] == 0:
            # Create a new version with the problematic first shard filtered out
            # This is a monkey patch to avoid the division by zero
            samples_per_shard = np.array([1 if s == 0 else s for s in samples_per_shard])
            setattr(self.dataset, 'samples_per_shard', samples_per_shard)

    def __iter__(self):
        sample_count = 0
        clip_count = 0
        
        try:
            for example in self.dataset:
                sample_count += 1
                
                try:
                    # Check if the example has the required keys
                    if 'frames' not in example or 'actions' not in example:
                        continue
                    
                    # Convert to tensors and check shapes
                    frames = torch.tensor(example['frames']) 
                    actions = torch.tensor(example['actions'])
                    
                    # Skip examples that are too small
                    if len(actions) < self.clip_size:
                        continue
                    
                    # Process the example into clips
                    while len(actions) >= self.clip_size:
                        clip_count += 1
                        yield frames[:self.clip_size], actions[:self.clip_size]
                        frames = frames[self.clip_size:]
                        actions = actions[self.clip_size:]
                        
                except Exception:
                    continue
                
        except Exception:
            pass

    def __len__(self):
        # Safer length estimate that can't be zero
        return max(1, len(self.dataset) * (1000 // self.clip_size))

class CsCollate:
    def __init__(self, clip_length):
        self.clip_length = clip_length

    def __call__(self, batch):
        frames, actions = zip(*batch)
        frames, actions = torch.stack(frames), torch.stack(actions)
        return frames, actions




# This is used to train the Diffusion model
# In i could do some factorization, but i think that for now it's more readable like this
class CsVaeDataset(StreamingDataset):
    def __init__(self, remote, clip_size, local='/tmp/streaming_dataset/dataset_compressed', shuffle=False, **kwargs):
        super().__init__(remote=remote, local=local, shuffle=shuffle, **kwargs)
        
        self.clip_size = clip_size

    def __iter__(self) -> Any:

        for example in super().__iter__():
            means, logvars, actions = torch.tensor(example['mean']), torch.tensor(example['logvar']), torch.tensor(example['action'])
            means = einops.rearrange(means, 'c t h w -> t c h w')
            logvars = einops.rearrange(logvars, 'c t h w -> t c h w')
            actions = einops.rearrange(actions, 'c t -> t c')


            while(actions.shape[0]>=self.clip_size):
                yield means[:self.clip_size], logvars[:self.clip_size], actions[:self.clip_size]
                means, logvars, actions = means[self.clip_size:], logvars[self.clip_size:], actions[self.clip_size:]
    
    def __len__(self):
        return super().__len__()*(250//self.clip_size)



class CsVaeCollate:
    def __call__(self, batch):
        means, logvars, actions = zip(*batch)
        means, logvars, actions = torch.stack(means), torch.stack(logvars), torch.stack(actions)
        return means, logvars, actions