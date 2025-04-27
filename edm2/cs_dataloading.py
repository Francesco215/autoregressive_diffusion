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
        
        # Print diagnostic info about the dataset
        print(f"Dataset initialized with {len(self.dataset)} samples across {len(getattr(self.dataset, 'shards', []))} shards")
        
        # Add a safety check for the first shard
        samples_per_shard = getattr(self.dataset, 'samples_per_shard', None)
        if samples_per_shard is not None and len(samples_per_shard) > 0 and samples_per_shard[0] == 0:
            print("WARNING: First shard has 0 samples, this may cause division by zero errors")
            # Create a new version with the problematic first shard filtered out
            # This is a monkey patch to avoid the division by zero
            samples_per_shard = np.array([1 if s == 0 else s for s in samples_per_shard])
            setattr(self.dataset, 'samples_per_shard', samples_per_shard)
            print("Fixed samples_per_shard to avoid division by zero errors")

    def __iter__(self):
        sample_count = 0
        clip_count = 0
        
        try:
            for example in self.dataset:
                sample_count += 1
                if sample_count == 1:
                    print(f"First example keys: {list(example.keys())}")
                
                try:
                    # Check if the example has the required keys
                    if 'frames' not in example or 'actions' not in example:
                        print(f"Example missing required keys, found: {list(example.keys())}")
                        continue
                    
                    # Convert to tensors and check shapes
                    frames = torch.tensor(example['frames']) 
                    actions = torch.tensor(example['actions'])
                    
                    if sample_count == 1:
                        print(f"First example shapes: frames={frames.shape}, actions={actions.shape}")
                    
                    # Skip examples that are too small
                    if len(actions) < self.clip_size:
                        print(f"Example too small: {len(actions)} actions, needed {self.clip_size}")
                        continue
                    
                    # Process the example into clips
                    extracted_clips = 0
                    while len(actions) >= self.clip_size:
                        clip_count += 1
                        extracted_clips += 1
                        yield frames[:self.clip_size], actions[:self.clip_size]
                        frames = frames[self.clip_size:]
                        actions = actions[self.clip_size:]
                    
                    if sample_count % 10 == 0:
                        print(f"Processed {sample_count} examples, extracted {clip_count} clips")
                        
                except Exception as e:
                    print(f"Error processing example {sample_count}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"Error in dataset iteration: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Dataset iteration complete: processed {sample_count} examples, yielded {clip_count} clips")

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