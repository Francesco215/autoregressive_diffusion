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
from typing import Optional



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
        return len(self.dataset)



class CsCollate:
    def __init__(self, state_size):
        self.state_size = state_size

    def __call__(self, batch):
        frames, actions = zip(*batch)
        frames, actions = torch.stack(frames), torch.stack(actions)
        return frames, actions