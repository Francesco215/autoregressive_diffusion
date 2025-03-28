# %%

import numpy as np
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

class CsCollate:
    def __init__(self, state_size):
        self.state_size = state_size

    def __call__(self, batch):
        actions = np.stack([item['actions'] for item in batch])
        frames = np.stack([item['frames'] for item in batch])
        cut = (actions.shape[1]//self.state_size) * self.state_size
        frames, actions = torch.tensor(frames[:,:cut]), torch.tensor(actions[:,:cut])
        
        frames = einops.rearrange(frames,'b (ts t) h w c ->  (b ts) t h w c', t=self.state_size)
        actions = einops.rearrange(actions, 'b (ts t) c -> (b ts) t c', t=self.state_size)
        return frames, actions