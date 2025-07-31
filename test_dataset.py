#%%
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



from edm2.cs_dataloading import CsVaeCollate, CsVaeDataset
from edm2.vae import VAE
import torch._dynamo.config

torch._dynamo.config.cache_size_limit = 100
        

        
micro_batch_size = 1
batch_size = 16
accumulation_steps = batch_size//micro_batch_size
clip_length = 16
dataset = CsVaeDataset(clip_size=clip_length, remote='s3://counter-strike-data/dataset_compressed_stable_diff', local = f'/data/streaming_dataset/cs_vae_sd', batch_size=micro_batch_size, shuffle=False, cache_limit = '50gb')
dataloader = DataLoader(dataset, batch_size=micro_batch_size, collate_fn=CsVaeCollate(), pin_memory=True, num_workers=4, shuffle=False)
# %%
m = torch.zeros(4)
s = torch.zeros(4)
num_iterations = 1000
for i, batch in enumerate (tqdm(dataloader)):
    means, logvars, _ = batch
    m+=means.mean(dim=(0,1,3,4))
    latents = means + torch.exp(logvars*0.5)
    s+=latents.std(dim=(0,1,3,4))
    if i ==num_iterations: break

print(m/num_iterations, s/num_iterations)


# %%
