#%%
from edm2.cs_dataloading import CsVaeCollate, CsVaeDataset
from torch.utils.data import DataLoader
from streaming.base.util import clean_stale_shared_memory

clean_stale_shared_memory()

dataset = CsVaeDataset(clip_size=12, remote='s3://counter-strike-data/dataset_compressed_stable_diff', local = f'/data/streaming_dataset/cs_vae_comp', batch_size=6, shuffle=False, cache_limit = '50gb')
# dataset = CsVaeDataset(clip_size=12, remote='s3://counter-strike-data/dataset_compressed', local = f'/data/streaming_dataset/cs_vae', batch_size=1, shuffle=False, cache_limit = '50gb')
# %%
print(dataset[0])
dataloader = DataLoader(dataset, batch_size=6, collate_fn=CsVaeCollate(), pin_memory=True, num_workers=0, shuffle=False)
# %%
next(iter(dataloader))
# %%
