#%%
import io
import torch
from typing import Optional
from datasets import load_dataset
# %%
ds = load_dataset("fal/cosmos-openvid-1m", split="train", streaming=True)
# %%
for i, example in enumerate(ds):
    print(example)
    if i==10:
        break

# %%
def deserialize_tensor(
    serialized_tensor: bytes, device: Optional[str] = None
) -> torch.Tensor:
    return torch.load(
        io.BytesIO(serialized_tensor),
        weights_only=True,
        map_location=torch.device(device) if device else None,
    )

asd=deserialize_tensor(example['serialized_latent'])
# %%
asd.shape

# %%
