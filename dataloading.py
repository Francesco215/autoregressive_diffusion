#%%
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.multiprocessing as mp

class EncodedVideoDataset(Dataset):
    """
    Dataset for loading encoded video latents.
    """
    def __init__(self, latent_dir):
        """
        Args:
            latent_dir: Directory where the encoded latents are stored.
        """
        self.latent_dir = latent_dir
        self.latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]
        # Sort the files to maintain a consistent order (important if you're not shuffling)
        self.latent_files.sort()

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        """
        Loads and returns a latent from the dataset.

        Args:
            idx: Index of the latent to load.

        Returns:
            latent: The loaded latent tensor.
        """
        latent_path = os.path.join(self.latent_dir, self.latent_files[idx])
        latent = torch.load(latent_path, map_location='cpu')
        #latent has shape (1, channels, time, height, width)
        latent = latent.squeeze(0)
        return latent

# Example usage:
latent_dir = "encoded_latents"  # Path to the directory where you saved the encoded latents
batch_size = 4  # Adjust as needed
num_workers = 4 # Adjust as needed (number of processes to use for data loading)

# Create dataset instance
dataset = EncodedVideoDataset(latent_dir)
#%%
# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle the data at each epoch
    num_workers=num_workers,  # Use multiple processes for data loading
    pin_memory=True, #  can speed up data transfer to the GPU
    drop_last=False # Might want to drop the last incomplete batch if it causes issues
)
#%%
num_epochs = 5  # Adjust as needed
# Example of iterating through the dataloader during training:
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Move batch to the appropriate device (e.g., GPU)
        batch = batch.to("cuda")

        # Perform your training operations with the batch here
        # ... your training code ...

        print(f"Epoch: {epoch+1}, Batch: {i+1}, Batch shape: {batch.shape}")
        break
    break
# %%
