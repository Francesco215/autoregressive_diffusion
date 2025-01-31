#%%
import torch
import numpy as np

# Parameters
m = 1000  # Number of rows
n = 500   # Number of columns
sigma = 2.0  # Standard deviation of normal distribution

# Generate random matrix X from N(0, sigma^2)
X = torch.randn(m, n) * sigma  # Each element ~ N(0, sigma^2)

# Define vector of ones
ones_vector = torch.ones(n, 1)

# Compute Y = X * ones_vector
Y = X @ ones_vector  # Shape (m, 1)

# Compute empirical standard deviation
empirical_std = Y.std().item()

theoretical_std = np.sqrt(n) * sigma

print(f"Empirical Std: {empirical_std:.4f}")
print(f"Theoretical Std: {theoretical_std:.4f}")

# %%
Y.shape

# %%
