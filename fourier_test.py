#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

def l(x):
    return (1-torch.exp(-x**2))-x*0.1

def fourier_function(x, c_cos, c_sin, freqs):
    cos = torch.cos(freqs*x)
    sin = torch.sin(freqs*x)
    return (c_cos@cos + c_sin@sin)
    
x = torch.randn(300)
interval_size = 8
x = x[(x > -interval_size) & (x < interval_size)]
y = l(x)+ torch.randn_like(x)*0.1

fourier_frequencies = 2

fourier_coefficient = 2*np.pi/(interval_size/2)
freqs = torch.arange(0, fourier_frequencies, device=x.device)
freqs = freqs.unsqueeze(1)*fourier_coefficient

s = x.repeat(fourier_frequencies, 1)
cos = torch.cos(freqs*s)
sin = torch.sin(freqs*s)


c_cos = (y * cos).mean(dim=-1)
c_sin = (y * sin).mean(dim=-1)

X = torch.linspace(-interval_size/2, interval_size/2, 100, device=x.device)
Y = fourier_function(X, c_cos, c_sin, freqs)
plt.plot(x,y,'.')
plt.plot(X,Y)
# %%
import torch
import matplotlib.pyplot as plt

def l(x):
    return (1-torch.exp(-x**2)) - x*0.1

# Corrected Fourier series implementation
def fourier_series(x, L, num_terms):
    """
    x: input tensor
    L: half-interval length (full interval [-L, L])
    num_terms: number of Fourier terms (n=0 to num_terms-1)
    """
    # Basis functions
    basis = [torch.ones_like(x) * 0.5]  # a0/2 term
    for n in range(1, num_terms):
        basis.append(torch.cos(n * torch.pi * x / L))
        basis.append(torch.sin(n * torch.pi * x / L))
    return torch.stack(basis, dim=1)

# Parameters
interval_size = 8  # Total interval [-4, 4] would be size=8
L = interval_size / 2  # Half-interval length = 4
num_terms = 5  # Includes DC term + 4 pairs of sin/cos

# Generate data (keep within [-L, L])
x = torch.randn(1000)
x = x[(x > -L) & (x < L)]
y = l(x) + torch.randn_like(x)*0.1

# Create Fourier basis matrix
X_basis = fourier_series(x, L, num_terms)

# Solve for coefficients using least squares
coefficients = torch.linalg.lstsq(X_basis, y.unsqueeze(1)).solution

# Reconstruction function
def fourier_recon(x, coeffs, L, num_terms):
    basis = fourier_series(x, L, num_terms)
    return basis @ coeffs

# Generate prediction points
X_grid = torch.linspace(-L, L, 300)
Y_pred = fourier_recon(X_grid, coefficients, L, num_terms)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x.numpy(), y.numpy(), alpha=0.3, label='Data')
plt.plot(X_grid.numpy(), Y_pred.squeeze().numpy(), 
         'r', linewidth=2, label=f'Fourier Fit ({num_terms} terms)')
plt.xlabel('x'), plt.ylabel('y'), plt.legend()
plt.title("Corrected Fourier Series Fit")
plt.grid(True)
plt.show()
# %%
