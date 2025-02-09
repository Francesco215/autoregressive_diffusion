import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import make_splrep
import einops
import warnings

class MultiNoiseLoss:
    def __init__(self, vertical_scaling=0, x_min=0., width=0., vertical_offset=0., min_loss=0.005, std_dev_multiplier=0.7, std_dev_shift=2):
        self.sigmas = torch.tensor([], dtype=torch.float32)
        self.losses = torch.tensor([], dtype=torch.float32)
        self.positions = torch.tensor([], dtype=torch.int64)
        self.history_size = 5_000
        self.fourier_approximator = FourierSeriesFit(-torch.pi, torch.pi, num_terms=4)

    torch.no_grad()
    def add_data(self, sigmas:Tensor, losses:Tensor):
        positions = torch.arange(sigmas.shape[0] * sigmas.shape[1]) % sigmas.shape[1]
        sigmas, losses, positions = sigmas.flatten().flip((0,)), losses.flatten().flip((0,)), positions.flip((0,))
        
        self.sigmas = torch.cat((self.sigmas, sigmas.flatten().detach().cpu()))[-self.history_size:]
        self.losses = torch.cat((self.losses, losses.flatten().detach().cpu()))[-self.history_size:]
        self.positions = torch.cat((self.positions, positions))[-self.history_size:]

    @torch.no_grad()
    def calculate_mean_loss(self, sigma:Tensor):
        mean_loss = self.fourier_approximator.predict(sigma)
        return mean_loss


    def fit_loss_curve(self, sigmas=None, losses=None):
        if sigmas is None: sigmas = self.sigmas
        if losses is None: losses = self.losses
        # try:
        self.fourier_approximator.fit_data(sigmas, losses)

    @torch.no_grad()
    def plot(self, save_path=None):
        if self.sigmas.size == 0:  # Only plot collected data if there is any
            return 

        # --- Plotting ---
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot of the data, colored by position using the viridis colormap
        scatter = ax.scatter(self.sigmas, self.losses, c=self.positions, cmap='viridis', norm=LogNorm(),
                                alpha=1, label='Data Points', s=.5)
        fig.colorbar(scatter, ax=ax, label='Position')

        # Generate logarithmically spaced sigma values and plot them
        num_points = 200  # Number of data points along the sigma axis
        sigma_values = torch.logspace(-2., 2., num_points)
        mean_loss = self.calculate_mean_loss(sigma_values)
        ax.plot(sigma_values, mean_loss, label='best fit', color='red')

        ax.set_xscale('log')
        ax.set_xlabel('σ (sigma)')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.set_title('Loss as a function of the noise σ')
        ax.legend()
        ax.grid(True)
        if save_path is not None:
            plt.savefig(save_path, dpi=600)
        plt.show()
        plt.close()

class FourierSeriesFit:
    def __init__(self, interval_min, interval_max, num_terms=8):

        self.interval_min = interval_min
        self.interval_max = interval_max
        self.L = (interval_max - interval_min) / 2  # Half-interval length
        self.num_terms = num_terms
        self.coefficients = None
        self.coefficients_history = []

    def fourier_series(self, x):

        x=torch.log10(x)
        # Shift and scale x to the interval [-pi, pi]
        # x = (x - self.interval_min) * (2 * torch.pi / (self.interval_max - self.interval_min)) - torch.pi
        basis = [torch.ones_like(x) * 0.5]  # a0/2 term
        for n in range(1, self.num_terms):
            basis.append(torch.cos(n * x))
            basis.append(torch.sin(n * x))
        return torch.stack(basis, dim=1)

    def fit_data(self, X:Tensor, Y:Tensor):
        # Ensure data is within the interval [min, max]
        mask = (torch.log10(X) >= self.interval_min) & (torch.log10(X) <= self.interval_max)
        X, Y = X[mask].flatten(), Y[mask].flatten()

        # Create Fourier basis matrix
        X_basis = self.fourier_series(X)

        # Solve for coefficients using least squares
        self.coefficients = torch.linalg.lstsq(X_basis, Y.log10().unsqueeze(1)).solution
        self.coefficients_history.append(self.coefficients)

    def predict(self, x:Tensor):
        if self.coefficients is None:
            return torch.ones_like(x)
            raise ValueError("Model has not been fitted yet. Call fit_data first.")
        if x.dim() == 2:
            b = x.shape[0]
            x = einops.rearrange(x, 'b t -> (b t)')
            return einops.rearrange(self.predict(x).squeeze(1), '(b t) -> b t', b=b)
        basis = self.fourier_series(x)
        return 10.**(basis @ self.coefficients.to(basis.device))
