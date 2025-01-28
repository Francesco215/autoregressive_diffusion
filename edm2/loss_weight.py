import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import make_splrep
import warnings

class MultiNoiseLoss:
    def __init__(self, vertical_scaling=0, x_min=0., width=0., vertical_offset=0., min_loss=0.005, std_dev_multiplier=0.7, std_dev_shift=2):
        self.sigmas = torch.tensor([], dtype=torch.float32)
        self.losses = torch.tensor([], dtype=torch.float32)
        self.positions = torch.tensor([], dtype=torch.int64)
        self.history_size = 50_000
        self.fourier_approximator = FourierSeriesFit(-torch.pi, torch.pi, num_terms=8)

    torch.no_grad()
    def add_data(self, sigma:Tensor, loss:Tensor):
        self.sigmas = torch.cat((self.sigmas, sigma.flatten().detach().cpu()))[-self.history_size:]
        self.losses = torch.cat((self.losses, loss.flatten().detach().cpu()))[-self.history_size:]
        positions = torch.arange(sigma.shape[0] * sigma.shape[1]) % sigma.shape[1]
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
                                alpha=0.3, label='Data Points', s=.5)
        fig.colorbar(scatter, ax=ax, label='Position')

        # Generate logarithmically spaced sigma values and plot them
        num_points = 200  # Number of data points along the sigma axis
        sigma_values = torch.logspace(-2., 2., num_points)
        mean_loss = self.calculate_mean_loss(sigma_values)
        ax.plot(sigma_values, mean_loss, label='best fit', color='red')

        ax.set_xscale('log')
        ax.set_xlabel('σ (sigma)')
        ax.set_ylabel('Loss')
        ax.set_title('Simulated Loss Curves with Increasing Noise')
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

    def fourier_series(self, x):

        x=torch.log10(x)
        print(x.min(),x.max())
        # Shift and scale x to the interval [-pi, pi]
        # x = (x - self.interval_min) * (2 * torch.pi / (self.interval_max - self.interval_min)) - torch.pi
        basis = [torch.ones_like(x) * 0.5]  # a0/2 term
        for n in range(1, self.num_terms):
            basis.append(torch.cos(n * x))
            basis.append(torch.sin(n * x))
        return torch.stack(basis, dim=1)

    def fit_data(self, X, Y):
        # Ensure data is within the interval [min, max]
        mask = (torch.log10(X) >= self.interval_min) & (torch.log10(X) <= self.interval_max)
        X, Y = X[mask], Y[mask]

        # Create Fourier basis matrix
        X_basis = self.fourier_series(X)

        # Solve for coefficients using least squares
        self.coefficients = torch.linalg.lstsq(X_basis, Y.unsqueeze(1)).solution

    def predict(self, x):
        if self.coefficients is None:
            return torch.ones_like(x)
            raise ValueError("Model has not been fitted yet. Call fit_data first.")
        basis = self.fourier_series(x)
        return basis @ self.coefficients
