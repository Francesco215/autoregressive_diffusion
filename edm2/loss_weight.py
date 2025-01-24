import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import make_splrep
import warnings

class MultiNoiseLoss:
    def __init__(self, vertical_scaling=0, x_min=0., width=0., vertical_offset=0., min_loss=0.005, std_dev_multiplier=0.7, std_dev_shift=2):
        self.loss_mean_popt = [vertical_scaling, x_min, width, vertical_offset]
        self.loss_std_popt = [min_loss, std_dev_multiplier, std_dev_shift]

        self.original_loss_mean_popt = self.loss_mean_popt.copy()
        self.original_loss_std_popt = self.loss_std_popt.copy()

        self.sigmas = np.array([])
        self.losses = np.array([])
        self.positions = np.array([])
        self.history_size = 50_000
        self.spline = None

    def add_data(self, sigma:Tensor, loss:Tensor):
        self.sigmas    = np.append(self.sigmas, sigma.flatten().cpu().detach().numpy())[-self.history_size:]
        self.losses    = np.append(self.losses, loss.flatten().cpu().detach().numpy())[-self.history_size:]
        self.positions = np.append(self.positions, np.arange(sigma.shape[0] * sigma.shape[1]) % sigma.shape[1])[-self.history_size:]

    def kernel_regression(self, sigmas, losses):

        # Convert to log space and remove duplicates
        
        sorted_indices = np.argsort(sigmas)
        sigmas = sigmas[sorted_indices]
        losses = losses[sorted_indices]
        # Find unique indices and average losses for duplicates
        unique_indices = np.unique(sigmas, return_index=True)[1]
        sigmas = sigmas[unique_indices]
        losses = losses[unique_indices]
        
        # Sort and preprocess

        log_sigmas = np.log10(sigmas)
        return make_splrep(log_sigmas, losses, s=len(sigmas)*5e-3)

    def calculate_mean_loss(self, sigma:Tensor):
        if self.spline is None:
            return torch.ones_like(sigma)

        device, dtype = sigma.device, sigma.dtype
        sigma = sigma.log10().cpu().numpy()
        mean_losses = torch.tensor(self.spline(sigma), device=device, dtype=dtype)
        return mean_losses


    def fit_loss_curve(self, sigmas=None, losses=None):
        if sigmas is None: sigmas = self.sigmas
        if losses is None: losses = self.losses
        # try:
        self.spline = self.kernel_regression(sigmas, losses)
        # except Exception as e:
        #     self.spline = None
        #     warnings.warn(f"fit failed with {e}", RuntimeWarning)

    def plot(self, save_path=None):
        # --- Simulation Parameters ---
        num_points = 200  # Number of data points along the sigma axis

        # Generate logarithmically spaced sigma values
        sigma_values = np.logspace(-2., 2., num_points)

        # --- Plotting ---
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 6))
        # loss_means = self.calculate_mean_loss(sigma_values)
        # loss_stds = self.calculate_std_loss(sigma_values)

        # Scatter plot of the data, colored by position using the viridis colormap
        if self.sigmas.size > 0:  # Only plot collected data if there is any
            scatter = ax.scatter(self.sigmas, self.losses, c=self.positions, cmap='viridis', norm=LogNorm(),
                                 alpha=0.3, label='Data Points', s=.5)
            fig.colorbar(scatter, ax=ax, label='Position')

        plt.plot(sigma_values, self.calculate_mean_loss(torch.tensor(sigma_values)), label='Best Fit', color='red')
        # ax.plot(sigma_values, loss_means, label='best fit', color='red')
        # ax.fill_between(sigma_values, loss_means - loss_stds, loss_means + loss_stds, color='red', alpha=0.2)

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

        