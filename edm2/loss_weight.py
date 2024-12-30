import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

class MultiNoiseLoss:
    def __init__(self, vertical_scaling = -0.9, x_min = 0.5, width = 1, vertical_offset = 1, min_loss = 0.005, std_dev_multiplier = 0.7, std_dev_shift = 2):
        self.loss_mean_popt = [vertical_scaling, x_min, width, vertical_offset]
        self.loss_std_popt =  [min_loss, std_dev_multiplier, std_dev_shift]

        self.sigmas = np.array([])
        self.losses = np.array([])
        self.positions = np.array([])
        self.history_size = 50_000

    def add_data(self, sigma, loss):
        self.sigmas=np.append(self.sigmas,sigma.flatten().cpu().detach().numpy())[-self.history_size:]
        self.losses=np.append(self.losses,loss.flatten().cpu().detach().numpy())[-self.history_size:]
        self.positions=np.append(self.positions,np.arange(sigma.shape[0]*sigma.shape[1])%sigma.shape[1])[-self.history_size:]

    def calculate_mean_loss_fn(self, sigma, vertical_scaling, x_min, width, vertical_offset):
        """
        Calculates the loss based on a parametric function.

        Args:
            sigma: Independent variable (noise level).
            vertical_scaling: Controls the vertical scaling of the dip.
            x_min: The log(sigma) value at which the minimum loss occurs.
            width: Affects the width of the curve around the minimum.
            vertical_offset: Vertical offset of the curve.

        Returns:
            The calculated loss value.
        """
        return vertical_scaling * np.exp(-((np.log(sigma) - x_min)**2) / (2 * width**2)) + vertical_offset

    def calculate_mean_loss(self,sigma):
        return self.calculate_mean_loss_fn(sigma, *self.loss_mean_popt)

    def calculate_std_loss_fn(self, sigma, min_loss, std_dev_multiplier, std_dev_shift):
        """
        Calculates the standard deviation of the loss, which increases with sigma.

        Args:
            sigma: Independent variable (noise level).
            min_loss: The minimum achievable loss (used to scale the standard deviation).
            std_dev_multiplier: Controls the overall magnitude of the standard deviation.
            std_dev_shift: Shifts the sigma value at which the standard deviation starts to increase rapidly.
            vertical_scaling: Vertical scaling parameter of the loss function.
            x_min: x_min parameter of the loss function.
            width: Width parameter of the loss function.
            vertical_offset: Vertical offset parameter of the loss function.

        Returns:
            The calculated standard deviation.
        """

        loss = self.calculate_mean_loss(sigma)
        return (loss - min_loss) * std_dev_multiplier * sigma / (std_dev_shift + sigma)
    
    def calculate_std_loss(self, sigma):
        return self.calculate_std_loss_fn(sigma, *self.loss_std_popt)

    def find_mean_loss_popt(self, sigma_values, loss_values):
        popt, _ = curve_fit(self.calculate_mean_loss_fn, sigma_values, loss_values, p0=self.loss_mean_popt, sigma=self.calculate_std_loss(sigma_values), absolute_sigma=True)
        self.mean_loss_popt = popt
        return popt

    def find_std_loss_popt(self, sigma_values, loss_values):
        loss_residuals = np.abs(loss_values - self.calculate_mean_loss(sigma_values))
        popt, _ = curve_fit(self.calculate_std_loss_fn, sigma_values, loss_residuals, p0=self.loss_std_popt, sigma=self.calculate_std_loss(sigma_values), absolute_sigma=True)
        self.loss_std_popt = popt
        return popt

    def fit_loss_curve(self, sigma_values, loss_values):
        self.loss_mean_popt = self.find_mean_loss_popt(sigma_values, loss_values)
        self.loss_std_popt = self.find_std_loss_popt(sigma_values, loss_values)


    def plot(self, save_path=None):
        # --- Simulation Parameters ---
        num_points = 200  # Number of data points along the sigma axis

        # Generate logarithmically spaced sigma values
        sigma_values = np.logspace(-2.3, 2.7, num_points)

        # --- Plotting ---
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 6))
        loss_means = self.calculate_mean_loss(sigma_values)
        loss_stds = self.calculate_std_loss(sigma_values)


        # Scatter plot of the data, colored by position using the viridis colormap
        if self.sigmas.size > 0: # Only plot collected data if there is any
            scatter = ax.scatter(self.sigmas, self.losses, c=self.positions, cmap='viridis', norm=LogNorm(), alpha=0.7, label='Data Points', s=.5)
            fig.colorbar(scatter, ax=ax, label='Position')

        ax.plot(sigma_values, loss_means, label='CIFAR-10', color='red')
        ax.fill_between(sigma_values, loss_means - loss_stds, loss_means + loss_stds, color='red', alpha=0.2)

        ax.set_xscale('log')
        ax.set_xlabel('σ (sigma)')
        ax.set_ylabel('Loss')
        ax.set_title('Simulated Loss Curves with Increasing Noise')
        ax.legend()
        ax.grid(True)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()