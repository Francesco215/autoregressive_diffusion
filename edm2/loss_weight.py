import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
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

    def add_data(self, sigma, loss):
        self.sigmas = np.append(self.sigmas, sigma.flatten().cpu().detach().numpy())[-self.history_size:]
        self.losses = np.append(self.losses, loss.flatten().cpu().detach().numpy())[-self.history_size:]
        self.positions = np.append(self.positions, np.arange(sigma.shape[0] * sigma.shape[1]) % sigma.shape[1])[
                       -self.history_size:]

    def calculate_mean_loss_fn(self, sigma, vertical_scaling, x_min, width, vertical_offset):
        """
        Calculates the loss based on a parametric function combining Gaussian and Logistic.

        Args:
            sigma: Independent variable (noise level).
            vertical_scaling: Controls the vertical scaling of the Gaussian dip.
            x_min: The log(sigma) value at which the minimum loss occurs.
            width: Affects the width of the curve around the minimum.
            vertical_offset: Vertical offset of the entire curve.
            logistic_k: Steepness of the logistic curve.
            logistic_x0: Midpoint of the logistic curve.
            logistic_L: Maximum value of the logistic curve.

        Returns:
            The calculated loss value.
        """
        # Convert to numpy arrays for scipy curve_fit compatibility
        if isinstance(sigma, torch.Tensor):
            sigma_np = sigma.detach().cpu().numpy()
        else:
            sigma_np = sigma
            
        gaussian_part = -vertical_scaling * np.exp(-((np.log(sigma_np) - x_min) ** 2) / (2 * width ** 2))
        # logistic_part = logistic_L / (1 + np.exp(-logistic_k * (np.log(sigma_np) - logistic_x0)))
        result = gaussian_part + vertical_offset
        
        # Convert back to tensor if input was a tensor
        if isinstance(sigma, torch.Tensor):
            result = torch.tensor(result, dtype=sigma.dtype, device=sigma.device)
        
        return result

    def calculate_mean_loss(self, sigma):
        return self.calculate_mean_loss_fn(sigma, *self.loss_mean_popt)

    def calculate_std_loss_fn(self, sigma, min_loss, std_dev_multiplier, std_dev_shift):
        """
        Calculates the standard deviation of the loss, which increases with sigma.

        Args:
            sigma: Independent variable (noise level).
            min_loss: The minimum achievable loss (used to scale the standard deviation).
            std_dev_multiplier: Controls the overall magnitude of the standard deviation.
            std_dev_shift: Shifts the sigma value at which the standard deviation starts to increase rapidly.

        Returns:
            The calculated standard deviation.
        """
        # Convert to numpy arrays for scipy curve_fit compatibility
        if isinstance(sigma, torch.Tensor):
            sigma_np = sigma.detach().cpu().numpy()
        else:
            sigma_np = sigma
        
        loss = self.calculate_mean_loss(sigma_np)
        result = (loss - min_loss) * std_dev_multiplier * sigma_np / (std_dev_shift + sigma_np)
        
        # Convert back to tensor if input was a tensor
        if isinstance(sigma, torch.Tensor):
            result = torch.tensor(result, dtype=sigma.dtype, device=sigma.device)
        
        return result

    def calculate_std_loss(self, sigma):
        return self.calculate_std_loss_fn(sigma, *self.loss_std_popt)

    def find_mean_loss_popt(self, sigma_values, loss_values):
      
        sigma_values_np = sigma_values.cpu().numpy()
        loss_values_np = loss_values.cpu().numpy()
        std_loss_np = self.calculate_std_loss(sigma_values_np)

        popt, _ = curve_fit(self.calculate_mean_loss_fn, sigma_values_np, loss_values_np, p0=self.loss_mean_popt,
                              maxfev=10000)
        return popt

    def find_std_loss_popt(self, sigma_values, loss_values):
        sigma_values_np = sigma_values.cpu().numpy()
        loss_values_np = loss_values.cpu().numpy()

        loss_residuals = np.abs(loss_values_np - self.calculate_mean_loss(sigma_values_np))
        std_loss_np = self.calculate_std_loss(sigma_values_np)
        
        popt, _ = curve_fit(self.calculate_std_loss_fn, sigma_values_np, loss_residuals, p0=self.loss_std_popt,
                            maxfev=10000)
        return popt

    def fit_loss_curve(self, sigma_values=None, loss_values=None):
        if sigma_values is None: sigma_values = torch.tensor(self.sigmas)
        if loss_values is None: loss_values = torch.tensor(self.losses)
        try:
            self.loss_mean_popt = self.find_mean_loss_popt(sigma_values, loss_values)
            self.loss_std_popt = self.find_std_loss_popt(sigma_values, loss_values)
        except :
            warnings.warn("fit failed", RuntimeWarning)
            self.loss_mean_popt = self.original_loss_mean_popt
            self.loss_std_popt = self.original_loss_std_popt

    def plot(self, save_path=None):
        # --- Simulation Parameters ---
        num_points = 200  # Number of data points along the sigma axis

        # Generate logarithmically spaced sigma values
        sigma_values = np.logspace(-2., 2., num_points)

        # --- Plotting ---
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 6))
        loss_means = self.calculate_mean_loss(sigma_values)
        loss_stds = self.calculate_std_loss(sigma_values)

        # Scatter plot of the data, colored by position using the viridis colormap
        if self.sigmas.size > 0:  # Only plot collected data if there is any
            scatter = ax.scatter(self.sigmas, self.losses, c=self.positions, cmap='viridis', norm=LogNorm(),
                                 alpha=0.3, label='Data Points', s=.5)
            fig.colorbar(scatter, ax=ax, label='Position')

        ax.plot(sigma_values, loss_means, label='best fit', color='red')
        ax.fill_between(sigma_values, loss_means - loss_stds, loss_means + loss_stds, color='red', alpha=0.2)

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