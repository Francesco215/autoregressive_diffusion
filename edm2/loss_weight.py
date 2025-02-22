import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class MultiNoiseLoss(nn.Module):
    def __init__(self, vertical_scaling=0, x_min=0., width=0., vertical_offset=0., 
                 min_loss=0.005, std_dev_multiplier=0.7, std_dev_shift=2):
        super().__init__()
        # Tensors to store history data.
        self.sigmas = torch.tensor([], dtype=torch.float32)
        self.losses = torch.tensor([], dtype=torch.float32)
        self.positions = torch.tensor([], dtype=torch.int64)
        self.history_size = 5000
        # Create the Fourier approximator as a submodule.
        self.fourier_approximator = FourierSeriesFit(-torch.pi, torch.pi, num_terms=4)
        
        # Save additional parameters if needed.
        self.vertical_scaling = vertical_scaling
        self.x_min = x_min
        self.width = width
        self.vertical_offset = vertical_offset
        self.min_loss = min_loss
        self.std_dev_multiplier = std_dev_multiplier
        self.std_dev_shift = std_dev_shift

    @torch.no_grad()
    def add_data(self, sigmas: Tensor, losses: Tensor):
        # Assuming sigmas and losses are 2D tensors.
        positions = torch.arange(sigmas.shape[0] * sigmas.shape[1]) % sigmas.shape[1]
        # Flatten and flip the data.
        self.sigmas = torch.cat((self.sigmas, sigmas.flatten().detach().cpu()))[-self.history_size:]
        self.losses = torch.cat((self.losses, losses.flatten().detach().cpu()))[-self.history_size:]
        self.positions = torch.cat((self.positions, positions))[-self.history_size:]

    @torch.no_grad()
    def calculate_mean_loss(self, sigma: Tensor) -> Tensor:
        return self.fourier_approximator(sigma)

    def fit_loss_curve(self, sigmas: Tensor = None, losses: Tensor = None):
        if sigmas is None: sigmas = self.sigmas
        if losses is None: losses = self.losses
        self.fourier_approximator.fit_data(sigmas, losses)

    @torch.no_grad()
    def plot(self, save_path=None):
        if self.sigmas.numel() == 0:
            return  # Nothing to plot if no data
        
        plt.close()
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot of data colored by position.
        scatter = ax.scatter(
            self.sigmas, self.losses, c=self.positions, cmap='viridis', norm=LogNorm(),
            alpha=1, label='Data Points', s=0.5
        )
        fig.colorbar(scatter, ax=ax, label='Position')
        
        # Generate sigma values (log-spaced) for plotting the fitted curve.
        num_points = 200
        sigma_values = torch.logspace(-2., 2., num_points)
        mean_loss = self.calculate_mean_loss(sigma_values)
        ax.plot(sigma_values, mean_loss, label='Best Fit', color='red')
        
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

        

class FourierSeriesFit(nn.Module):
    def __init__(self, interval_min, interval_max, num_terms=8):
        super().__init__()
        self.interval_min = interval_min
        self.interval_max = interval_max
        self.num_terms = num_terms
        self.num_basis = 2 * self.num_terms - 1  # one constant, and pairs of cos and sin terms
        
        # Initialize coefficients as learnable parameters.
        # They will be updated via our custom "fit_data" method.
        self.coefficients = nn.Parameter(torch.zeros(self.num_basis, 1), requires_grad=False)
        self.coefficients_history = []

    def fourier_series(self, x: Tensor) -> Tensor:
        """
        Create a Fourier basis evaluated at the log10 of x.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: A tensor of shape (..., num_basis) where each column is one basis function.
        """
        # Compute log10(x) (x is expected to be positive)
        x_log = torch.log10(x)
        # Start with the constant term (a0/2)
        basis = [torch.ones_like(x_log) * 0.5]
        # Append cosine and sine basis functions
        for n in range(1, self.num_terms):
            basis.append(torch.cos(n * x_log))
            basis.append(torch.sin(n * x_log))
        # Stack along the last dimension so that for each input we get a vector of length num_basis
        return torch.stack(basis, dim=-1)

    @torch.no_grad()
    def fit_data(self, X: Tensor, Y: Tensor):
        """
        Fit the Fourier coefficients to data by solving a least-squares problem.
        Only uses data where log10(X) is within the specified interval.
        
        Args:
            X (Tensor): Input tensor of x-values.
            Y (Tensor): Input tensor of y-values.
        """
        # Only consider points where log10(x) is in the desired interval.
        X_log = torch.log10(X)
        mask = (X_log >= self.interval_min) & (X_log <= self.interval_max)
        X_filtered = X[mask].flatten()
        Y_filtered = Y[mask].flatten()
        if X_filtered.numel() == 0:
            print("No data in the specified interval for fitting.")
            return
        
        # Build the Fourier basis for the filtered x values.
        basis = self.fourier_series(X_filtered)  # shape: (N, num_basis)
        # We will fit the model in log10 space:
        target = Y_filtered.log10().unsqueeze(1)  # shape: (N, 1)
        # Solve the least squares problem.
        sol = torch.linalg.lstsq(basis, target).solution  # shape: (num_basis, 1)
        # Update the coefficients in-place.
        self.coefficients.data.copy_(sol)
        # Optionally, store a history of the coefficients.
        self.coefficients_history.append(sol.detach().clone())

    def forward(self, x: Tensor) -> Tensor:
        original_shape = x.shape
        # Flatten the input so that we can process any shape.
        x_flat = x.reshape(-1)
        # Compute the Fourier basis
        basis = self.fourier_series(x_flat)  # shape: (N, num_basis)
        # Linear combination: (N, num_basis) @ (num_basis, 1) --> (N, 1)
        pred_log = basis @ self.coefficients.to(basis.device)
        # Undo the log10 transformation (i.e. compute 10**(prediction))
        pred = 10 ** pred_log
        # Reshape the predictions to the original input shape.
        return pred.reshape(original_shape)
