#%%
import torch
import matplotlib.pyplot as plt

class FourierSeriesFit:
    def __init__(self, interval_min, interval_max, num_terms=8):

        self.interval_min = interval_min
        self.interval_max = interval_max
        self.L = (interval_max - interval_min) / 2  # Half-interval length
        self.num_terms = num_terms
        self.coefficients = None

    def fourier_series(self, x):

        # Shift and scale x to the interval [-L, L]
        x_scaled = (x - self.interval_min) * (2 * torch.pi / (self.interval_max - self.interval_min)) - torch.pi

        basis = [torch.ones_like(x) * 0.5]  # a0/2 term
        for n in range(1, self.num_terms):
            basis.append(torch.cos(n * x_scaled))
            basis.append(torch.sin(n * x_scaled))
        return torch.stack(basis, dim=1)

    def fit_data(self, X, Y):
        # Ensure data is within the interval [min, max]
        mask = (X >= self.interval_min) & (X <= self.interval_max)
        X, Y = X[mask], Y[mask]

        # Create Fourier basis matrix
        X_basis = self.fourier_series(X)

        # Solve for coefficients using least squares
        self.coefficients = torch.linalg.lstsq(X_basis, Y.unsqueeze(1)).solution

    def predict(self, x):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call fit_data first.")
        basis = self.fourier_series(x)
        return basis @ self.coefficients

    def plot(self, X, Y, X_grid=None, num_points=300):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call fit_data first.")

        if X_grid is None:
            X_grid = torch.linspace(self.interval_min, self.interval_max, num_points)

        Y_pred = self.predict(X_grid)

        plt.figure(figsize=(10, 6))
        plt.scatter(X.numpy(), Y.numpy(), alpha=0.3, label='Data')
        plt.plot(X_grid.numpy(), Y_pred.squeeze().numpy(), 
                 'r', linewidth=2, label=f'Fourier Fit ({self.num_terms} terms)')
        plt.xlabel('x'), plt.ylabel('y'), plt.legend()
        plt.title("Fourier Series Fit")
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Define the function to fit
    def l(x):
        return (1 - torch.exp(-x**2)) - x * 0.1

    # Generate data
    x = torch.randn(1000)
    y = l(x) + torch.randn_like(x) * 0.1

    # Define the interval [min, max]
    interval_min = -3.0
    interval_max = 3.0

    # Initialize and fit the Fourier series
    fourier_fitter = FourierSeriesFit(interval_min=interval_min, interval_max=interval_max, num_terms=8)
    fourier_fitter.fit_data(x, y)

    # Plot the results
    fourier_fitter.plot(x, y)
# %%
