import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
import numpy as np

class PoissonSolver(nn.Module):
    def __init__(self, model: nn.Module,
                 n_interior: int = 10000,
                 n_boundary: int = 800,
                 alpha: float = 0.01):
        super().__init__()
        self.model = model
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.alpha = alpha

    def sample_points(self) -> Tuple[th.Tensor, th.Tensor]:
        """Sample interior and boundary points"""
        # Interior points
        x_interior = th.rand((self.n_interior, 2)) * 2 - 1  # [-1, 1]^2

        # Boundary points (uniformly on the boundary)
        theta = th.linspace(0, 2 * np.pi, self.n_boundary)
        x_boundary = th.stack([th.cos(theta), th.sin(theta)], dim=1)

        return x_interior, x_boundary

    def true_solution(self, x: th.Tensor) -> th.Tensor:
        """Ground truth solution u = sin(πx)sin(πy²)"""
        return th.sin(np.pi * x[:, 0]) * th.sin(np.pi * x[:, 1] ** 2)

    def forcing(self, x: th.Tensor) -> th.Tensor:
        """Right hand side f = -Δu"""
        return -np.pi ** 2 * (1 + 4 * x[:, 1] ** 2) * th.sin(np.pi * x[:, 0]) * \
            th.sin(np.pi * x[:, 1] ** 2) + \
            2 * np.pi * th.sin(np.pi * x[:, 0]) * th.cos(np.pi * x[:, 1] ** 2)

    def compute_gradients(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Compute first and second derivatives"""
        x.requires_grad_(True)  # Ensure x requires gradients
        u = self.model(x)

        # First derivatives
        du_dx = th.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_x = du_dx[:, 0]
        u_y = du_dx[:, 1]

        # Second derivatives
        u_xx = th.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0]
        u_yy = th.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, 1]

        return u_xx + u_yy, u

    def loss(self) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute PDE loss"""
        x_interior, x_boundary = self.sample_points()
        x_interior = x_interior.to(next(self.parameters()).device)
        x_boundary = x_boundary.to(next(self.parameters()).device)

        # Interior loss
        laplacian, _ = self.compute_gradients(x_interior)
        interior_loss = th.mean((laplacian - self.forcing(x_interior)) ** 2)

        # Boundary loss
        boundary_loss = th.mean(self.model(x_boundary) ** 2)

        # Total loss
        total_loss = self.alpha * interior_loss + boundary_loss

        return total_loss, interior_loss, boundary_loss

    def l2_error(self, n_points: int = 1000) -> th.Tensor:
        """Compute L2 error against true solution"""
        x = th.rand((n_points, 2), device=next(self.parameters()).device) * 2 - 1
        pred = self.model(x)
        true = self.true_solution(x)
        return th.mean((pred - true) ** 2)

    def h1_error(self, n_points: int = 1000) -> th.Tensor:
        """Compute H1 error against true solution"""
        device = next(self.parameters()).device

        # Generate points and explicitly set requires_grad
        x = th.rand((n_points, 2), device=device, requires_grad=True)
        x = 2 * x - 1  # Scale to [-1, 1]

        # Forward pass to get predictions
        pred = self.model(x)

        # Get true solution
        true = self.true_solution(x)

        # L2 part
        l2_err = th.mean((pred - true) ** 2)

        # For predicted solution gradient
        pred_sum = pred.sum()
        pred_grad = th.autograd.grad(pred_sum, x, create_graph=True, retain_graph=True)[0]

        # Analytical gradients for true solution
        true_grad_x = np.pi * th.cos(np.pi * x[:, 0].detach()) * th.sin(np.pi * x[:, 1].detach() ** 2)
        true_grad_y = 2 * np.pi * x[:, 1].detach() * th.sin(np.pi * x[:, 0].detach()) * th.cos(
            np.pi * x[:, 1].detach() ** 2)
        true_grad = th.stack([true_grad_x, true_grad_y], dim=1).to(device)

        # Gradient error
        grad_err = th.mean(th.sum((pred_grad - true_grad) ** 2, dim=1))

        return l2_err + grad_err