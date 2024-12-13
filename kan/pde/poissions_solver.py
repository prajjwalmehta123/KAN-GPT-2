import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
import numpy as np

class PoissonSolver(nn.Module):
    def __init__(self, model: nn.Module,
                 n_interior: int = 100,
                 n_boundary: int = 20,
                 alpha: float = 0.01,
                 domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.model = model
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.alpha = alpha
        self.domain_min, self.domain_max = domain

    def compute_interior_loss(self, x_interior: th.Tensor) -> th.Tensor:
        """Compute PDE loss on interior points"""
        x_interior.requires_grad_(True)
        u = self.model(x_interior)

        # Compute derivatives batch-wise
        laplacian = th.zeros_like(u)
        for i in range(x_interior.shape[0]):
            # First derivatives
            grad = th.autograd.grad(u[i], x_interior, create_graph=True, retain_graph=True)[0][i]

            # Second derivatives
            for j in range(2):
                grad2 = th.autograd.grad(grad[j], x_interior, create_graph=True, retain_graph=True)[0][i, j]
                laplacian[i] += grad2

        return th.mean((laplacian - self.forcing(x_interior)) ** 2)

    def compute_boundary_loss(self, x_boundary: th.Tensor) -> th.Tensor:
        """Compute boundary condition loss"""
        return th.mean(self.model(x_boundary) ** 2)

    def forcing(self, x: th.Tensor) -> th.Tensor:
        """Right hand side f = -Δu"""
        return -np.pi ** 2 * (1 + 4 * x[:, 1] ** 2) * th.sin(np.pi * x[:, 0]) * \
            th.sin(np.pi * x[:, 1] ** 2) + \
            2 * np.pi * th.sin(np.pi * x[:, 0]) * th.cos(np.pi * x[:, 1] ** 2)

    def l2_error(self, n_points: int = 100) -> th.Tensor:
        """Compute L2 error with small number of points"""
        device = next(self.parameters()).device
        x = th.rand((n_points, 2), device=device)
        x = x * (self.domain_max - self.domain_min) + self.domain_min

        with th.no_grad():
            pred = self.model(x)
            true = self.true_solution(x)
        return th.mean((pred - true) ** 2)

    def h1_error(self, n_points: int = 100) -> th.Tensor:
        """Compute H1 error with small number of points"""
        device = next(self.parameters()).device
        x = th.rand((n_points, 2), device=device, requires_grad=True)
        x = x * (self.domain_max - self.domain_min) + self.domain_min

        pred = self.model(x)
        pred_grad = th.autograd.grad(pred.sum(), x, create_graph=True)[0]

        with th.no_grad():
            true = self.true_solution(x)
            true_grad_x = np.pi * th.cos(np.pi * x[:, 0]) * th.sin(np.pi * x[:, 1] ** 2)
            true_grad_y = 2 * np.pi * x[:, 1] * th.sin(np.pi * x[:, 0]) * th.cos(np.pi * x[:, 1] ** 2)
            true_grad = th.stack([true_grad_x, true_grad_y], dim=1)

        l2_err = th.mean((pred - true) ** 2)
        h1_seminorm = th.mean(th.sum((pred_grad - true_grad) ** 2, dim=1))

        return l2_err + h1_seminorm

    def true_solution(self, x: th.Tensor) -> th.Tensor:
        """Ground truth solution u = sin(πx)sin(πy²)"""
        return th.sin(np.pi * x[:, 0]) * th.sin(np.pi * x[:, 1] ** 2)