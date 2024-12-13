import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
import numpy as np

class PoissonSolver(nn.Module):
    def __init__(self, model: nn.Module,
                 n_interior: int = 10000,
                 n_boundary: int = 800,
                 alpha: float = 0.01,
                 domain: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.model = model
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.alpha = alpha
        self.domain_min, self.domain_max = domain
        
    def sample_points(self) -> Tuple[th.Tensor, th.Tensor]:
        """Sample interior and boundary points"""
        # Interior points scaled to domain
        x_interior = th.rand((self.n_interior, 2), device=next(self.parameters()).device)
        x_interior = x_interior * (self.domain_max - self.domain_min) + self.domain_min

        # Boundary points
        theta = th.linspace(0, 2*np.pi, self.n_boundary, device=next(self.parameters()).device)
        x_boundary = th.stack([
            (th.cos(theta) + 1) * (self.domain_max - self.domain_min)/2 + self.domain_min,
            (th.sin(theta) + 1) * (self.domain_max - self.domain_min)/2 + self.domain_min
        ], dim=1)
        
        return x_interior, x_boundary
    
    def compute_gradients(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Compute first and second derivatives with better gradient handling"""
        x.requires_grad_(True)
        u = self.model(x)
        
        # First derivatives - compute gradient of each output separately
        grads = []
        for i in range(u.shape[0]):
            grad = th.autograd.grad(u[i], x, create_graph=True, retain_graph=True)[0][i]
            grads.append(grad)
        du_dx = th.stack(grads)
        
        # Second derivatives 
        laplacian = th.zeros_like(u)
        for i in range(2):  # For each dimension
            grad2s = []
            for j in range(u.shape[0]):
                grad2 = th.autograd.grad(du_dx[j,i], x, create_graph=True, retain_graph=True)[0][j,i]
                grad2s.append(grad2)
            laplacian += th.stack(grad2s)
            
        return laplacian, u

    def h1_error(self, n_points: int = 1000) -> th.Tensor:
        """Compute H1 error with proper scaling"""
        device = next(self.parameters()).device
        x = th.rand((n_points, 2), device=device, requires_grad=True)
        x = x * (self.domain_max - self.domain_min) + self.domain_min
        
        # Predicted solution and gradient
        pred = self.model(x)
        pred_grad = th.autograd.grad(pred.sum(), x, create_graph=True)[0]
        
        # True solution and gradient
        with th.no_grad():
            true = self.true_solution(x)
            true_grad_x = np.pi * th.cos(np.pi * x[:,0]) * th.sin(np.pi * x[:,1]**2)
            true_grad_y = 2 * np.pi * x[:,1] * th.sin(np.pi * x[:,0]) * th.cos(np.pi * x[:,1]**2)
            true_grad = th.stack([true_grad_x, true_grad_y], dim=1)
        
        # Compute errors with proper scaling
        l2_err = th.mean((pred - true)**2)
        h1_seminorm = th.mean(th.sum((pred_grad - true_grad)**2, dim=1))
        
        return l2_err + 0.01 * h1_seminorm  # Scale H1 seminorm term

    def loss(self) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Compute PDE loss"""
        x_interior, x_boundary = self.sample_points()

        # Interior loss
        laplacian, _ = self.compute_gradients(x_interior)
        interior_loss = th.mean((laplacian - self.forcing(x_interior)) ** 2)

        # Boundary loss
        boundary_loss = th.mean(self.model(x_boundary) ** 2)

        # Total loss
        total_loss = self.alpha * interior_loss + boundary_loss

        return total_loss, interior_loss, boundary_loss

    def forcing(self, x: th.Tensor) -> th.Tensor:
        """Right hand side f = -Δu"""
        return -np.pi ** 2 * (1 + 4 * x[:, 1] ** 2) * th.sin(np.pi * x[:, 0]) * \
            th.sin(np.pi * x[:, 1] ** 2) + \
            2 * np.pi * th.sin(np.pi * x[:, 0]) * th.cos(np.pi * x[:, 1] ** 2)

    def true_solution(self, x: th.Tensor) -> th.Tensor:
        """Ground truth solution u = sin(πx)sin(πy²)"""
        return th.sin(np.pi * x[:, 0]) * th.sin(np.pi * x[:, 1] ** 2)