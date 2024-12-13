import torch as th
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from ..networks import LinearKanLayers, BSpline
from ..pde import PoissonSolver
from .utils import MLP


def run_pde_experiment(
        grid_sizes=[3],
        kan_shapes=[(2, 3, 1)],
        mlp_depths=[2],
        mlp_width=10,
        n_epochs=50,
        n_interior=100,
        n_boundary=20,
        batch_size=10,
        device='cpu'
):
    """Run PDE experiment with proper gradient handling"""
    device = th.device(device)
    results = {'kan': [], 'mlp': []}

    # Train KANs
    for shape in kan_shapes:
        shape_results = {'params': [], 'l2_errors': [], 'h1_errors': []}

        for grid_size in grid_sizes:
            try:
                print(f"\nTraining KAN shape {shape}, grid size {grid_size}")

                model = LinearKanLayers(
                    layers=[(shape[i], shape[i + 1]) for i in range(len(shape) - 1)],
                    act_fun=BSpline(degree=3, grid_size=grid_size),
                    res_act_fun=F.silu
                ).to(device)

                n_params = sum(p.numel() for p in model.parameters())
                print(f"KAN parameters: {n_params}")

                solver = PoissonSolver(
                    model,
                    n_interior=n_interior,
                    n_boundary=n_boundary,
                    domain=(0, 1)
                ).to(device)

                optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

                # Training loop
                for epoch in range(n_epochs):
                    total_loss = 0
                    n_batches = n_interior // batch_size

                    for i in range(n_batches):
                        # Create tensors with gradients enabled
                        x_interior = th.rand((batch_size, 2), device=device, requires_grad=True)
                        x_boundary = th.rand((batch_size // 5, 2), device=device)

                        # Scale to domain
                        x_interior = x_interior * (solver.domain_max - solver.domain_min) + solver.domain_min
                        x_boundary = x_boundary * (solver.domain_max - solver.domain_min) + solver.domain_min

                        interior_loss = solver.compute_interior_loss(x_interior)
                        boundary_loss = solver.compute_boundary_loss(x_boundary)
                        batch_loss = solver.alpha * interior_loss + boundary_loss

                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()

                        total_loss += batch_loss.item()

                        # Clean up
                        del x_interior, x_boundary, interior_loss, boundary_loss, batch_loss
                        if hasattr(th.cuda, 'empty_cache'):
                            th.cuda.empty_cache()

                    if epoch % 5 == 0:
                        print(f"Epoch {epoch}, Average Loss: {total_loss / n_batches:.2e}")

                # Evaluate with proper gradient handling
                with th.no_grad():
                    l2_err = solver.l2_error(100).item()

                # H1 error needs gradients
                h1_err = solver.h1_error(100).item()

                shape_results['params'].append(n_params)
                shape_results['l2_errors'].append(l2_err)
                shape_results['h1_errors'].append(h1_err)

                print(f"KAN Final L2 error: {l2_err:.2e}")
                print(f"KAN Final H1 error: {h1_err:.2e}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM for grid size {grid_size}")
                    if hasattr(th.cuda, 'empty_cache'):
                        th.cuda.empty_cache()
                    continue
                raise e

        results['kan'].append(shape_results)

    # Train MLPs
    mlp_results = []
    for depth in mlp_depths:
        try:
            print(f"\nTraining MLP depth {depth}, width {mlp_width}")

            # Create MLP
            layers = [2] + [mlp_width] * (depth - 1) + [1]
            model = MLP(layers).to(device)

            n_params = sum(p.numel() for p in model.parameters())
            print(f"MLP parameters: {n_params}")

            # Create solver
            solver = PoissonSolver(
                model,
                n_interior=n_interior,
                n_boundary=n_boundary,
                domain=(0, 1)
            ).to(device)

            optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

            # Training loop with batching
            for epoch in range(n_epochs):
                total_loss = 0
                n_batches = n_interior // batch_size

                for i in range(n_batches):
                    x_interior = th.rand((batch_size, 2), device=device)
                    x_boundary = th.rand((batch_size // 5, 2), device=device)

                    interior_loss = solver.compute_interior_loss(x_interior)
                    boundary_loss = solver.compute_boundary_loss(x_boundary)
                    batch_loss = solver.alpha * interior_loss + boundary_loss

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    total_loss += batch_loss.item()

                    del x_interior, x_boundary, interior_loss, boundary_loss, batch_loss
                    if hasattr(th.cuda, 'empty_cache'):
                        th.cuda.empty_cache()

                if epoch % 5 == 0:
                    print(f"Epoch {epoch}, Average Loss: {total_loss / n_batches:.2e}")

            # Evaluate
            with th.no_grad():
                l2_err = solver.l2_error(100).item()

            h1_err = solver.h1_error(100).item()
            mlp_results.append({
                    'depth': depth,
                    'params': n_params,
                    'l2_error': l2_err,
                    'h1_error': h1_err
            })

            print(f"MLP Final L2 error: {l2_err:.2e}")
            print(f"MLP Final H1 error: {h1_err:.2e}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM for MLP depth {depth}")
                if hasattr(th.cuda, 'empty_cache'):
                    th.cuda.empty_cache()
                continue
            raise e

    results['mlp'] = mlp_results

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot L2 errors
    plt.subplot(121)
    for kan_result in results['kan']:
        plt.loglog(kan_result['params'],
                   kan_result['l2_errors'],
                   'o-', label='KAN')

    mlp_params = [r['params'] for r in results['mlp']]
    mlp_l2_errors = [r['l2_error'] for r in results['mlp']]
    plt.loglog(mlp_params, mlp_l2_errors, 's-', label='MLP')

    plt.grid(True)
    plt.xlabel('Number of parameters')
    plt.ylabel('L2 error squared')
    plt.title('L2 Error vs Parameters')
    plt.legend()

    # Plot H1 errors
    plt.subplot(122)
    for kan_result in results['kan']:
        plt.loglog(kan_result['params'],
                   kan_result['h1_errors'],
                   'o-', label='KAN')

    mlp_h1_errors = [r['h1_error'] for r in results['mlp']]
    plt.loglog(mlp_params, mlp_h1_errors, 's-', label='MLP')

    plt.grid(True)
    plt.xlabel('Number of parameters')
    plt.ylabel('H1 error squared')
    plt.title('H1 Error vs Parameters')
    plt.legend()

    plt.tight_layout()
    plt.savefig('pde_results.png')
    plt.close()

    return results