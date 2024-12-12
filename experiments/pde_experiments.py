import torch as th
import matplotlib.pyplot as plt
from typing import Tuple, List
from tqdm import tqdm

import torch.nn.functional as F
from ..networks import LinearKanLayers, BSpline
from ..pde import PoissonSolver
from utils import train_model,MLP


def run_pde_experiment(
        grid_sizes: List[int] = [3, 5, 10, 20, 50, 100, 200],
        kan_shapes: List[Tuple[int, ...]] = [(2, 5, 1), (2, 7, 1), (2, 10, 1)],
        mlp_depths: List[int] = [2, 3, 4, 5],
        mlp_width: int = 100,
        device: str = 'cuda'
) -> None:
    """Run PDE solving experiment comparing KANs and MLPs"""

    device = th.device(device if th.cuda.is_available() else 'cpu')
    results = {'kan': [], 'mlp': []}

    # Train KANs of different shapes
    for shape in kan_shapes:
        shape_results = {'grid_sizes': [], 'l2_errors': [], 'h1_errors': []}

        for grid_size in grid_sizes:
            print(f"\nTraining KAN shape {shape} with grid size {grid_size}")

            # Create KAN model
            model = LinearKanLayers(
                layers=[(shape[i], shape[i + 1]) for i in range(len(shape) - 1)],
                act_fun=BSpline(degree=3, grid_size=grid_size),
                res_act_fun=F.silu
            ).to(device)

            # Create PDE solver
            solver = PoissonSolver(model).to(device)
            optimizer = th.optim.Adam(model.parameters())

            # Training loop
            for epoch in tqdm(range(250)):
                loss, _, _ = solver.loss()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Record results
            with th.no_grad():
                l2_err = solver.l2_error().item()
                h1_err = solver.h1_error().item()

                shape_results['grid_sizes'].append(grid_size)
                shape_results['l2_errors'].append(l2_err)
                shape_results['h1_errors'].append(h1_err)

        results['kan'].append(shape_results)

    # Train MLPs of different depths
    for depth in mlp_depths:
        print(f"\nTraining MLP depth {depth}")

        # Create MLP model
        layers = [2] + [mlp_width] * (depth - 1) + [1]
        model = MLP(layers).to(device)

        # Create PDE solver
        solver = PoissonSolver(model).to(device)
        optimizer = th.optim.Adam(model.parameters())

        # Training loop
        for epoch in tqdm(range(250)):
            loss, _, _ = solver.loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Record results
        with th.no_grad():
            l2_err = solver.l2_error().item()
            h1_err = solver.h1_error().item()

            results['mlp'].append({
                'depth': depth,
                'l2_error': l2_err,
                'h1_error': h1_err
            })

    # Plot results
    plot_pde_results(results)


def plot_pde_results(results: dict) -> None:
    """Plot PDE experiment results"""
    plt.figure(figsize=(12, 5))

    # L2 error plot
    plt.subplot(121)
    for kan_result in results['kan']:
        plt.loglog(kan_result['grid_sizes'],
                   kan_result['l2_errors'],
                   'o-', label=f'KAN')

    mlp_params = [model['params'] for model in results['mlp']]
    mlp_l2_errors = [model['l2_error'] for model in results['mlp']]
    plt.loglog(mlp_params, mlp_l2_errors, 's-', label='MLP')

    plt.grid(True)
    plt.xlabel('Number of parameters')
    plt.ylabel('L2 error')
    plt.legend()

    # H1 error plot
    plt.subplot(122)
    for kan_result in results['kan']:
        plt.loglog(kan_result['grid_sizes'],
                   kan_result['h1_errors'],
                   'o-', label=f'KAN')

    mlp_h1_errors = [model['h1_error'] for model in results['mlp']]
    plt.loglog(mlp_params, mlp_h1_errors, 's-', label='MLP')

    plt.grid(True)
    plt.xlabel('Number of parameters')
    plt.ylabel('H1 error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pde_results.png')
    plt.show()
