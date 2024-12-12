import torch as th
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


from ..networks import LinearKanLayers, BSpline
from ..function_fitting import SpecialFunctionDataset, FeynmanDataset
from .utils import MLP, train_model


def run_function_fitting_experiment(
        function_type: str = 'special',  # 'special' or 'feynman'
        function_name: str = 'jv',  # function identifier
        grid_sizes: List[int] = [3, 5, 10, 20, 50, 100, 200],
        kan_shapes: List[List[int]] = [[2, 5, 1], [2, 7, 1], [2, 10, 1]],
        mlp_configs: List[Dict] = [
            {'depth': d, 'width': 100}
            for d in [2, 3, 4, 5]
        ],
        device: str = 'cuda',
        n_epochs: int = 100
) -> Dict:
    """Run function fitting experiment comparing KANs and MLPs"""

    device = th.device(device if th.cuda.is_available() else 'cpu')
    results = {'kan': [], 'mlp': []}

    # Create datasets
    if function_type == 'special':
        dataset_class = SpecialFunctionDataset
    else:
        dataset_class = FeynmanDataset

    train_dataset = dataset_class(function_name, train=True)
    test_dataset = dataset_class(function_name, train=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Train KANs of different shapes
    for shape in kan_shapes:
        shape_results = {
            'grid_sizes': [],
            'train_losses': [],
            'test_losses': [],
            'params': []
        }

        for grid_size in grid_sizes:
            print(f"\nTraining KAN shape {shape} with grid size {grid_size}")

            # Create KAN model
            model = LinearKanLayers(
                layers=[(shape[i], shape[i + 1]) for i in range(len(shape) - 1)],
                act_fun=BSpline(degree=3, grid_size=grid_size),
                res_act_fun=F.silu
            ).to(device)

            # Train model
            train_loss, test_loss = train_model(
                model, train_loader, test_loader,
                th.optim.Adam(model.parameters()),
                device, n_epochs=n_epochs
            )

            # Record results
            shape_results['grid_sizes'].append(grid_size)
            shape_results['train_losses'].append(train_loss)
            shape_results['test_losses'].append(test_loss)
            shape_results['params'].append(model.count_parameters())

        results['kan'].append(shape_results)

    # Train MLPs with different configurations
    for config in mlp_configs:
        print(f"\nTraining MLP depth {config['depth']} width {config['width']}")

        # Create MLP model
        layers = [train_dataset.inputs.shape[1]] + \
                 [config['width']] * (config['depth'] - 1) + \
                 [1]
        model = MLP(layers).to(device)

        # Train model
        train_loss, test_loss = train_model(
            model, train_loader, test_loader,
            th.optim.Adam(model.parameters()),
            device, n_epochs=n_epochs
        )

        # Record results
        results['mlp'].append({
            'config': config,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'params': model.count_parameters()
        })

    return results


def plot_function_fitting_results(
        results: Dict,
        function_name: str,
        save_path: str = None
) -> None:
    """Plot function fitting experiment results"""
    plt.figure(figsize=(12, 5))

    # Training curves
    plt.subplot(121)
    for kan_result in results['kan']:
        plt.loglog(kan_result['params'],
                   kan_result['test_losses'],
                   'o-', label=f'KAN')

    mlp_params = [r['params'] for r in results['mlp']]
    mlp_losses = [r['test_loss'] for r in results['mlp']]
    plt.loglog(mlp_params, mlp_losses, 's-', label='MLP')

    # Add theoretical scaling lines
    x = np.logspace(1, 5, 100)
    plt.loglog(x, 1e-1 * x ** (-4), '--', label='N⁻⁴ (KAN Theory)')
    plt.loglog(x, 1e-1 * x ** (-2), '--', label='N⁻² (MLP Theory)')

    plt.grid(True)
    plt.xlabel('Number of parameters')
    plt.ylabel('Test MSE')
    plt.title(f'Scaling Laws for {function_name}')
    plt.legend()

    # Parameter efficiency
    plt.subplot(122)
    all_params = []
    all_losses = []

    for kan_result in results['kan']:
        all_params.extend(kan_result['params'])
        all_losses.extend(kan_result['test_losses'])

    all_params.extend(mlp_params)
    all_losses.extend(mlp_losses)

    pareto_mask = np.ones(len(all_params), dtype=bool)
    for i, (p1, l1) in enumerate(zip(all_params, all_losses)):
        for p2, l2 in zip(all_params, all_losses):
            if p2 < p1 and l2 < l1:
                pareto_mask[i] = False
                break

    plt.loglog(np.array(all_params)[pareto_mask],
               np.array(all_losses)[pareto_mask],
               'k-', label='Pareto Frontier')

    plt.grid(True)
    plt.xlabel('Number of parameters')
    plt.ylabel('Test MSE')
    plt.title('Pareto Frontier')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
