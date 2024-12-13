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
        grid_sizes=[3, 5, 10],  # Reduced grid sizes
        kan_shapes=[(2, 5, 1)],  # Single shape
        mlp_depths=[2],  # Fewer depths
        mlp_width=20,  # Much smaller width
        n_epochs=50,  # Fewer epochs
        n_interior=1000,  # Reduced from 10000
        n_boundary=100,  # Reduced from 800
        device='cuda'  # Default to CPU to avoid GPU memory issues
):
    """Run PDE experiment with reduced memory usage"""
    device = th.device(device)
    results = {'kan': [], 'mlp': []}
    
    # Train KANs
    for shape in kan_shapes:
        shape_results = {'params': [], 'l2_errors': [], 'h1_errors': []}
        
        for grid_size in grid_sizes:
            try:
                print(f"\nTraining KAN shape {shape}, grid size {grid_size}")
                
                # Create model
                model = LinearKanLayers(
                    layers=[(shape[i], shape[i+1]) for i in range(len(shape)-1)],
                    act_fun=BSpline(degree=3, grid_size=grid_size),
                    res_act_fun=F.silu
                ).to(device)
                
                n_params = sum(p.numel() for p in model.parameters())
                print(f"Number of parameters: {n_params}")
                
                # Create solver with reduced points
                solver = PoissonSolver(
                    model, 
                    n_interior=n_interior,
                    n_boundary=n_boundary,
                    domain=(0, 1)
                ).to(device)
                
                optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
                
                # Training loop
                for epoch in range(n_epochs):
                    total_loss, interior_loss, boundary_loss = solver.loss()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}, Loss: {total_loss.item():.2e}")
                    
                    # Clear memory
                    if hasattr(th.cuda, 'empty_cache'):
                        th.cuda.empty_cache()
                
                # Record results
                with th.no_grad():
                    l2_err = solver.l2_error().item()
                    h1_err = solver.h1_error().item()
                    
                    shape_results['params'].append(n_params)
                    shape_results['l2_errors'].append(l2_err)
                    shape_results['h1_errors'].append(h1_err)
                    
                print(f"Final L2 error: {l2_err:.2e}")
                print(f"Final H1 error: {h1_err:.2e}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM for grid size {grid_size}")
                    if hasattr(th.cuda, 'empty_cache'):
                        th.cuda.empty_cache()
                    continue
                raise e
                
        results['kan'].append(shape_results)
    
    # Train MLPs with very small config
    for depth in mlp_depths:
        try:
            layers = [2] + [mlp_width] * (depth-1) + [1]
            model = MLP(layers).to(device)
            
            solver = PoissonSolver(
                model,
                n_interior=n_interior,
                n_boundary=n_boundary,
                domain=(0, 1)
            ).to(device)
            
            # Rest of MLP training code...
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM for MLP depth {depth}")
                if hasattr(th.cuda, 'empty_cache'):
                    th.cuda.empty_cache()
                continue
            raise e
    
    return results

def plot_pde_results(results: Dict, save_path: str = 'pde_results.png') -> None:
    """Plot PDE results in the style matching the paper Figure 3.3"""
    
    plt.style.use('seaborn-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color scheme
    kan_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # Blue, Green, Orange
    mlp_color = '#d62728'  # Red
    
    # Plot L2 errors
    ax1.set_title('L2 error squared')
    ax2.set_title('H1 error squared')
    
    # Plot KAN results
    kan_shapes = [(2, 5, 1), (2, 7, 1), (2, 10, 1)]
    for i, (shape_results, shape) in enumerate(zip(results['kan'], kan_shapes)):
        params = shape_results['params']
        l2_errors = shape_results['l2_errors']
        h1_errors = shape_results['h1_errors']
        
        label = f'KAN {shape}'
        ax1.loglog(params, l2_errors, 'o-', color=kan_colors[i], label=label, 
                  markersize=6, linewidth=2)
        ax2.loglog(params, h1_errors, 'o-', color=kan_colors[i], label=label,
                  markersize=6, linewidth=2)
    
    # Plot MLP results if available
    if results['mlp']:
        mlp_params = [r['params'] for r in results['mlp']]
        mlp_l2_errors = [r['l2_error'] for r in results['mlp']]
        mlp_h1_errors = [r['h1_error'] for r in results['mlp']]
        
        ax1.loglog(mlp_params, mlp_l2_errors, 's-', color=mlp_color, 
                  label='MLP', markersize=6, linewidth=2)
        ax2.loglog(mlp_params, mlp_h1_errors, 's-', color=mlp_color,
                  label='MLP', markersize=6, linewidth=2)
    
    # Add theoretical scaling lines
    x = np.logspace(1, 5, 100)
    ax1.loglog(x, 1e-1 * x**(-4), 'k--', label='N⁻⁴', linewidth=1.5)
    ax2.loglog(x, 1e-1 * x**(-4), 'k--', label='N⁻⁴', linewidth=1.5)
    
    # Formatting
    for ax in [ax1, ax2]:
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_xlabel('Number of parameters')
        
        # Set reasonable axis limits based on data
        ax.set_xlim(1e1, 1e5)
        ax.set_ylim(1e-7, 1e1)
        
        # Format ticks
        ax.tick_params(direction='in', length=6, width=1, colors='k',
                      grid_color='gray', grid_alpha=0.5)
        
        # Legend
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    ax1.set_ylabel('L2 error squared')
    ax2.set_ylabel('H1 error squared')
    
    # Overall title
    plt.suptitle('PDE Solving: KAN vs MLP Scaling', y=1.02)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()