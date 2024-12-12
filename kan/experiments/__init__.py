from .pde_experiments import run_pde_experiment
from .function_fitting import run_function_fitting_experiment, plot_function_fitting_results
from .utils import MLP,train_model


__all__ = [
    'run_pde_experiment',
    'run_function_fitting_experiment',
    'plot_function_fitting_results'
]