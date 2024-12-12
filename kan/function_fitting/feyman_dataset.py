import torch as th
import numpy as np
from typing import Callable, Dict, List, Tuple
from torch.utils.data import Dataset


class FeynmanDataset(Dataset):
    """Dataset for Feynman equations"""

    EQUATIONS: Dict[str, Tuple[Callable, List[str]]] = {
        'I.6.2': (
            lambda theta, sigma: np.exp(-theta ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2),
            ['theta', 'sigma']
        ),
        'I.6.2b': (
            lambda theta, theta1, sigma: np.exp(-(theta - theta1) ** 2 / (2 * sigma ** 2)) / np.sqrt(
                2 * np.pi * sigma ** 2),
            ['theta', 'theta1', 'sigma']
        ),
        # Add more equations here
    }

    def __init__(self,
                 equation_id: str,
                 n_samples: int = 1000,
                 param_ranges: Dict[str, Tuple[float, float]] = None,
                 train: bool = True):
        """
        Args:
            equation_id: Feynman equation identifier
            n_samples: Number of samples
            param_ranges: Dict of parameter ranges {param: (min, max)}
            train: If True, use training set random seed
        """
        self.equation_id = equation_id
        self.n_samples = n_samples

        equation_func, param_names = self.EQUATIONS[equation_id]

        # Default parameter ranges
        if param_ranges is None:
            param_ranges = {name: (-1, 1) for name in param_names}
        self.param_ranges = param_ranges

        # Set random seed
        np.random.seed(0 if train else 1)

        # Generate random inputs
        self.inputs = []
        for param in param_names:
            min_val, max_val = param_ranges[param]
            param_vals = np.random.uniform(min_val, max_val, n_samples)
            self.inputs.append(param_vals)
        self.inputs = np.stack(self.inputs, axis=1)

        # Compute equation values
        self.outputs = equation_func(*[self.inputs[:, i]
                                       for i in range(self.inputs.shape[1])])

        # Convert to tensors
        self.inputs = th.FloatTensor(self.inputs)
        self.outputs = th.FloatTensor(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, th.Tensor]:
        return self.inputs[idx], self.outputs[idx]