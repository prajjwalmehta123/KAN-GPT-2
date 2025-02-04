import torch as th
import numpy as np
from scipy import special
from typing import Callable, Dict, Tuple
from torch.utils.data import Dataset


class SpecialFunctionDataset(Dataset):
    """Dataset for fitting special functions"""

    FUNCTIONS: Dict[str, Callable] = {
        'ellipj': lambda x, y: special.ellipj(x, y)[0],
        'ellipkinc': special.ellipkinc,
        'ellipeinc': special.ellipeinc,
        'jv': special.jv,
        'yv': special.yv,
        'kv': special.kv,
        'iv': special.iv,
        'sph_harm': lambda m, n, x, y: np.abs(special.sph_harm(m, n, x, y))
    }

    def __init__(self,
                 function_name: str,
                 n_samples: int = 1000,
                 param_ranges: Dict[str, Tuple[float, float]] = None,
                 train: bool = True):
        """
        Args:
            function_name: Name of special function to fit
            n_samples: Number of samples
            param_ranges: Dict of parameter ranges {param: (min, max)}
            train: If True, use training set random seed
        """
        self.function_name = function_name
        self.n_samples = n_samples

        # Default parameter ranges
        if param_ranges is None:
            param_ranges = {
                'x': (0, 1),
                'y': (0, 1)
            }
        self.param_ranges = param_ranges

        # Set random seed
        np.random.seed(0 if train else 1)

        # Generate random inputs
        self.inputs = []
        for param, (min_val, max_val) in param_ranges.items():
            param_vals = np.random.uniform(min_val, max_val, n_samples)
            self.inputs.append(param_vals)
        self.inputs = np.stack(self.inputs, axis=1)

        # Compute function values
        self.outputs = self.FUNCTIONS[function_name](*[self.inputs[:, i]
                                                       for i in range(self.inputs.shape[1])])

        # Convert to tensors
        self.inputs = th.FloatTensor(self.inputs)
        self.outputs = th.FloatTensor(self.outputs)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, th.Tensor]:
        return self.inputs[idx], self.outputs[idx]
