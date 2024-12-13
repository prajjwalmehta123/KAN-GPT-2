from typing import Callable, List, Tuple

import torch as th
from torch import nn

from .activation import ActivationFunction
from .linear import LinearKAN
from .utils import InfoModule


class LinearKanLayers(nn.Sequential, InfoModule):
    def __init__(
        self,
        layers: List[Tuple[int, int]],
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ):
        super().__init__(
            *[LinearKAN(c_i, c_o, act_fun, res_act_fun) for c_i, c_o in layers]
        )
