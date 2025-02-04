# -*- coding: utf-8 -*-
from typing import Callable

import torch as th
from torch import nn
from torch.nn.init import normal_, xavier_normal_

from .activation import ActivationFunction


class LinearKAN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_fun: ActivationFunction,
        res_act_fun: Callable[[th.Tensor], th.Tensor],
    ) -> None:
        super().__init__()

        self.__act_fun = act_fun
        self.__res_act_fun = res_act_fun

        self._w_b = nn.Parameter(th.ones(out_features, in_features))
        self._w_s = nn.Parameter(th.ones(out_features, in_features))
        self._c = nn.Parameter(
            th.ones(self.__act_fun.get_size(), out_features, in_features)
        )

        xavier_normal_(self._w_b, gain=1e-3)
        normal_(self._c, 0, 1e-3)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2

        return th.sum(
            th.einsum(
                "boi,oi->boi",
                th.einsum("bai,aoi->boi", self.__act_fun(x), self._c),
                self._w_s,
            )
            + th.einsum("bi,oi->boi", self.__res_act_fun(x), self._w_b),
            dim=2,
        )

    def extend_grid(self, new_grid_size: int):
        """Extend grid for all activation functions"""
        # Store old coefficients
        old_coeffs = self._c.data

        # Extend activation function grid
        self.act_fun.extend_grid(new_grid_size)

        # Create new parameter tensor with extended size
        new_size = self.act_fun.get_size()
        new_coeffs = th.zeros(new_size, self.out_features, self.in_features)

        # Copy old coefficients
        new_coeffs[:old_coeffs.size(0)] = old_coeffs

        # Update parameter
        self._c = nn.Parameter(new_coeffs)
