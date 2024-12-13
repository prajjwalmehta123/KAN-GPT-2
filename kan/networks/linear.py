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
    ):
        super().__init__()

        self.__act_fun = act_fun
        self.__res_act_fun = res_act_fun

        # Initialize spline coefficients
        self._c = nn.Parameter(
            th.zeros(self.__act_fun.get_size(), out_features, in_features)
        )

        # Initialize residual weights
        self._w_b = nn.Parameter(th.ones(out_features, in_features))
        self._w_s = nn.Parameter(th.ones(out_features, in_features))

        # Xavier initialization for residual weights
        xavier_normal_(self._w_b, gain=1e-3)
        # Initialize spline coefficients close to zero
        normal_(self._c, 0, 1e-3)

    def forward(self, x: th.Tensor):
        # x shape: [batch_size, in_features]
        batch_size = x.size(0)

        # Get basis values: [batch_size, basis_size, in_features]
        basis_vals = self.__act_fun(x)

        # Compute spline outputs
        # [batch, basis, in] @ [basis, out, in] -> [batch, out, in]
        spline_out = th.einsum('bni,aoi->boi', basis_vals, self._c)

        # Scale spline outputs
        # [batch, out, in] * [out, in] -> [batch, out, in]
        spline_out = spline_out * self._w_s

        # Compute residual path
        # [batch, in] -> [batch, in] -> [batch, out, in]
        res_out = (self.__res_act_fun(x).unsqueeze(1) * self._w_b.unsqueeze(0))

        # Combine paths and sum over input dimension
        # [batch, out, in] -> [batch, out]
        return (spline_out + res_out).sum(dim=-1)

    def extend_grid(self, new_grid_size: int):
        """Extend grid for activation functions"""
        # Store old coefficients
        old_coeffs = self._c.data

        # Extend activation function grid
        self.__act_fun.extend_grid(new_grid_size)

        # Create new parameter tensor with extended size
        new_size = self.__act_fun.get_size()
        new_coeffs = th.zeros(new_size, self._c.size(1), self._c.size(2),
                              device=self._c.device)

        # Copy old coefficients
        new_coeffs[:old_coeffs.size(0)] = old_coeffs

        # Update parameter
        self._c = nn.Parameter(new_coeffs)