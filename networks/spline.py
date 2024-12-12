# -*- coding: utf-8 -*-
from typing import Dict

import torch as th

from .activation import ActivationFunction


def b_spline(
    x: th.Tensor, k: int, n: int, x_min: float = 0.0, x_max: float = 1.0
) -> th.Tensor:
    x = x.unsqueeze(-1)

    def __knots(_i: th.Tensor) -> th.Tensor:
        return _i / n * (x_max - x_min) + x_min

    i_s = th.arange(-k, n, device=x.device)

    def __b_spline(curr_i_s: th.Tensor, curr_k: int) -> th.Tensor:
        if curr_k == 0:
            return th.logical_and(
                th.le(__knots(curr_i_s), x), th.lt(x, __knots(curr_i_s + 1))
            ).to(th.float)

        return __b_spline(curr_i_s, curr_k - 1) * (x - __knots(curr_i_s)) / (
            __knots(curr_i_s + curr_k) - __knots(curr_i_s)
        ) + __b_spline(curr_i_s + 1, curr_k - 1) * (
            __knots(curr_i_s + curr_k + 1) - x
        ) / (
            __knots(curr_i_s + curr_k + 1) - __knots(curr_i_s + 1)
        )

    return th.movedim(__b_spline(i_s, k), -1, 1)


def interpolate_bspline(old_grid: th.Tensor, old_coeffs: th.Tensor, new_grid: th.Tensor) -> th.Tensor:
    """
    Interpolate B-spline coefficients from old grid to new grid.
    """
    # Get dimensions
    n_old = old_grid.size(0)
    n_new = new_grid.size(0)

    # Create evaluation matrix for old grid
    old_basis = th.zeros((n_old, n_old), device=old_grid.device)
    for i in range(n_old):
        x = old_grid[i]
        # Evaluate B-spline basis functions at x
        basis_vals = b_spline(x.unsqueeze(0), k=3, n=n_old)  # k=3 for cubic splines
        old_basis[i] = basis_vals.squeeze()

    # Solve linear system to get spline coefficients
    # f(x) = Σ cᵢBᵢ(x) where Bᵢ are basis functions
    spline_coeffs = th.linalg.solve(old_basis, old_coeffs.T).T

    # Create evaluation matrix for new grid
    new_basis = th.zeros((n_new, n_new), device=new_grid.device)
    for i in range(n_new):
        x = new_grid[i]
        basis_vals = b_spline(x.unsqueeze(0), k=3, n=n_new)
        new_basis[i] = basis_vals.squeeze()

    # Interpolate to get new coefficients
    new_coeffs = th.matmul(spline_coeffs, new_basis)

    return new_coeffs


class BSpline(ActivationFunction):
    def __init__(self, degree: int, grid_size: int) -> None:
        super().__init__()

        self.__degree = degree
        self.__grid_size = grid_size
        self.register_buffer(
            "_coefficients",
            th.zeros(self.get_size())
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        basis_vals = b_spline(x, self.__degree, self.__grid_size)
        return th.matmul(basis_vals, self._coefficients)

    def get_size(self) -> int:
        return self.__grid_size + self.__degree

    def extend_grid(self, new_grid_size: int):
        """Extend the grid while preserving function behavior"""
        if new_grid_size <= self.__grid_size:
            return

        # Store old coefficients and grid points
        old_grid = th.linspace(0, 1, self.__grid_size)
        old_coeffs = self._coefficients

        # Create new grid points
        new_grid = th.linspace(0, 1, new_grid_size)

        # Interpolate old function to get coefficients at new grid points
        new_coeffs = interpolate_bspline(old_grid, old_coeffs, new_grid)

        # Update grid size and coefficients
        self.__grid_size = new_grid_size
        self._coefficients = new_coeffs

    @classmethod
    def from_dict(cls, options: Dict[str, str]) -> "ActivationFunction":
        assert (
            "degree" in options
        ), 'Must specify "degree", example : "-a degree=2"'
        assert (
            "grid_size" in options
        ), 'Must specify "grid_size", example : "-a grid_size=8"'

        return cls(
            int(options["degree"]),
            int(options["grid_size"]),
        )
