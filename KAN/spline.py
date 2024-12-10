import torch


def B_batch(x, grid, k=0, extend=True, device="cpu"):
    """
    evaludate x on B-spline bases
    """

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(
            x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device
        )
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:(-k)]
        ) * B_km1[
            :, 1:
        ]
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    """
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    y_eval = torch.einsum(
        "ij,ijk->ik", coef, B_batch(x_eval, grid, k, device=device)
    )
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    """
    converting B-spline curves to B-spline coefficients using least squares.
    """
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    coef = torch.linalg.lstsq(
        mat.to("cpu"), y_eval.unsqueeze(dim=2).to("cpu")
    ).solution[
        :, :, 0
    ]
    return coef.to(device)