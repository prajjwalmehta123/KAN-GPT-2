import numpy as np
import torch
import torch.nn as nn

from .spline import coef2curve, curve2coef


class KANLayer(nn.Module):
    """
    KANLayer class
    """

    def __init__(
        self,
        in_dim=3,
        out_dim=2,
        num=5,
        k=3,
        noise_scale=0.1,
        scale_base=1.0,
        scale_sp=1.0,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, 1],
        sp_trainable=True,
        sb_trainable=True,
    ):
        """'
        initialize a KANLayer
        """
        super(KANLayer, self).__init__()
        # size
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        self.grid = torch.einsum(
            "i,j->ij",
            torch.ones(
                size,
            ),
            torch.linspace(grid_range[0], grid_range[1], steps=num + 1),
        )
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        self.noises = (
            (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        )
        # shape: (size, coef)
        self.coef = torch.nn.Parameter(
            curve2coef(self.grid, self.noises, self.grid, k)
        )
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(
                torch.ones(
                    size,
                )
                * scale_base
            ).requires_grad_(
                sb_trainable
            )  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(scale_base).requires_grad_(
                sb_trainable
            )
        self.scale_sp = torch.nn.Parameter(
            torch.ones(
                size,
            )
            * scale_sp
        ).requires_grad_(
            sp_trainable
        )  # make scale trainable
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(
            torch.ones(
                size,
            )
        ).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size)

    def forward(self, x):
        """
        KANLayer forward given input x
        """
        batch = x.shape[0]
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x = (
            torch.einsum(
                "ij,k->ikj",
                x,
                torch.ones(
                    self.out_dim,
                ).type_as(x),
            )
            .reshape(batch, self.size)
            .permute(1, 0)
        )
        preacts = (
            x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim)
        )
        base = self.base_fun(x).permute(1, 0)  # shape (batch, size)
        y = coef2curve(
            x_eval=x,
            grid=self.grid[self.weight_sharing],
            coef=self.coef[self.weight_sharing],
            k=self.k,
            device=x.device,
        )  # shape (size, batch)
        y = y.permute(1, 0)  # shape (batch, size)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = (
            self.scale_base.unsqueeze(dim=0) * base
            + self.scale_sp.unsqueeze(dim=0) * y
        )
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = torch.sum(
            y.reshape(batch, self.out_dim, self.in_dim), dim=2
        )  # shape (batch, out_dim)
        # y shape: (batch, out_dim); preacts shape: (batch, in_dim, out_dim)
        # postspline shape: (batch, in_dim, out_dim); postacts: (batch, in_dim, out_dim)
        # postspline is for extension; postacts is for visualization
        return y, preacts, postacts, postspline

    def update_grid_from_samples(self, x):
        """
        update grid from samples
        """
        batch = x.shape[0]
        x = (
            torch.einsum(
                "ij,k->ikj",
                x,
                torch.ones(
                    self.out_dim,
                ).type_as(x),
            )
            .reshape(batch, self.size)
            .permute(1, 0)
        )
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coef2curve(
            x_pos, self.grid, self.coef, self.k, device=x.device
        )
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [
            -1
        ]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat(
            [
                grid_adaptive[:, [0]]
                - margin
                + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin)
                * a
                for a in np.linspace(0, 1, num=self.grid.shape[1])
            ],
            dim=1,
        )
        self.grid.data = (
            self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        )
        self.coef.data = curve2coef(
            x_pos, y_eval, self.grid, self.k, device=x.device
        )

    def initialize_grid_from_parent(self, parent, x):
        """
        update grid from a parent KANLayer & samples
        """
        batch = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = (
            torch.einsum(
                "ij,k->ikj",
                x,
                torch.ones(
                    self.out_dim,
                ).type_as(x),
            )
            .reshape(batch, self.size)
            .permute(1, 0)
        )
        x_pos = parent.grid
        sp2 = KANLayer(
            in_dim=1,
            out_dim=self.size,
            k=1,
            num=x_pos.shape[1] - 1,
            scale_base=0.0,
        ).type_as(x)
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1)
        y_eval = coef2curve(
            x_eval, parent.grid, parent.coef, parent.k, device=x.device
        )
        percentile = torch.linspace(-1, 1, self.num + 1).type_as(x)
        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(
            x_eval, y_eval, self.grid, self.k, x.device
        )

    def get_subset(self, in_id, out_id):
        """
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        """
        spb = KANLayer(
            len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun
        )
        spb.grid.data = self.grid.reshape(
            self.out_dim, self.in_dim, spb.num + 1
        )[out_id][:, in_id].reshape(-1, spb.num + 1)
        spb.coef.data = self.coef.reshape(
            self.out_dim, self.in_dim, spb.coef.shape[1]
        )[out_id][:, in_id].reshape(-1, spb.coef.shape[1])
        spb.scale_base.data = self.scale_base.reshape(
            self.out_dim, self.in_dim
        )[out_id][:, in_id].reshape(
            -1,
        )
        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[
            out_id
        ][:, in_id].reshape(
            -1,
        )
        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][
            :, in_id
        ].reshape(
            -1,
        )

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb

    def lock(self, ids):
        """
        lock activation functions to share parameters based on ids
        """
        self.lock_counter += 1
        # ids: [[i1,j1],[i2,j2],[i3,j3],...]
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = (
                    ids[0][1] * self.in_dim + ids[0][0]
                )
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = (
                self.lock_counter
            )

    def unlock(self, ids):
        """
        unlock activation functions
        """
        # check ids are locked
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]]
                == self.weight_sharing[ids[0][1] * self.in_dim + ids[0][0]]
            )
        if not locked:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = (
                ids[i][1] * self.in_dim + ids[i][0]
            )
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1