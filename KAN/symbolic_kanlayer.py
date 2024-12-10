import torch
import torch.nn as nn

from .utils import SYMBOLIC_LIB, fit_params


class Symbolic_KANLayer(nn.Module):
    """
    KANLayer class
  """

    def __init__(self, in_dim=3, out_dim=2):
        """
        initialize a Symbolic_KANLayer (activation functions are initialized to be identity functions)
        """
        super(Symbolic_KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mask = torch.nn.Parameter(
            torch.zeros(out_dim, in_dim)
        ).requires_grad_(False)
        # torch
        self.funs = [
            [lambda x: x for i in range(self.in_dim)]
            for j in range(self.out_dim)
        ]
        # name
        self.funs_name = [
            ["" for i in range(self.in_dim)] for j in range(self.out_dim)
        ]
        # sympy
        self.funs_sympy = [
            ["" for i in range(self.in_dim)] for j in range(self.out_dim)
        ]

        self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4))
        # c*f(a*x+b)+d

    def forward(self, x):
        """
        forward
        """

        batch = x.shape[0]  # noqa
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                xij = (
                    self.affine[j, i, 2]
                    * self.funs[j][i](
                        self.affine[j, i, 0] * x[:, [i]] + self.affine[j, i, 1]
                    )
                    + self.affine[j, i, 3]
                )
                postacts_.append(self.mask[j][i] * xij)
            postacts.append(torch.stack(postacts_))

        postacts = torch.stack(postacts)
        postacts = postacts.permute(2, 1, 0, 3)[:, :, :, 0]
        y = torch.sum(postacts, dim=2)

        return y, postacts

    def get_subset(self, in_id, out_id):
        """
        get a smaller Symbolic_KANLayer from a larger Symbolic_KANLayer (used for pruning)
        """
        sbb = Symbolic_KANLayer(self.in_dim, self.out_dim)
        sbb.in_dim = len(in_id)
        sbb.out_dim = len(out_id)
        sbb.mask.data = self.mask.data[out_id][:, in_id]
        sbb.funs = [[self.funs[j][i] for i in in_id] for j in out_id]
        sbb.funs_sympy = [
            [self.funs_sympy[j][i] for i in in_id] for j in out_id
        ]
        sbb.funs_name = [[self.funs_name[j][i] for i in in_id] for j in out_id]
        sbb.affine.data = self.affine.data[out_id][:, in_id]
        return sbb

    def fix_symbolic(
        self,
        i,
        j,
        fun_name,
        x=None,
        y=None,
        random=False,
        a_range=(-10, 10),
        b_range=(-10, 10),
        verbose=True,
    ):
        """
        fix an activation function to be symbolic
        """
        if isinstance(fun_name, str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name
            if x is None or y is None:
                # initialzie from just fun
                self.funs[j][i] = fun
                if not random:
                    self.affine.data[j][i] = torch.tensor([1.0, 0.0, 1.0, 0.0])
                else:
                    self.affine.data[j][i] = (
                        torch.rand(
                            4,
                        )
                        * 2
                        - 1
                    )
                return None
            else:
                # initialize from x & y and fun
                params, r2 = fit_params(
                    x,
                    y,
                    fun,
                    a_range=a_range,
                    b_range=b_range,
                    verbose=verbose,
                )
                self.funs[j][i] = fun
                self.affine.data[j][i] = params
                return r2
        else:
            # if fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            if not random:
                self.affine.data[j][i] = torch.tensor([1.0, 0.0, 1.0, 0.0])
            else:
                self.affine.data[j][i] = (
                    torch.rand(
                        4,
                    )
                    * 2
                    - 1
                )
            return None