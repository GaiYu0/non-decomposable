import torch as th
import torch.nn as nn
import mlp


class RN(nn.Module):
    def __init__(self, n_objects, in_features, unary, binary, terminal, nonlinear, triu):
        super(RN, self).__init__()
        self.n_objects = n_objects
        self.in_features = in_features

        if unary:
            assert unary[0] == in_features
            assert unary[-1] * 2 == binary[0]
        else:
            assert binary[0] == in_features * 2
        assert binary[-1] == terminal[0]

        self.unary = mlp.MLP(unary, nonlinear) if unary else None
        self.binary = mlp.MLP(binary, nonlinear)
        self.terminal = mlp.MLP(terminal, nonlinear)
        self.nonlinear = nonlinear

        if triu:
            self.mask = th.triu(th.ones(self.n_objects, self.n_objects), 1)
            self.mask = self.mask.view(1, self.n_objects * self.n_objects, 1)
        else:
            self.mask = None

    def cuda(self):
        super().cuda()
        if self.mask is not None:
            self.mask = self.mask.cuda()

    def forward(self, x):
        """
        Parameters
        ----------
        x : (n, self.n_objects * self.in_features)
        """

        n, _ = x.shape
        if self.unary:
            x = x.view(n * self.n_objects, self.in_features)
            x = self.nonlinear(self.unary(x))
        x = x.view(n, self.n_objects, -1)
        u = x.unsqueeze(1).repeat(1, self.n_objects, 1, 1)
        v = x.unsqueeze(2).repeat(1, 1, self.n_objects, 1)
        x = th.cat((u, v), 3).view(n * self.n_objects * self.n_objects, -1)
        x = self.nonlinear(self.binary(x))
        x = x.view(n, self.n_objects * self.n_objects, -1)
        if self.mask is not None:
            x *= self.mask
        x = th.mean(x, 1)
        x = self.terminal(x)
        return x
