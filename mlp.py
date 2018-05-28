import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, features, nonlinear):
        super(MLP, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(in_features, out_features) for in_features, out_features in zip(features[:-1], features[1:])])
        self.nonlinear = nonlinear

    def forward(self, x):
        for linear in self.linear_list[:-1]:
            x = self.nonlinear(linear(x))
        return self.linear_list[-1](x)
