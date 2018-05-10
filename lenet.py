import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, in_channels, out_features, size=32, nonlinear=F.tanh):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, 1, {28 : 2, 32 : 0}[size])
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, out_features)
        self.nonlinear = nonlinear

    def forward(self, x):
        x = self.nonlinear(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.nonlinear(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.nonlinear(self.linear1(x))
        x = self.nonlinear(self.linear2(x))
        x = self.linear3(x)
        return x
