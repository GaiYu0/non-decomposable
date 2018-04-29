import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, nonlinear):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 2)
        self.linear1 = nn.Linear(16 * 7 * 7, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
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
