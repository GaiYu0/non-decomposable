import torch as th
import torch.nn as nn
import torch.nn.functional as F


class WeightedL1Loss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.w = nn.Parameter(th.ones(n_classes, 1))

    def forward(self, yz):
        yz = th.chunk(yz, int(yz.size(1) / self.n_classes), 1)
        y, z = th.cat(yz[::2]), th.cat(yz[1::2])
        w = F.embedding(th.max(y, 1)[1], F.softmax(self.w, 0))
        return -th.sum(w * th.abs(y - z))
