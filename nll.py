import torch as th
import torch.nn as nn
import torch.nn.functional as F
import my


class WeightedNLLLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.w = nn.Parameter(th.ones(n_classes, 1))

    def forward(self, yz):
        yzz = th.chunk(yz, int(yz.size(1) / self.n_classes), 1)
        y, z = th.cat(yzz[::2]), th.cat(yzz[1::2])
        w = F.embedding(th.max(y, 1)[1], F.softmax(self.w, 0))
        return -th.sum(w * y * z) / yz.size(0)

if __name__ == '__main__':
    n_samples = 100
    n_classes = 100

    for i in range(100):
        y = th.randint(n_classes - 1, [n_samples]).long()
        z = th.randn(n_samples, n_classes)

        onehot_x = my.one_hot(y, n_classes)
        logsoftmax_x = F.log_softmax(z, 1)

        nllloss_x = F.nll_loss(logsoftmax_x, y)

        cat_x = th.cat([onehot_x, logsoftmax_x], 1)
        weightednllloss_x = WeightedNLLLoss(n_classes)(cat_x.view(1, -1))

        assert abs(nllloss_x.item() - weightednllloss_x.item()) < 1e-3
