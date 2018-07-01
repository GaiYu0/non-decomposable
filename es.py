import torch as th
import my


class GuidedES:
    def __init__(self, std, alpha, n, k):
        self.std, self.alpha, self.n, self.k = std, alpha, n, k
        self.g_list = []

    def perturb(self, module):
        std, alpha, n, k = self.std, self.alpha, self.n, self.k 
        if len(self.g_list) < k or any(p.grad is None for p in module.parameters()):
            my.perturb(module, std)
        else:
            self.g_list = self.g_list[1:]
            self.g_list.append(th.cat([p.grad.view(-1, 1) for p in module.parameters()]))
            u, _ = th.qr(th.cat(self.g_list, 1))
            g = std * (alpha / n) ** 0.5 * th.randn(n, 1) + \
                    std * ((1 - alpha) / k) ** 0.5 * th.mm(u, th.randn(k, 1))
            i = 0
            for p in module.parameters():
                p.data += g[i : i + p.grad.nelem()].view(p.data.shape)
                i += p.data.numel()
