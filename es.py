import torch as th
import my


class GuidedES:
    def __init__(self, std, alpha, n, k, gpu):
        self.std, self.alpha, self.n, self.k = std, alpha, n, k
        self.device = th.device('cpu') if gpu < 0 else th.device(gpu)
        self.g_list = []

    def perturb(self, module):
        std, alpha, n, k = self.std, self.alpha, self.n, self.k 
        if any(p.grad is None for p in module.parameters()):
            my.perturb(module, std)
        else:
            if len(self.g_list) == k:
                self.g_list = self.g_list[1:]
            self.g_list.append(th.cat([p.grad.view(-1, 1) for p in module.parameters()]))
            u, _ = th.qr(th.cat(self.g_list, 1))
            if u.size(1) < k:
                u = th.cat([u, th.zeros(u.size(0), k - u.size(1), device=self.device)], 1)
            epsilon_n = th.randn(n, 1, device=self.device)
            epsilon_k = th.randn(k, 1, device=self.device)
            g = std * (alpha / n) ** 0.5 * epsilon_n + \
                    std * ((1 - alpha) / k) ** 0.5 * th.mm(u, epsilon_k)
            i = 0
            for p in module.parameters():
                p.data += g[i : i + p.grad.numel()].view(p.data.shape)
                i += p.data.numel()
