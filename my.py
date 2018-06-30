import collections
import itertools
import torch as th
import torch.utils as utils


def copy(m, n):
    for p, q in zip(m.parameters(), n.parameters()):
        q[:] = p


def get_requires_grad(module):
    return [p.requires_grad for p in module.parameters()]


def set_requires_grad(module, requires_grad):
    if isinstance(requires_grad, bool):
        for p in module.parameters():
            p.requires_grad = requires_grad
    else:
        for i, p in enumerate(module.parameters()):
            p.requires_grad = requires_grad[i]


def global_scores(module, loader, scores):
    requires_grad = get_requires_grad(module)
    set_requires_grad(module, False)

    y_bar_list, y_list = [], []
    for x, y in loader:
        if next(module.parameters()).is_cuda:
            x, y = x.cuda(), y.cuda()   
        y_bar = th.max(module(x), 1)[1]
        y_bar_list.append(y_bar.detach())
        y_list.append(y)

    set_requires_grad(module, requires_grad)

    y, y_bar = th.cat(y_list), th.cat(y_bar_list)
    if callable (scores):
        return scores(y_bar, y)
    else:
        return [s(y_bar, y) for s in scores]


def onehot(x, d):
    """
    Parameters
    ----------
    x : (n,) or (n, 1)
    """

    if x.dim() == 1:
        x = x.unsqueeze(1)
    z = th.zeros(x.size(0), d)
    is_cuda = x.is_cuda
    x = x.cpu()
    z.scatter_(1, x, 1)
    return z.cuda() if is_cuda else z


def parse_report(report):
    def parse_line(line):
        keys = ('precision', 'recall', 'f1', 'support')
        p, r, f1, s = line.replace(' / ', '').split()[1:]
        p, r, f1, s = float(p), float(r), float(f1), int(s)
        return collections.OrderedDict(zip(keys, (p, r, f1, s)))
    line_list = report.split('\n')
    return tuple(map(parse_line, line_list[2 : -3])), parse_line(line_list[-2])


def perturb(module, std):
    p = next(module.parameters())
    device = p.get_device() if p.is_cuda else None
    for p in module.parameters():
        p += th.randn(*p.shape, device=device) * std
    return module


def isnan(x):
    return int(th.sum((x != x).long())) > 0


def module_isnan(module):
    return any(isnan(x) for x in module.parameters())
