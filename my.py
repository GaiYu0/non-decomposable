import copy
import itertools
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets


def load_mnist(partition=[], rbg=False, epsilon=1e-5):
    def process(x, y):
        if rbg:
            x = x.view((-1, 1, 28, 28))
        if partition:
            for i, (m, n) in enumerate(zip(partition[:-1], partition[1:])):
                y[(m <= y) * (y < n)] = i
        return x, y

    d = datasets.MNIST('MNIST/', train=True)
    train_x, train_y = d.train_data.float(), d.train_labels
    train_x = train_x.view(-1, 28 * 28)
    mean = th.mean(train_x, 0, keepdim=True)
    std = th.std(train_x, 0, keepdim=True) + epsilon
    train_x = (train_x - mean) / std
    train_x, train_y = process(train_x, train_y)

    d = datasets.MNIST('MNIST/', train=False)
    test_x, test_y = d.test_data.float(), d.test_labels
    test_x = test_x.view(-1, 28 * 28)
    test_x = (test_x - mean) / std
    test_x, test_y = process(test_x, test_y)

    return train_x, train_y, test_x, test_y


def load_cifar10(partition=[], rbg=False, torch=False, epsilon=1e-5):
    def process(x, y):
        if rbg:
            x = x.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        if partition:
            p_list = [np.full(y.shape, p, y.dtype) for p in partition]
            for i, (m, n) in enumerate(zip(p_list[:-1], p_list[1:])):
                y[np.logical_and(m <= y, y < n)] = i
        return x, y

    d = datasets.CIFAR10('CIFAR10/', train=True)
    train_x, train_y = d.train_data, np.array(d.train_labels)
    train_x = np.reshape(train_x, (len(train_x), -1))
    mean = np.mean(train_x, 0, keepdims=True)
    std = np.std(train_x, 0, keepdims=True) + epsilon
    train_x = (train_x - mean) / std
    train_x, train_y = process(train_x, train_y)

    d = datasets.CIFAR10('CIFAR10/', train=False)
    test_x, test_y = d.test_data, np.array(d.test_labels)
    test_x = np.reshape(test_x, (len(test_x), -1))
    test_x = (test_x - mean) / std
    test_x, test_y = process(test_x, test_y)

    if torch:
        train_x = th.from_numpy(train_x).float()
        train_y = th.from_numpy(train_y)
        test_x = th.from_numpy(test_x).float()
        test_y = th.from_numpy(test_y)

    return train_x, train_y, test_x, test_y


def accuracy(y_bar, y):
    return th.sum(((y_bar - y) == 0).float()) / float(y.size(0))


def tp(y_bar, y): # true positive (labeled as 1)
    return th.sum((y_bar * y).float())


def fp(y_bar, y): # false positive
    return th.sum((y_bar * (1 - y)).float())


def fn(y_bar, y): # false negative
    return th.sum(((1 - y_bar) * y).float())


def precision(y_bar, y):
    tp_, fp_ = tp(y_bar, y), fp(y_bar, y)
    return tp_ / (tp_ + fp_ + 1e-5) # TODO


def recall(y_bar, y):
    tp_, fn_ = tp(y_bar, y), fn(y_bar, y)
    return tp_ / (tp_ + fn_ + 1e-5) # TODO


def f_beta(y_bar, y, beta=1):
    p, r = precision(y_bar, y), recall(y_bar, y)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-5) # TODO


def onehot(x, d):
    """
    Parameters
    ----------
    x : (n,) or (n, 1)
    """

    x = x.unsqueeze(1) if x.dim() == 1 else x
    ret = th.zeros(x.size(0), d)
    is_cuda = x.is_cuda
    x = x.cpu()
    ret.scatter_(1, x, 1)
    return ret.cuda() if is_cuda else ret


def nd_precision(nd_y_bar, nd_y, d):
    nd_y_bar, nd_y = onehot(nd_y_bar, d), onehot(nd_y, d)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, d, 1), th.chunk(nd_y, d, 1)
    return sum(precision(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / d


def nd_recall(nd_y_bar, nd_y, d):
    nd_y_bar, nd_y = onehot(nd_y_bar, d), onehot(nd_y, d)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, d, 1), th.chunk(nd_y, d, 1)
    return sum(recall(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / d


def nd_f_beta(nd_y_bar, nd_y, d, beta=1):
    # macro averaging
    nd_y_bar, nd_y = onehot(nd_y_bar, d), onehot(nd_y, d)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, d, 1), th.chunk(nd_y, d, 1)
    p = sum(precision(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / d
    r = sum(recall(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / d
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-5) # TODO


def global_stats(module, loader, stats):
    y_bar_list, y_list = [], []
    for x, y in loader:
        if next(module.parameters()).is_cuda:
            x, y = x.cuda(), y.cuda()   
        y_bar_list.append(th.max(module(x), 1)[1])
        y_list.append(y)

    y_bar, y = th.cat(y_bar_list).view(-1, 1), th.cat(y_list).view(-1, 1)
    if callable (stats):
        return stats(y_bar, y)
    else:
        return [s(y_bar, y) for s in stats]


def parallel_global_stats(module, loader, stats, devices):
    # assert next(module.parameters()).is_cuda
    m = nn.DataParallel(module, devices, next(module.parameters()).get_device())
    y_bar_list, y_list = [], []
    for (x, y) in loader:
        y_bar_list.append(th.max(m(x.cuda()), 1)[1])
        y_list.append(y.cuda())

    y_bar, y = th.cat(y_bar_list).view(-1, 1), th.cat(y_list).view(-1, 1)
    if callable (stats):
        return stats(y_bar, y)
    else:
        return tuple(s(y_bar, y) for s in stats)


def perturb(module, std):
    module = copy.deepcopy(module)
    p = next(module.parameters())
    device = p.get_device() if p.is_cuda else None
    for p in module.parameters():
        p.data += th.randn(*p.size(), device=device) * std
    return module


def sample(dataset, sample_size, batch_size, cuda):
    dl = DataLoader(dataset, sample_size, shuffle=True)
    s = itertools.takewhile(lambda x: x[0] < batch_size, enumerate(dl))
    if cuda:
        s = [(x.cuda(), y.cuda()) for _, (x, y) in s]
    return s


class MLP(nn.Module):
    def __init__(self, d, nonlinear):
        super(MLP, self).__init__()
        self.linear_list = nn.ModuleList([nn.Linear(d[i], d[i + 1])
                                          for i in range(len(d) - 1)])
        self.nonlinear = nonlinear

    def forward(self, x):
        for linear in self.linear_list[:-1]:
            x = self.nonlinear(linear(x))
        return self.linear_list[-1](x)


class RN(nn.Module):
    def __init__(self, n_objects, n_features, unary, binary, mlp, nonlinear, triu):
        super(RN, self).__init__()
        self.n_objects = n_objects
        self.n_features = n_features

        if unary:
            assert unary[0] == n_features
            assert unary[-1] * 2 == binary[0]
        else:
            assert binary[0] == n_features * 2
        assert binary[-1] == mlp[0]

        self.unary = MLP(unary, nonlinear) if unary else None
        self.binary = MLP(binary, nonlinear)
        self.mlp = MLP(mlp, nonlinear)
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
        x : (n, self.n_objects * self.n_features)
        """

        # TODO correctness

        n, _ = x.size()
        if self.unary:
            x = x.view(n * self.n_objects, self.n_features)
            x = self.nonlinear(self.unary(x))
        x = x.view(n, self.n_objects, -1)
        u = x.unsqueeze(1).repeat(1, self.n_objects, 1, 1)
        v = x.unsqueeze(2).repeat(1, 1, self.n_objects, 1)
        x = th.cat((u, v), 3).view(n * self.n_objects * self.n_objects, -1)
        x = self.nonlinear(self.binary(x))
        x = x.view(n, self.n_objects * self.n_objects, -1)
        x = th.mean(x, 1) if self.mask is None else th.mean(x * self.mask, 1)
        x = self.mlp(x)
        return x


'''
class RN(nn.Module):
    def __init__(self, n_objects, n_features, d_conv2d, d_linear, nonlinear):
        super(RN, self).__init__()
        self.n_objects = n_objects
        self.n_features = n_features
        self.d_conv2d = d_conv2d
        self.d_linear = d_linear
        self.d_linear = d_linear
        self.nonlinear = nonlinear

        self.conv2d_w0 = nn.Parameter(th.randn(d_conv2d[0], n_features, 2, 1))
        self.conv2d_b0 = nn.Parameter(th.zeros(d_conv2d[0]))
        self.conv2d_w = nn.ParameterList([nn.Parameter(th.randn(m, n, 1, 1)) \
                                          for m, n in zip(d_conv2d[1:], d_conv2d[:-1])])
        self.conv2d_b = nn.ParameterList([nn.Parameter(th.zeros(d)) for d in d_conv2d[1:]])
        self.mlp = MLP((d_conv2d[-1],) + d_linear, nonlinear)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (N, N_OBJECTS * N_FEATURES)
        """

        N, d_linear = x.size()
        x = x.view(N, self.n_objects, self.n_features, 1).transpose(1, 2)
        # TODO reverse iteration order
        ij_list = []
        for d in range(1, self.n_objects):
            ij = self.nonlinear(F.conv2d(x, self.conv2d_w0, self.conv2d_b0, dilation=d))
            for w, b in zip(self.conv2d_w, self.conv2d_b):
                ij = self.nonlinear(F.conv2d(ij, w, b))
            ij = F.avg_pool2d(ij, (self.n_objects - d, 1))
            ij_list.append(ij.view(N, self.d_linear[0]))
        x = sum(ij_list) / len(ij_list)
        return self.mlp(x)
'''


'''
class RN(nn.Module):
    def __init__(self, n_objects, n_features, D, nonlinear):
        super(RN, self).__init__()
        self.n_objects = n_objects
        self.n_features = n_features
        self.D = D
        self.nonlinear = nonlinear

        self.conv2d_w = nn.Parameter(th.randn(D[0], n_features, 2, 1))
        self.conv2d_b = nn.Parameter(th.zeros(D[0]))
        self.mlp = MLP(D, nonlinear)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (N, N_OBJECTS * N_FEATURES)
        """

        N, D = x.size()
        x = x.view(N, self.n_objects, self.n_features, 1).transpose(1, 2)
        # TODO reverse iteration order
        ij_list = []
        for d in range(1, self.n_objects):
            ij = self.nonlinear(F.conv2d(x, self.conv2d_w, self.conv2d_b, dilation=d))
            ij_list.append(F.avg_pool2d(ij, (self.n_objects - d, 1)).view(N, self.D[0]))
        x = sum(ij_list) / len(ij_list)
        return self.mlp(x)
'''


def isnan(x):
    return int(th.sum((x != x).long())) > 0


def module_isnan(module):
    return any(isnan(x) for x in module.parameters())


state_dict_cpu2gpu = lambda state_dict: {key : value.cuda() \
                                         for key, value in state_dict.items()}


state_dict_gpu2cpu = lambda state_dict: {key : value.cpu() \
                                         for key, value in state_dict.items()}


def optim_state_dict_gpu2cpu(gpu):
    cpu = {'param_groups' : gpu['param_groups']}
    to_cpu = lambda x: x.cpu() if th.is_tensor(x) else x
    cpu['state'] = {key : {k : to_cpu(v) for k, v in value.items()}
                    for key, value in gpu['state'].items()}
    return cpu


def optim_state_dict_cpu2gpu(cpu):
    gpu = {'param_groups' : cpu['param_groups']}
    to_gpu = lambda x: x.cuda() if th.is_tensor(x) else x
    gpu['state'] = {key : {k : to_gpu(v) for k, v in value.items()}
                    for key, value in cpu['state'].items()}
    return gpu
