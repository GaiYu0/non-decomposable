import copy
import numpy as np
import numpy.random as npr
from matplotlib.mlab import PCA
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

def np_normalize(x, epsilon=1e-5):
    x = x - np.mean(x, 0, keepdims=True)
    x = x / (np.sqrt(np.mean(np.square(x), 0, keepdims=True)) + epsilon)
    return x

def th_normalize(x, epsilon=1e-5):
    x = x.float()
    x = x - th.mean(x, 0, keepdim=True)
    x = x / (th.sqrt(th.mean(x * x, 0, keepdim=True)) + epsilon)
    return x

def unbalanced_dataset(dataset, n_train=0, n_test=0, pca=False, minfrac=1e-2, D=None,
                       epsilon=1e-5, p=[], tensor=False, variable=False):
    def process(X, y, N):
        if p:
            select = lambda y, lower, upper: np.logical_and(lower <= y, y < upper)
            x_list = [X[select(y, p[i], p[i + 1])] for i in range(len(p) - 1)]
            Xy = [np.hstack((x, np.full((len(x), 1), i))) for i, x in enumerate(x_list)]
            Xy = np.vstack(Xy)
        else:
            Xy = np.hstack((X, y.reshape(-1, 1)))
        idx = np.arange(len(Xy))
        npr.shuffle(idx)
        if N > 0:
            idx = idx[:N]
        Xy = Xy[idx]
        X, y = Xy[:, :X.shape[1]], Xy[:, -1]
        if tensor:
            X, y = th.from_numpy(X).float(), th.from_numpy(y).long()
        if variable:
            X, y = Variable(X), Variable(y)
        return X, y

    d = getattr(datasets, dataset)(dataset, train=True)
    train_data = d.train_data.numpy()
    train_data = np.reshape(train_data, (train_data.shape[0], -1))
    train_labels = d.train_labels.numpy()

    if n_train > 0:
        train_data, train_labels = train_data[:n_train], train_labels[:n_train]
    if pca:
        pca = PCA(train_data, standardize=False)
        train_data = pca.project(train_data, minfrac)
    if D is not None:
        p = npr.randn(train_data.shape[1], D)
        train_data = np.dot(train_data, p)
    mean = np.mean(train_data, 0, keepdims=True)
    train_data = train_data - mean
    std = np.sqrt(np.mean(np.square(train_data), 0, keepdims=True)) + epsilon
    train_data = train_data / std
    train_data, train_labels = process(train_data, train_labels, n_train)

    d = getattr(datasets, dataset)(dataset, train=False)
    test_data = d.test_data.numpy()
    test_data = np.reshape(test_data, (test_data.shape[0], -1))
    test_labels = d.test_labels.numpy()

    if n_test > 0:
        test_data, test_labels = test_data[:n_test], test_labels[:n_test]
    if pca:
        test_data = pca.project(test_data, minfrac)
    if D is not None:
        test_data = np.dot(test_data, p)
    test_data = (test_data - mean) / std
    test_data, test_labels = process(test_data, test_labels, n_test)

    return train_data, train_labels, test_data, test_labels

def unbalanced_cifar10(n_train, n_test, shape=None, p=[], epsilon=1e-5):
    def process(X, y, N):
        if p:
            select = lambda y, lower, upper: np.logical_and(lower <= y, y < upper)
            x_list = [X[select(y, p[i], p[i + 1])] for i in range(len(p) - 1)]
            Xy = [np.hstack((x, np.full((len(x), 1), i))) for i, x in enumerate(x_list)]
            Xy = np.vstack(Xy)
        else:
            Xy = np.hstack((X, y.reshape(-1, 1)))
        idx = np.arange(len(Xy))
        npr.shuffle(idx)
        if N > 0:
            idx = idx[:N]
        Xy = Xy[idx]
        X, y = Xy[:, :X.shape[1]], Xy[:, -1]
        if shape:
            X = np.reshape(X, (N,) + shape)
        return X, y

    cifar10 = datasets.CIFAR10('CIFAR10/', train=True)
    train_data, train_labels = cifar10.train_data, np.array(cifar10.train_labels)
    train_data = np.reshape(train_data, (len(train_data), -1))
    mean = np.mean(train_data, 0, keepdims=True)
    train_data = train_data - mean
    std = np.std(train_data, 0, keepdims=True) + epsilon
    train_data = train_data / std
    train_data, train_labels = process(train_data, train_labels, n_train)

    cifar10 = datasets.CIFAR10('CIFAR10/', train=False)
    test_data, test_labels = cifar10.test_data, np.array(cifar10.test_labels)
    test_data = np.reshape(test_data, (len(test_data), -1))
    test_data = test_data - mean
    test_data = test_data / std
    test_data, test_labels = process(test_data, test_labels, n_test)
    return train_data, train_labels, test_data, test_labels

def shuffle(x):
    i = np.arange(len(x))
    npr.shuffle(i)
    return x[i]

def load_cifar100(partition=[], epsilon=1e-5):
    def process(x, y):
        p_list = [np.full(y.shape, p, y.dtype) for p in partition]
        for i, (m, n) in enumerate(zip(p_list[:-1], p_list[1:])):
            y[np.logical_and(m <= y, y < n)] = i
        return x, y

    d = datasets.CIFAR100('CIFAR100/', train=True)
    train_x, train_y = d.train_data, np.array(d.train_labels)
    train_x = np.reshape(train_x, (len(train_x), -1))
    mean = np.mean(train_x, 0, keepdims=True)
    train_x = train_x - mean
    std = np.std(train_x, 0, keepdims=True) + epsilon
    train_x = train_x / std
    train_x, train_y = process(train_x, train_y)

    d = datasets.CIFAR100('CIFAR100/', train=False)
    test_x, test_y = d.test_data, np.array(d.test_labels)
    test_x = np.reshape(test_x, (len(test_x), -1))
    test_x = test_x - mean
    test_x = test_x / std
    test_x, test_y = process(test_x, test_y)

    return train_x, train_y, test_x, test_y

def predict(classifier, x):
    return th.max(classifier(x), 1)[1]

def accuracy(y_bar, y):
    return th.sum(((y_bar - y) == 0).float()) / float(y.size()[0])

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
    return tp_ / (tp_ + fn_ + 1e-5)

def f_beta(y_bar, y, beta=1):
    p, r = precision(y_bar, y), recall(y_bar, y)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-5)

def onehot(y, D):
    variable = False
    if isinstance(y, Variable):
        variable = True
        y = y.data
    if y.dim() == 1:
        y = y.view(-1, 1)
    y_onehot = th.zeros(y.size()[0], D)
    is_cuda = y.is_cuda
    if y.is_cuda:
        y = y.cpu()
    y_onehot.scatter_(1, y, 1)
    y_onehot = y_onehot.cuda() if is_cuda else y_onehot
    y_onehot = Variable(y_onehot) if variable else y_onehot
    return y_onehot

def nd_precision(nd_y_bar, nd_y, D):
    nd_y_bar, nd_y = onehot(nd_y_bar, D), onehot(nd_y, D)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, D, 1), th.chunk(nd_y, D, 1)
    return sum(precision(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / D

def nd_recall(nd_y_bar, nd_y, D):
    nd_y_bar, nd_y = onehot(nd_y_bar, D), onehot(nd_y, D)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, D, 1), th.chunk(nd_y, D, 1)
    return sum(recall(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / D

def nd_f_beta(nd_y_bar, nd_y, D, beta=1):
    # macro averaging
    nd_y_bar, nd_y = onehot(nd_y_bar, D), onehot(nd_y, D)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, D, 1), th.chunk(nd_y, D, 1)
    p = sum(precision(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / D
    r = sum(recall(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / D
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-5)

def nd_curry(f, D):
    return lambda y_bar, y: f(y_bar, y, D)

def global_stats(module, loader, stats):
    is_cuda = next(module.parameters()).is_cuda
    y_bar_list, y_list = [], []
    for (x, y) in loader:
        if is_cuda:
            x, y = x.cuda(), y.cuda()   
        x, y = Variable(x), Variable(y)
        y_bar = predict(module, x)

        y_bar_list.append(y_bar)
        y_list.append(y)

    y_bar, y = th.cat(y_bar_list).view(-1, 1), th.cat(y_list).view(-1, 1)
    if callable (stats):
        return stats(y_bar, y)
    else:
        return tuple(s(y_bar, y) for s in stats)

def parallel_global_stats(module, loader, stats, devices):
    # assert next(module.parameters()).is_cuda
    m = nn.DataParallel(module, devices, next(module.parameters()).get_device())
    y_bar_list, y_list = [], []
    for (x, y) in loader:
        y_bar_list.append(th.max(m(Variable(x).cuda()), 1)[1])
        y_list.append(Variable(y.cuda()))

    y_bar, y = th.cat(y_bar_list).view(-1, 1), th.cat(y_list).view(-1, 1)
    if callable (stats):
        return stats(y_bar, y)
    else:
        return tuple(s(y_bar, y) for s in stats)

def perturb(module, std):
    module = copy.deepcopy(module)
    is_cuda = next(module.parameters()).is_cuda
    if is_cuda:
        device = next(module.parameters()).get_device()
    contextualize = lambda x: x.cuda(device) if is_cuda else x
    for p in module.parameters():
        p.data += contextualize(th.randn(p.data.size()) * std)
    return module

def sample_subset(x, y, size):
    idx = np.random.randint(0, len(x) - 1, size)
    x, y = th.from_numpy(x[idx]).float(), th.from_numpy(y[idx]).long()
    return x, y

class MLP(nn.Module):
    def __init__(self, D, nonlinear):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(D[i], D[i + 1])
                                      for i in range(len(D) - 1)])
        self.nonlinear = nonlinear
        self.expose = False

    def forward(self, x):
        if x.dim != 2:
            x = x.view(x.size()[0], -1)
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = self.nonlinear(x)
        return x

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
