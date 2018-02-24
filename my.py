import copy
import numpy as np
import numpy.random as npr
from matplotlib.mlab import PCA
import torch as th
from torch.autograd import Variable
import torch.nn as nn
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

def predict(classifier, X):
    return th.max(classifier(X), 1)[1]

def accuracy(y_bar, y):
    return th.sum(((y_bar - y) == 0).float()) / float(y.size()[0])

def tp(y_bar, y): # true positive
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
    nd_y_bar, nd_y = onehot(nd_y_bar, D), onehot(nd_y, D)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, D, 1), th.chunk(nd_y, D, 1)
    pr = ((precision(y_bar, y), recall(y_bar, y)) for y_bar, y in zip(nd_y_bar, nd_y))
    return sum((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-5) for p, r in pr) / D

def nd_curry(f, D):
    return lambda y_bar, y: f(y_bar, y, D)

def global_stats(module, loader, stats):
    is_cuda = next(module.parameters()).is_cuda
    y_bar_list, y_list = [], []
    for (X, y) in loader:
        if is_cuda:
            X, y = X.cuda(), y.cuda()   
        X, y = Variable(X), Variable(y)
        y_bar = predict(module, X)

        y_bar_list.append(y_bar)
        y_list.append(y)

    y_bar, y = th.cat(y_bar_list).view(-1, 1), th.cat(y_list).view(-1, 1)
    if callable (stats):
        return stats(y_bar, y)
    else:
        return tuple(s(y_bar, y) for s in stats)

def perturb(module, std):
    module = copy.deepcopy(module)
    contextualize = lambda x: x.cuda() if next(module.parameters()).is_cuda else x
    for p in module.parameters():
#       p.data += contextualize(th.rand(p.data.size()) * std - std / 2)
        p.data += contextualize(th.randn(p.data.size()) * std)
    return module

def sample_subset(X, y, size):
    idx = np.random.randint(0, len(X) - 1, size)
    X, y = th.from_numpy(X[idx]).float(), th.from_numpy(y[idx]).long()
    return X, y

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
