import copy
import numpy as np
import numpy.random as npr
from matplotlib.mlab import PCA
import torch as th
from torch.autograd import Variable
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

def unbalanced_mnist(n_train=0, n_test=0, pca=False, minfrac=1e-2, D=None, epsilon=1e-5):
    def process(X, y, N):
        positive, negative = X[y == 0], X[y != 0]
        positive = np.hstack((positive, np.ones((len(positive), 1))))
        negative = np.hstack((negative, np.zeros((len(negative), 1))))
        Xy = np.vstack((positive, negative))
        idx = np.arange(len(Xy))
        npr.shuffle(idx)
        if N > 0:
            idx = idx[:N]
        Xy = Xy[idx]
        X, y = Xy[:, :X.shape[1]], Xy[:, -1]
        X, y = Variable(th.from_numpy(X).float()), Variable(th.from_numpy(y).long())
        return X, y

    MNIST = datasets.MNIST('MNIST/', train=True)
    train_data = MNIST.train_data.numpy().reshape((-1, 28 * 28))
    train_labels = MNIST.train_labels.numpy()
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

    MNIST = datasets.MNIST('MNIST/', train=False)
    test_data = MNIST.test_data.numpy().reshape((-1, 28 * 28))
    test_labels = MNIST.test_labels.numpy()
    if n_test > 0:
        test_data, test_labels = test_data[:n_test], test_labels[:n_test]
    if pca:
        test_data = pca.project(test_data, minfrac)
    if D is not None:
        test_data = np.dot(test_data, p)
    test_data = (test_data - mean) / std
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
    return tp_ / (tp_ + fp_ + 1) # TODO

def recall(y_bar, y):
    tp_, fn_ = tp(y_bar, y), fn(y_bar, y)
    return tp_ / (tp_ + fn_ + 1)

def f_beta(y_bar, y, beta=1):
    p, r = precision(y_bar, y), recall(y_bar, y)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1)

def onehot(y, D):
    if isinstance(y, Variable):
        y = y.data
    y_onehot = th.zeros(y.size()[0], D)
    y_onehot.scatter_(1, y, 1)
    return Variable(y_onehot)

def nd_precision(nd_y_bar, nd_y, D):
    nd_y_bar, nd_y = onehot(nd_y_bar, D), onehot(nd_y, D)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, D, 1), th.chunk(nd_y, D, 1)
    return sum(precision(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / D

def nd_recall(nd_y_bar, nd_y, D):
    nd_y_bar, nd_y = onehot(nd_y_bar, D), onehot(nd_y, D)
    nd_y_bar, nd_y = th.chunk(nd_y_bar, D, 1), th.chunk(nd_y, D, 1)
    return sum(recall(y_bar, y) for y_bar, y in zip(nd_y_bar, nd_y)) / D

def nd_f_beta(y_bar, y, D, beta=1):
    p, r = nd_precision(y_bar, y, D), nd_recall(y_bar, y, D)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1)

def nd_curry(f, D):
    return lambda y_bar, y: f(y_bar, y, D)

def global_stats(module, loader, stats):
    y_bar_list, y_list = [], []
    for (X, y) in loader:
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
    for p in module.parameters():
        p.data += th.randn(p.data.size()) * std
    return module

def sample(X, y, size):
    if isinstance(X, Variable):
        X = X.data
    if isinstance(y, Variable):
        y = y.data
    X, y = X.numpy(), y.numpy()
    idx = np.random.randint(0, len(X) - 1, size)
    X, y = Variable(th.from_numpy(X[idx])), Variable(th.from_numpy(y[idx]))
    return X, y
