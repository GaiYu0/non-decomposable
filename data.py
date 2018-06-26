import numpy as np
import torch as th
import torch.utils as utils
from torchvision import datasets

def load_mnist(labelling={}, rbg=False, epsilon=1e-5):
    def process(x, y):
        if rbg:
            x = x.view((-1, 1, 28, 28))
        if labelling:
            y_bar = y.copy()
            for (m, n), label in labelling.items():
                y_bar[(m <= y) * (y < n)] = label
            y = y_bar
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


def load_cifar10(labelling={}, rbg=False, torch=False, epsilon=1e-5):
    def process(x, y):
        if rbg:
            x = x.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        if labelling:
            y_bar = y.copy()
            for (m, n), label in labelling.items():
                y_bar[(m <= y) * (y < n)] = label
            y = y_bar
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


def load_cifar100(labelling={}, rbg=False, torch=False, epsilon=1e-5):
    def process(x, y):
        if rbg:
            x = x.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        if labelling:
            y_bar = y.copy()
            for (m, n), label in labelling.items():
                y_bar[(m <= y) * (y < n)] = label
            y = y_bar
        return x, y

    d = datasets.CIFAR100('CIFAR100/', train=True)
    train_x, train_y = d.train_data, np.array(d.train_labels)
    train_x = np.reshape(train_x, (len(train_x), -1))
    mean = np.mean(train_x, 0, keepdims=True)
    std = np.std(train_x, 0, keepdims=True) + epsilon
    train_x = (train_x - mean) / std
    train_x, train_y = process(train_x, train_y)

    d = datasets.CIFAR100('CIFAR100/', train=False)
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

def BalancedDataLoader(x, y, batch_size, cuda):
    idx_list = [y == lbl for lbl in range(th.min(y), th.max(y) + 1)]
    ds_list = [utils.data.TensorDataset(x[idx], y[idx]) for idx in idx_list]
    bs_list = [int(batch_size * th.sum(idx) / len(x)) for idx in idx_list]

    new_loader = lambda ds, bs: iter(utils.data.DataLoader(ds, bs, shuffle=True))
    loader_list = [new_loader(ds, bs) for ds, bs in zip(ds_list, bs_list)]
    contextualize = lambda x, y: (x.cuda(), y.cuda()) if cuda else (x, y)
    while True:
        try:
            batch_list = [contextualize(*next(loader)) for loader in loader_list]
        except StopIteration:
            loader_list = [new_loader(ds, bs) for ds, bs in zip(ds_list, bs_list)]
        x_tuple, y_tuple = tuple(zip(*batch_list))
        yield th.cat(x_tuple), th.cat(y_tuple)
