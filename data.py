import numpy as np
import torch as th
import torch.utils as utils
from torchvision import datasets


def load_dataset(dataset, rbg=False):
    """
    Parameters
    ----------
    dataset : MNIST/CIFAR10/CIFAR100/covtype
    """
    if dataset == 'covtype':
        x, y = np.load('covtype/x.npy'), np.load('covtype/y.npy')
        x, y = th.from_numpy(x).float(), th.from_numpy(y) - 1

        mask_list = [y == i for i in th.unique(y)]
        x_list = list(map(x.__getitem__, mask_list))
        y_list = list(map(y.__getitem__, mask_list))
        n_list = [int(0.7 * len(z)) for z in x_list]

        trainx_list = [z[:n] for z, n in zip(x_list, n_list)]
        trainy_list = [z[:n] for z, n in zip(y_list, n_list)]

        testx_list = [z[n:] for z, n in zip(x_list, n_list)]
        testy_list = [z[n:] for z, n in zip(y_list, n_list)]

        train_x, train_y = th.cat(trainx_list), th.cat(trainy_list)
        test_x, test_y = th.cat(testx_list), th.cat(testy_list)

        return train_x, train_y, test_x, test_y

    def process(x, y):
        if dataset == 'MNIST':
            x = x.float()
            x = x if rbg else x.view(x.size(0), -1) 
        elif dataset.startswith('CIFAR'):
            x, y = th.from_numpy(x), th.tensor(y)
            x = x.float().contiguous()
            x = th.transpose(x, 1, 3) if rbg else x.view(x.size(0), -1)
        return x, y

    container = getattr(datasets, dataset)(dataset, train=True)
    train_x, train_y = process(container.train_data, container.train_labels)

    container = getattr(datasets, dataset)(dataset, train=False)
    test_x, test_y = process(container.test_data, container.test_labels)

    return train_x, train_y, test_x, test_y


def normalize(train_x, test_x, epsilon=1e-5):
    """
    Parameters
    ----------
    train_x, test_x : torch.Tensor
    """
    trainx_shape = train_x.shape
    train_x = train_x.view(train_x.size(0), -1)
    mean = th.mean(train_x, 0, keepdim=True)
    train_x = train_x - mean
    std = th.sqrt(th.mean(train_x * train_x, 0, keepdim=True)) + epsilon
    train_x = train_x / std
    train_x = train_x.view(trainx_shape)

    testx_shape = test_x.shape
    test_x = test_x.view(test_x.size(0), -1)
    test_x = (test_x - mean) / std
    test_x = test_x.view(testx_shape)

    return train_x, test_x


def random_subset(train_x, train_y, test_x, test_y, label2ratio):
    """
    Parameters
    ----------
    train_x, train_y, test_x, test_y: torch.Tensor
    """
    def process(x_tensor, y_tensor):
        x_list, y_list = [], []
        for y in th.unique(y_tensor):
            ratio = label2ratio.get(y.item(), 0)
            if ratio > 0:
                mask = y_tensor == y
                m = th.sum(mask)
                n = int(ratio * m.item())
                x_list.append(x_tensor[mask][th.randperm(m)[:n]])
                y_list.append(th.tensor([y.item()] * n, device=y.device))
        x_tensor, y_tensor = th.cat(x_list, 0), th.cat(y_list, 0)
        randperm = th.randperm(len(x_tensor))
        x_tensor, y_tensor = x_tensor[randperm], y_tensor[randperm]
        return x_tensor, y_tensor

    train_x, train_y = process(train_x, train_y)
    test_x, test_y = process(test_x, test_y)
    return train_x, train_y, test_x, test_y


def relabel(train_x, train_y, test_x, test_y, label2label):
    """
    Parameters
    ----------
    train_x, train_y, test_x, test_y: torch.Tensor
    """
    def process(x, y):
        y_bar = y.clone()
        for key, value in label2label.items():
            y_bar[y == key] = value
        y = y_bar
        return x, y

    train_x, train_y = process(train_x, train_y)
    test_x, test_y = process(test_x, test_y)
    return train_x, train_y, test_x, test_y


def BalancedDataLoader(x, y, batch_size, cuda, infinite=True):
    idx_list = [y == lbl for lbl in range(th.min(y), th.max(y) + 1)]
    ds_list = [utils.data.TensorDataset(x[idx], y[idx]) for idx in idx_list]
    bs_list = [int(batch_size * th.sum(idx) / len(x)) for idx in idx_list[:-1]]
    bs_list.append(batch_size - sum(bs_list))

    kwargs = {'shuffle' : True, 'drop_last' : True, 'num_workers' : 0}
    new_loader = lambda ds, bs: iter(utils.data.DataLoader(ds, bs, **kwargs))
    loader_list = [new_loader(ds, bs) for ds, bs in zip(ds_list, bs_list)]
    contextualize = lambda x, y: (x.cuda(), y.cuda()) if cuda else (x, y)
    while True:
        try:
            batch_list = [contextualize(*next(loader)) for loader in loader_list]
        except StopIteration:
            if infinite:
                loader_list = [new_loader(ds, bs) for ds, bs in zip(ds_list, bs_list)]
            else:
                raise StopIteration()
        x_tuple, y_tuple = tuple(zip(*batch_list))
        yield th.cat(x_tuple), th.cat(y_tuple)
