import numpy as np
import numpy.random as npr
from matplotlib.mlab import PCA
import torch as th
from torch.autograd import Variable
from torchvision import datasets

def np_normalize(X, epsilon=1e-5):
    X = X - np.mean(X, 0, keepdims=True)
    X = X / (np.sqrt(np.mean(np.square(X), 0, keepdims=True)) + epsilon)
    return X

def th_normalize(X, epsilon=1e-5):
    X = X - th.mean(X, 0, keepdim=True)
    X = X / (th.sqrt(th.mean(X * X, 0, keepdim=True)) + epsilon)
    return X

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
