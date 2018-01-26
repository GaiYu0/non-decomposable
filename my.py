import numpy as np
import torch as th

def np_normalize(X, epsilon=1e-5):
    X = X - np.mean(X, 0, keepdims=True)
    X = X / (np.sqrt(np.mean(np.square(X), 0, keepdims=True)) + epsilon)
    return X

def th_normalize(X, epsilon=1e-5):
    X = X - th.mean(X, 0, keepdim=True)
    X = X / (th.sqrt(th.mean(X * X, 0, keepdim=True)) + epsilon)
    return X
