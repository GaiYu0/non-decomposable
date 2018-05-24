
# coding: utf-8

# In[ ]:


import copy
import collections
import math
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import my
import lenet
import resnet


# In[ ]:


class Args:
    pass
args = Args()

args.n_iterations_critic = 100
args.iw = 'quadratic' # None/linear/quadratic
args.gpu = 0
args.n_iterations = 5000
args.n_perturbations = 25
args.batch_size = 50
args.std = 1e-1
args.tau = 1e-1

verbose = None


# In[ ]:


cuda = args.gpu >= 0
if cuda:
    th.cuda.set_device(args.gpu)

th.random.manual_seed(1)
if cuda:
    th.cuda.manual_seed_all(1)

train_x, train_y, test_x, test_y = my.load_cifar10(rbg=True, torch=True)
# train_x, train_y, test_x, test_y = my.load_cifar10(rbg=False, torch=True)

test_loader = DataLoader(TensorDataset(test_x, test_y), 4096, drop_last=False)

n_classes = int(train_y.max() - train_y.min() + 1)


# In[ ]:


def TrainLoader():
    train_loader = iter(DataLoader(TensorDataset(train_x, train_y), args.batch_size, shuffle=True))
    contextualize = lambda x, y: (x.cuda(), y.cuda()) if cuda else (x, y)
    while True:
        try:
            yield contextualize(*next(train_loader))
        except StopIteration:
            train_loader = iter(DataLoader(TensorDataset(train_x, train_y), args.batch_size, shuffle=True))
            yield contextualize(*next(train_loader))

train_loader = TrainLoader()

def forward(c, xy):
    x, y = xy
    y = my.onehot(y, n_classes)
    y_bar = F.softmax(c(x), 1)
    return th.cat((y, y_bar), 1).view(1, -1)

def L_mini_batch(c, xy):
    x, y = xy
    return my.nd_f_beta(th.max(c(x), 1)[1], y, n_classes).view(1, 1)

def global_stats(c, loader):
    curry = lambda stat: lambda y_bar, y: stat(y_bar, y, n_classes)
    stats = (my.accuracy,) + tuple(map(curry, (my.nd_precision, my.nd_recall, my.nd_f_beta)))
    keys = ('accuracy', 'precision', 'recall', 'f1')
    values = [value.item() for value in my.global_stats(c, loader, stats)]
    return collections.OrderedDict(zip(keys, values))


# In[ ]:


# c = my.MLP((3072, n_classes), F.relu)
# c = my.MLP((3072,) + (1024,) + (n_classes,), F.relu)
# c = my.MLP((3072,) + (1024,) * 2 + (n_classes,), F.relu)
# c = my.MLP((3072,) + (1024,) * 3 + (n_classes,), F.relu)
c = lenet.LeNet(3, n_classes)
# c = resnet.ResNet(depth=18, n_classes=n_classes)

critic = my.RN(args.batch_size, 2 * n_classes, tuple(), (4 * n_classes, 64, 256), (256, 64) + (1,), F.relu, triu=True)

if cuda:
    c.cuda()
    critic.cuda()

c_optim = Adam(c.parameters(), eps=1e-3)
critic_optim = Adam(critic.parameters())

for key, value in global_stats(c, test_loader).items():
    print(key, value)


# In[ ]:


hist = []
phi = (lambda x: x) if args.iw == 'linear' else lambda x: x ** 2 if args.iw == 'quadratic' else lambda _: 0

for i in range(args.n_iterations):
    hist.append({})
#     hist[-1]['critic_state_dict'] = copy.deepcopy(my.state_dict_gpu2cpu(critic.state_dict()))        
#     hist[-1]['critic_optim_state_dict'] = my.optim_state_dict_gpu2cpu(critic_optim.state_dict())
#     hist[-1]['c_state_dict'] = copy.deepcopy(my.state_dict_gpu2cpu(c.state_dict()))
#     hist[-1]['c_optim_state_dict'] = my.optim_state_dict_gpu2cpu(c_optim.state_dict())

    if verbose == 0:
        t0 = time.time()

    mini_batch = next(train_loader)
    L_c = L_mini_batch(c, mini_batch)
    c_bar_list = []
    L_bar_list = []
    t_list = []
    for j in range(args.n_perturbations):
        c_bar_list.append(my.perturb(c, args.std))
        L_bar = L_mini_batch(c_bar_list[-1], mini_batch)
        L_bar_list.append(L_bar)
        t_list.append(L_c - L_bar)
    w_list = th.cat([th.exp(phi(t) / args.tau) for t in t_list], 1)
    w_list = th.chunk((w_list / th.sum(w_list, 1, keepdim=True)).detach(), args.n_perturbations, 1)

    hist[-1]['L_bar_tensor'] = th.cat(L_bar_list, 0)
    hist[-1]['w_tensor'] = th.cat(w_list, 0)

    if verbose == 0:
        t1 = time.time()
        print('[iteration %d]t1 - t0: %f' % (i + 1, t1 - t0))
            
    y = forward(c, mini_batch).detach()
    y_bar_list = [forward(c_bar, mini_batch).detach() for c_bar in c_bar_list]
    for j in range(args.n_iterations_critic):
        for y_bar, t, w in zip(y_bar_list, t_list, w_list):
            delta = critic(y) - critic(y_bar)
            mse = th.sum(w * (t - delta) ** 2)
            critic_optim.zero_grad()
            mse.backward()
            critic_optim.step()
#     assert not my.module_isnan(critic)

    if verbose == 0:
        t2 = time.time()
        print('[iteration %d]t2 - t1: %f' % (i + 1, t2 - t1))

    y_bar = forward(c, mini_batch)
    objective = -th.mean(critic(y_bar))
    c_optim.zero_grad()
    objective.backward()
    c_optim.step()
#     assert not my.module_isnan(c)

    if verbose == 0:
        t3 = time.time()
        print('[iteration %d]t3 - t2: %f' % (i + 1, t3 - t2))

    hist[-1]['stats'] = global_stats(c, test_loader)
    
    if (i + 1) % 1 == 0:
        print('[iteration %d]f1: %f' % (i + 1, hist[-1]['stats']['f1']))


# In[ ]:


import pickle
fields = [f for f in dir(args) if '__' not in f]
values = [getattr(args, f) for f in fields]
path = 'hist/' + '-'.join('%s-%s' % (f, v) for f, v in zip(fields, values))
pickle.dump(hist, open(path, 'wb'))

