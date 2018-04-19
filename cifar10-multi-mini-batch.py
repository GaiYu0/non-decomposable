
# coding: utf-8

# In[ ]:


from __future__ import print_function
import argparse
import copy
import itertools as it
import pickle
import time
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
import my


# In[ ]:


import math
import matplotlib.pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


cuda = True
if cuda:
    th.cuda.set_device(0)

th.random.manual_seed(1)
if cuda:
    th.cuda.manual_seed_all(1)


# In[ ]:


train_x, train_y, test_x, test_y = my.load_cifar10(rbg=True)

train_x = th.from_numpy(train_x).float()
train_y = th.from_numpy(train_y)
test_x = th.from_numpy(test_x).float()
test_y = th.from_numpy(test_y)

train_loader = DataLoader(TensorDataset(train_x, train_y), 1024 * 4, drop_last=False)
test_loader = DataLoader(TensorDataset(test_x, test_y), 1024 * 4, drop_last=False)

n_classes = int(train_y.max() - train_y.min() + 1)

dataset = TensorDataset(train_x, train_y)

def sample(sample_size, batch_size):
    dl = DataLoader(dataset, sample_size, shuffle=True)
    def predicate(z): # TODO
        i, (x, y) = z
    s = it.takewhile(lambda x: x[0] < batch_size, enumerate(dl))
    s = [(Variable(x), Variable(y)) for _, (x, y) in s]
    if cuda:
        s = [(x.cuda(), y.cuda()) for (x, y) in s]
    return s


# In[ ]:


class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 2, 1)
        self.linear = nn.Linear(8, n_classes)
    
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 8)
        x = self.linear(x.view(-1, 8))
        return x    
    
def forward(classifier, xy):
    x, y = xy
    y = my.onehot(y, n_classes)
    y_bar = F.softmax(classifier(x), 1)
    return th.cat((y, y_bar), 1).view(1, -1)

L = lambda c, loader: my.global_stats(c, loader, my.nd_curry(my.nd_f_beta, n_classes))

def L_mini_batch(c, loader):
    L_list = [my.nd_f_beta(my.predict(c, x), y, n_classes) for x, y in loader]
    return th.cat(L_list).view(-1, 1)


# In[ ]:


batch_size = 5
sample_size = 50

c = CNN(n_classes)
critic = my.RN(sample_size, 2 * n_classes, (32, 64), (64,) * 1 + (1,), F.relu)
if cuda:
    c.cuda()
    critic.cuda()
c_optim = Adam(c.parameters(), 1e-3)
critic_optim = Adam(critic.parameters(), 1e-3)

'initial f1: %f' % L(c, test_loader)


# In[ ]:


n_iterations = 100
n_perturbations = 50
actor_iterations = 10
critic_iterations = 10
std = 1e-1
tau = 1e-2

hist = []
for i in range(n_iterations):
    hist.append({})
#   hist[-1]['c_state_dict'] = copy.deepcopy(my.state_dict_gpu2cpu(c.state_dict()))

    if i % 20 == 0:
        s = sample(sample_size, batch_size)
#   hist[-1]['s'] = s

    L_c = L_mini_batch(c, s)
    c_bar_list = []
    L_bar_list = []
    t_list = []
    for j in range(n_perturbations):
        c_bar_list.append(my.perturb(c, std))
        L_bar = L_mini_batch(c_bar_list[-1], s)
        L_bar_list.append(L_bar)
        t_list.append(L_c - L_bar)
    w_list = th.cat([th.exp(t**2 / tau) for t in t_list], 1)
    w_list = th.chunk((w_list / th.sum(w_list, 1, keepdim=True)).detach(), n_perturbations, 1)

    hist[-1]['L_bar_list'] = L_bar_list
    hist[-1]['w_list'] = w_list

    y = th.cat([forward(c, xy) for xy in s], 0).detach()
    y_bar_list = [th.cat([forward(c_bar, xy) for xy in s], 0).detach()                   for c_bar in c_bar_list]
    for j in range(critic_iterations):
        for y_bar, t, w in zip(y_bar_list, t_list, w_list):
            delta = critic(y) - critic(y_bar)
            mse = th.sum(w * (t - delta) ** 2)
            critic_optim.zero_grad()
            mse.backward()
            critic_optim.step()

#     hist[-1]['critic_state_dict'] = copy.deepcopy(my.state_dict_gpu2cpu(critic.state_dict()))

    c_param = copy.deepcopy(tuple(c.parameters()))
    for j in range(actor_iterations):
        y_bar = th.cat([forward(c, xy) for xy in s], 0)
        objective = -th.mean(critic(y_bar))
        c_optim.zero_grad()
        objective.backward()
        c_optim.step()
        if any(float(th.max(th.abs(p - q))) > std for p, q in zip(c_param, c.parameters())):
            break

    if (i + 1) % 1 == 0:
        f1 = L(c, test_loader)
        hist[-1]['f1'] = float(f1)
        print('[iteration %d]f1: %f' % (i + 1, f1))


# In[ ]:


pl.plot(range(len(hist)), (-math.log(1 / n_perturbations),) * len(hist), 'r')
entropy_list = []
for i, h in enumerate(hist):
    w_list = th.cat(h['w_list'], 1)
    entropy_list.append(-th.sum(w_list * th.log(w_list), 1))
    pl.plot((i,) * batch_size, entropy_list[-1].data.cpu().numpy(), 'bx')

tolist = lambda x: th.cat(x, 1).data.cpu().numpy().flatten()

for i in range(batch_size):
    pl.figure()
    pl.plot(range(len(hist)), (-math.log(1 / n_perturbations),) * len(hist), 'r')
    pl.plot(range(len(hist)), tolist([e[i : i + 1].view(1, 1) for e in entropy_list]))


# In[ ]:


for i, h in enumerate(hist):
    pl.plot((i,) * batch_size * n_perturbations, tolist(h['L_bar_list']), 'bx')
for i in range(batch_size):
    pl.figure()
    for j, h in enumerate(hist):
        pl.plot((j,) * n_perturbations, tolist([L_bar[i : i + 1] for L_bar in h['L_bar_list']]), 'bx')


# In[ ]:


pl.plot(range(len(hist)), [h['f1'] for h in hist])

