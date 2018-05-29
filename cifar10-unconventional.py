
# coding: utf-8

# In[ ]:


import argparse
import collections
import copy
import itertools
import time
import numpy as np
import sklearn.metrics as metrics
import tensorboardX as tb
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import data
import my
import lenet
import resnet
import rn


# In[ ]:


'''
args = argparse.Namespace()
args.c_n_iterations = 25
args.c_n_batches = 8
args.critic_n_batches = 8
args.critic_n_iterations = 25
args.gpu = 3
args.n_iterations = 500
args.n_perturbations = 100
args.batch_size = 50
args.radius = 1e-1
args.std = 1e-1
args.tau = 1e-1
'''

parser = argparse.ArgumentParser()
parser.add_argument('--c-n-iterations', type=int, default=25)
parser.add_argument('--c-n-batches', type=int, default=8)
parser.add_argument('--critic-n-batches', type=int, default=8)
parser.add_argument('--critic-n-iterations', type=int, default=25)
parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--n-iterations', type=int, default=500)
parser.add_argument('--n-perturbations', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--radius', type=float, default=1e-1)
parser.add_argument('--std', type=float, default=1e-1)
parser.add_argument('--tau', type=float, default=1e-1)
args = parser.parse_args()

verbose = None

keys = sorted(vars(args).keys())
run_id = 'unconventional-' + '-'.join('%s-%s' % (key, str(getattr(args, key))) for key in keys)
writer = tb.SummaryWriter('runs/' + run_id)


# In[ ]:


train_x, train_y, test_x, test_y = data.load_cifar10(rbg=True, torch=True)
# train_x, train_y, test_x, test_y = my.load_cifar10(rbg=False, torch=True)

train_set = utils.data.TensorDataset(train_x, train_y)
test_set = utils.data.TensorDataset(test_x, test_y)
test_loader = utils.data.DataLoader(test_set, 4096, drop_last=False)

n_classes = int(train_y.max() - train_y.min() + 1)


# In[ ]:


def forward(c, batch):
    x, y = batch
    y = my.onehot(y, n_classes)
    y_bar = F.softmax(c(x), 1)
    return th.cat((y, y_bar), 1).view(1, -1)

def global_scores(c, loader):
    keys = ('accuracy', 'precision', 'recall', 'f1')
    scores = (
        metrics.accuracy_score,
        lambda y, y_bar: metrics.precision_score(y, y_bar, average='micro'),
        lambda y, y_bar: metrics.recall_score(y, y_bar, average='micro'),
        lambda y, y_bar: metrics.f1_score(y, y_bar, average='micro'),
    )
    values = [value.item() for value in my.global_scores(c, loader, scores)]
    return collections.OrderedDict(zip(keys, values))

def L_batches(c, batches):
    L = [[metrics.f1_score(th.max(c(x), 1)[1], y, average='micro').item()] for x, y in batches]
    return new_tensor(L)

def log_statistics(tag, tensor, global_step):
    writer.add_scalar(tag + '-min', th.min(tensor), global_step)
    writer.add_scalar(tag + '-max', th.max(tensor), global_step)
    writer.add_scalar(tag + '-mean', th.mean(tensor), global_step)
    writer.add_scalar(tag + '-std', th.std(tensor), global_step)


# In[ ]:


if args.gpu < 0:
    cuda = False
    new_tensor = th.FloatTensor
else:
    cuda = True
    new_tensor = th.cuda.FloatTensor
    th.cuda.set_device(args.gpu)
    th.cuda.manual_seed_all(1)

th.random.manual_seed(1)

# c = mlp.MLP((3072, n_classes), F.relu)
# c = mlp.MLP((3072,) + (1024,) + (n_classes,), F.relu)
# c = mlp.MLP((3072,) + (1024,) * 2 + (n_classes,), F.relu)
# c = mlp.MLP((3072,) + (1024,) * 3 + (n_classes,), F.relu)
# c = lenet.LeNet(3, n_classes)
c = resnet.ResNet(depth=18, n_classes=n_classes)

critic = rn.RN(args.batch_size, 2 * n_classes, tuple(), (4 * n_classes, 64, 256), (256, 64) + (1,), F.relu, triu=True)

if cuda:
    c.cuda()
    critic.cuda()

c_optim = optim.Adam(c.parameters(), eps=1e-3)
critic_optim = optim.Adam(critic.parameters())

for key, value in global_scores(c, test_loader).items():
    print(key, value)


# In[ ]:


# TODO L_bar_tensor?
# TODO sampling schedule for `critic_batches`


# In[ ]:


hist = []
critic_batches = my.sample_batches(train_set, args.batch_size, args.critic_n_batches, cuda)
for i in range(args.n_iterations):
    hist.append({})

    if verbose == 0:
        t0 = time.time()

    my.set_requires_grad(c, False)
    L_c = L_batches(c, critic_batches)
    c_bar_list, L_bar_list, t_list = [], [], []
    for j in range(args.n_perturbations):
        c_bar = copy.deepcopy(c)
        my.set_requires_grad(c_bar, False)
        c_bar_list.append(my.perturb(c_bar, args.std))
        L_bar_list.append(L_batches(c_bar_list[-1], critic_batches))
        t_list.append(L_c - L_bar_list[-1])
    w_tensor = th.exp(th.cat(t_list, 1) ** 2 / args.tau)
    w_tensor /= th.sum(w_tensor, 1, keepdim=True)
    w_list = th.chunk(w_tensor, w_tensor.size(1), 1)

    L_bar_tensor = th.cat(L_bar_list, 1)
    e_tensor = -th.sum(w_tensor * th.log(w_tensor), 1)

    log_statistics('L_c', L_c, i)
    log_statistics('L_bar', L_bar_tensor, i)
    log_statistics('entropy', e_tensor, i)

    if verbose == 0:
        t1 = time.time()
        print('[iteration %d]t1 - t0: %f' % (i + 1, t1 - t0))
    
    y = th.cat([forward(c, batch) for batch in critic_batches], 0).detach()
    y_bar_list = [th.cat([forward(c_bar, batch) for batch in critic_batches], 0) for c_bar in c_bar_list]
    for j in range(args.critic_n_iterations):
        for y_bar, t, w in zip(y_bar_list, t_list, w_list):
            delta = critic(y) - critic(y_bar)
            mse = th.sum(w * (t - delta) ** 2)
            critic_optim.zero_grad()
            mse.backward()
            critic_optim.step()

    if verbose == 0:
        t2 = time.time()
        print('[iteration %d]t2 - t1: %f' % (i + 1, t2 - t1))
    
    my.set_requires_grad(c, True)
    c_parameters = copy.deepcopy(tuple(c.parameters()))
    for j in range(args.c_n_iterations):
        batches = critic_batches + my.sample_batches(train_set, args.batch_size, args.c_n_batches, cuda)
        y_bar = th.cat([forward(c, batch) for batch in batches], 0)
        objective = -th.mean(critic(y_bar))
        c_optim.zero_grad()
        objective.backward()
        c_optim.step()
        if any(float(th.max(th.abs(p - q))) > args.radius for p, q in zip(c_parameters, c.parameters())):
            break

    if verbose == 0:
        t3 = time.time()
        print('[iteration %d]t3 - t2: %f' % (i + 1, t3 - t2))
    
    if th.min(L_c) > 0.8 and th.mean(L_c) > 0.9:
        critic_batches = my.sample_batches(train_set, args.batch_size, args.critic_n_batches, cuda)

#     f1 = th.mean(L_batches(c, critic_batches))

    hist[-1]['stats'] = global_scores(c, test_loader)
    for key, value in hist[-1]['stats'].items():
        writer.add_scalar(key, value, i)
    if (i + 1) % 1 == 0:
        print('[iteration %d]%f %f %f' % (i + 1, hist[-1]['stats']['f1'], th.min(L_bar_tensor), th.max(L_bar_tensor)))

