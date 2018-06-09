
# coding: utf-8

# In[ ]:


import argparse
import copy
import collections
import pickle
import time
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
args.n_iterations_critic = None
# args.iw = 'none'
# args.iw = 'sqrt'
# args.iw = 'linear'
args.iw = 'quadratic'
args.gpu = None
args.n_iterations = None
args.n_perturbations = None
args.batch_size_c = None
args.batch_size_critic = None
args.std = None
args.tau = None
args.topk = 0
'''

parser = argparse.ArgumentParser()
parser.add_argument('--n-iterations-critic', type=int, default=None)
parser.add_argument('--iw', type=str, default=None)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--n-iterations', type=int, default=None)
parser.add_argument('--n-perturbations', type=int, default=None)
parser.add_argument('--batch-size-c', type=int, default=None)
parser.add_argument('--batch-size-critic', type=int, default=None)
parser.add_argument('--std', type=float, default=None)
parser.add_argument('--tau', type=float, default=None)
parser.add_argument('--topk', type=int, default=0)
args = parser.parse_args()

verbose = None

keys = sorted(vars(args).keys())
run_id = 'parameter-' + '-'.join('%s-%s' % (key, str(getattr(args, key))) for key in keys)
writer = tb.SummaryWriter('runs/' + run_id)


# In[ ]:


if args.gpu < 0:
    cuda = False
    new_tensor = th.FloatTensor
else:
    cuda = True
    new_tensor = th.cuda.FloatTensor
    th.cuda.set_device(args.gpu)

train_x, train_y, test_x, test_y = data.load_cifar10(rbg=True, torch=True)
# train_x, train_y, test_x, test_y = data.load_cifar10(rbg=False, torch=True)

train_set = utils.data.TensorDataset(train_x, train_y)
train_loader = utils.data.DataLoader(train_set, 4096, drop_last=False)
test_set = utils.data.TensorDataset(test_x, test_y)
test_loader = utils.data.DataLoader(test_set, 4096, drop_last=False)

def TrainLoader():
    dataset = utils.data.TensorDataset(train_x, train_y)
    new_loader = lambda: iter(utils.data.DataLoader(dataset, args.batch_size_c, shuffle=True))
    contextualize = lambda x, y: (x.cuda(), y.cuda()) if cuda else (x, y)
    while True:
        try:
            yield contextualize(*next(loader))
        except:
            loader = new_loader()
            yield contextualize(*next(loader))

loader = TrainLoader()

n_classes = int(train_y.max() - train_y.min() + 1)


# In[ ]:


def forward(y, y_bar):
    # TODO batchify
    y = my.onehot(y, n_classes)
    y_bar = F.softmax(y_bar, 1)
    return th.cat((y, y_bar), 1).view(1, -1)

def L_batch(y, y_bar):
    y_bar = th.max(y_bar, 1)[1].detach()
    return metrics.f1_score(y, y_bar, average='micro')

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

iw = {
    'none' : lambda x: th.zeros_like(x),
    'sqrt' : lambda x: th.sqrt(th.abs(x)),
    'linear' : lambda x: x,
    'quadratic' : lambda x: x * x,
}


# In[ ]:


th.random.manual_seed(1)
if cuda:
    th.cuda.manual_seed_all(1)

# c = my.MLP((3072, n_classes), F.relu)
# c = my.MLP((3072,) + (1024,) + (n_classes,), F.relu)
# c = my.MLP((3072,) + (1024,) * 2 + (n_classes,), F.relu)
# c = my.MLP((3072,) + (1024,) * 3 + (n_classes,), F.relu)
# c = lenet.LeNet(3, n_classes)
c = resnet.ResNet(depth=18, n_classes=n_classes)

critic = rn.RN(args.batch_size_c, 2 * n_classes, tuple(), (4 * n_classes, 64, 256), (256, 64) + (1,), F.relu, triu=True)

if cuda:
    c.cuda()
    critic.cuda()

c_optim = optim.Adam(c.parameters())
critic_optim = optim.Adam(critic.parameters())

for key, value in global_scores(c, test_loader).items():
    print(key, value)


# In[ ]:


for i in range(args.n_iterations):
    if verbose == 0:
        t0 = time.time()

    x, y = next(loader)
    
    y_c = c(x)
    L_c = L_batch(y, y_c)
    
    y_bar_listz, L_bar_listz, t_listz = [], [], []
    for j in range(args.n_perturbations):
        c_bar = copy.deepcopy(c)
        my.set_requires_grad(c_bar, False)
        my.perturb(c_bar, args.std)
        y_bar_listz.append(c_bar(x))
        L_bar_listz.append(L_batch(y, y_bar_listz[-1]))
        t_listz.append(L_c - L_bar_listz[-1])

    y_bar_tensorz = th.cat([y_bar.unsqueeze(0) for y_bar in y_bar_listz], 0)
    L_bar_tensorz = new_tensor(L_bar_listz)
    t_tensorz = new_tensor(t_listz)
    w_tensorz = th.exp(iw[args.iw](t_tensorz) / args.tau)
    w_tensorz /= th.sum(w_tensorz)

    writer.add_scalar('th.min(L_bar_tensorz)', th.min(L_bar_tensorz), i)
    writer.add_scalar('th.max(L_bar_tensorz)', th.max(L_bar_tensorz), i)
    writer.add_scalar('entropy', -th.sum(w_tensorz * th.log(w_tensorz)), i)

    if args.topk > 0:
        w_tensor, topk = th.topk(w_tensorz, args.topk)
        y_bar_tensor, t_tensor = y_bar_tensorz[topk], t_tensorz[topk]
        y_bar_list = [y_bar.squeeze(0) for y_bar in th.chunk(y_bar_tensor, args.topk, 0)]
    else:
        w_tensor, y_bar_tensor, t_tensor = w_tensorz, y_bar_tensorz, t_tensorz
        y_bar_list = y_bar_listz
    
    if verbose == 0:
        t1 = time.time()
        print('[iteration %d]t1 - t0: %f' % (i + 1, t1 - t0))
    
    z_c = forward(y, y_c)
    z_detached = z_c.detach()
    z_bar_list = [forward(y, y_bar).detach() for y_bar in y_bar_list] # TODO batchify
    z_bar_tensor = th.cat(z_bar_list, 0)
    
    if args.topk > 0:
        n_batches = int(args.topk / args.batch_size_critic)
    else:
        n_batches = int(args.n_perturbations / args.batch_size_critic)
    chunk = lambda x: th.chunk(x, n_batches, 0)
    z_bar_list, t_list, w_list = tuple(map(chunk, (z_bar_tensor, t_tensor, w_tensor)))
    for j in range(args.n_iterations_critic):
        for z_bar, t, w in zip(z_bar_list, t_list, w_list):
            delta = critic(z_detached) - critic(z_bar)
            mse = th.sum(w * (t - delta) ** 2)
            critic_optim.zero_grad()
            mse.backward()
            critic_optim.step()
        delta = critic(z_detached) - critic(z_bar_tensor)
        mse = th.sum(w_tensor * (t_tensor - delta) ** 2)
        writer.add_scalar('mse', mse, i * args.n_iterations_critic + j)

    if verbose == 0:
        t2 = time.time()
        print('[iteration %d]t2 - t1: %f' % (i + 1, t2 - t1))

    objective = -critic(z_c)
    c_optim.zero_grad()
    objective.backward()
    c_optim.step()

    if verbose == 0:
        t3 = time.time()
        print('[iteration %d]t3 - t2: %f' % (i + 1, t3 - t2))
    
    L_c = L_batch(y, c(x))
    writer.add_scalar('L_c', L_c, i)
    
    train_scores = global_scores(c, train_loader)
    test_scores = global_scores(c, test_loader)
    
    prefix = '0' * (len(str(args.n_iterations)) - len(str(i + 1)))
    print('[iteration %s%d]' % (prefix, i + 1) +           ' | '.join('%s %0.3f/%0.3f' % (key, value, test_scores[key]) for key, value in train_scores.items()))
    
    for key, value in train_scores.items():
        writer.add_scalar('train-' + key, value, i)
        
    for key, value in test_scores.items():
        writer.add_scalar('test-' + key, value, i)

