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
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--actor-iterations', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--critic-iterations', type=int, default=10)
parser.add_argument('--hist', type=str, default='')
parser.add_argument('--n-iterations', type=int)
parser.add_argument('--n-perturbations', type=int, default=50)
parser.add_argument('--n-test', type=int, default=0)
parser.add_argument('--n-train', type=int, default=0)
parser.add_argument('--pre', type=str, default='')
parser.add_argument('--sample-size', type=int, default=64)
parser.add_argument('--std', type=float, default=1e-1)
parser.add_argument('--tau', type=float, default=1e-3)
args = parser.parse_args()

print(args)

th.random.manual_seed(1)
th.cuda.manual_seed_all(1)

train_x, train_y, test_x, test_y = my.load_cifar10(rbg=True)
# train_x, train_y, test_x, test_y = my.load_cifar10(partition=(0, 1, 10), rbg=True)

if args.n_train > 0:
    train_x, train_y = train_x[:args.n_train], train_y[:args.n_train]

if args.n_test > 0:
    test_x, test_y = test_x[:args.n_test], test_y[:args.n_test]

train_x = th.from_numpy(train_x).float()
train_y = th.from_numpy(train_y).long()
test_x = th.from_numpy(test_x).float()
test_y = th.from_numpy(test_y).long()
dataset = TensorDataset(train_x, train_y)

n_classes = int(train_y.max() - train_y.min() + 1)

cuda = True # always GPU 0

# TODO correctness of drop_last?
train_loader = DataLoader(TensorDataset(train_x, train_y),
                          1024 * 4, shuffle=True, drop_last=False)
test_loader = DataLoader(TensorDataset(test_x, test_y), 1024 * 4, drop_last=False)

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

c = CNN(n_classes)
# c = resnet.ResNet(18, n_classes)
if args.pre:
    c.load_state_dict(th.load(args.pre))
critic = my.RN(args.sample_size, 2 * n_classes, (32, 64), (64,) * 1 + (1,), F.relu)
if cuda:
    c.cuda()
    critic.cuda()
c_optim = Adam(c.parameters(), 1e-3)
critic_optim = Adam(critic.parameters(), 1e-3)

nd_f_beta = my.nd_curry(my.nd_f_beta, n_classes)
L = lambda c, loader: my.parallel_global_stats(c, loader, nd_f_beta, range(4))
# L = lambda c, loader: my.global_stats(c, loader, nd_f_beta)
L_mini = lambda c, loader: my.global_stats(c, loader, nd_f_beta)

def forward(classifier, xy):
    x, y = xy
    y = my.onehot(y, n_classes)
    y_bar = F.softmax(classifier(x), 1)
    return th.cat((y, y_bar), 1).view(1, -1)

def objective(c, critic, s, ce=0.0):
    y_onehot = [my.onehot(y, n_classes) for x, y in s]
    z_list = [c(x) for x, y in s]
    cat = [th.cat((y, F.softmax(z, 1)), 1).view(1, -1) for y, z in zip(y_onehot, z_list)]
    ret = -th.mean(critic(th.cat(cat, 0)))
    if ce > 0:
        ret += ce / len(s) * sum(F.nll_loss(F.log_softmax(z, 1), y)
                                 for z, (x, y) in zip(z_list, s))
    return ret

def sample(sample_size, batch_size):
    dl = DataLoader(dataset, sample_size, shuffle=True)
    s = it.takewhile(lambda x: x[0] < batch_size, enumerate(dl))
    s = [(Variable(x), Variable(y)) for _, (x, y) in s]
    if cuda:
        s = [(x.cuda(), y.cuda()) for (x, y) in s]
    return s

state_dict_cpu2gpu = lambda state_dict: {key : value.cuda() \
                                         for key, value in state_dict.items()}
state_dict_gpu2cpu = lambda state_dict: {key : value.cpu() \
                                         for key, value in state_dict.items()}

# TODO: RN: 1x1 conv

# print('initial f1: %f' % L(c, train_loader))
print('initial f1: %f' % L(c, test_loader))

hist = []
for i in range(args.n_iterations):
    hist.append({})
#   hist[-1]['c_state_dict'] = copy.deepcopy(state_dict_gpu2cpu(c.state_dict()))

    t0 = time.time()

    s = sample(args.sample_size, args.batch_size)

    c.eval()
    critic.train()
    L_c = L_mini(c, s)
#   L_c = L(c, train_loader)
    c_bar_list, t_list = [], []
    f1_list = []
    for j in range(args.n_perturbations):
        c_bar_list.append(my.perturb(c, args.std))
        L_bar = L_mini(c_bar_list[-1], s)
#       L_bar = L(c_bar_list[-1], train_loader)
        f1_list.append(float(L_bar))
        t = L_c - L_bar
        t_list.append(t[0])
    hist[-1]['f1_list'] = f1_list
    w_list = [th.exp((L_c - t) * t**2 / args.tau) for t in t_list]
#     w_list = [th.exp(L_c - t / tau) for t in t_list]
#     w_list = [th.exp(t**2 / tau) for t in t_list]
    z = sum(w_list)
    w_list = [(w / z).detach() for w in w_list]
    hist[-1]['w_list'] = w_list

    '''
    t1 = time.time()
    print('[iteration %d]t1 - t0: %f' % (i + 1, t1 - t0))
    '''

#   s = sample(args.sample_size, args.batch_size)
#   hist[-1]['s'] = s
    y_bar_list = [th.cat([forward(c_bar, x) for x in s], 0).detach() \
                  for c_bar in c_bar_list]

    y = th.cat([forward(c, x) for x in s], 0).detach()
    for j in range(args.critic_iterations):
        mse = 0
        for y_bar, t, w in zip(y_bar_list, t_list, w_list):
            delta = th.mean(critic(y) - critic(y_bar), 0)
            mse += w * MSELoss()(delta, t)
        critic_optim.zero_grad()
        mse.backward()
        critic_optim.step()
#   hist[-1]['critic_state_dict'] = copy.deepcopy(state_dict_gpu2cpu(critic.state_dict()))

    '''
    t2 = time.time()
    print('[iteration %d]t2 - t1: %f' % (i + 1, t2 - t1))
    '''

    c.train()
    critic.eval()
    c_param = copy.deepcopy(tuple(c.parameters()))
    for j in range(args.actor_iterations):
        ret = objective(c, critic, s, 1)
        c_optim.zero_grad()
        ret.backward()
        c_optim.step()
        if any(float(th.max(th.abs(p - q))) > args.std \
               for p, q in zip(c_param, c.parameters())): break

    '''
    t3 = time.time()
    print('[iteration %d]t3 - t2: %f' % (i + 1, t3 - t2))
    '''

    if (i + 1) % 1 == 0:
#       f1 = L(c, train_loader)
        f1 = L(c, test_loader)
        hist[-1]['f1'] = float(f1)
        print('[iteration %d]f1: %f' % (i + 1, f1))

if args.hist:
    pickle.dump(hist, open(args.hist, 'wb'))
