
# coding: utf-8

# In[ ]:


import argparse
import collections
import importlib
import sklearn.metrics as metrics
import tensorboardX as tb
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as loss
import torch.optim as optim
import torch.utils as utils
import data
import my
import lenet
import resnet


# In[ ]:


'''
args = argparse.Namespace()
args.actor = 'linear'
# args.actor = 'lenet'
# args.actor = 'resnet'
args.average = 'binary'
args.batch_size = 100
# args.dataset = 'mnist'
args.dataset = 'cifar10'
args.gpu = 1
# args.labelling = ''
args.labelling = '91'
args.lr = 1e-3
args.n_iterations = 1000
args.report_every = 1
args.sampler = 'over_sampling.RandomOverSampler'
'''

parser = argparse.ArgumentParser()
parser.add_argument('--actor', type=str, default='linear')
parser.add_argument('--average', type=str, default='binary')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--labelling', type=str, default='91')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--n-iterations', type=int, default=10000)
parser.add_argument('--report-every', type=int, default=1)
parser.add_argument('--sampler', type=str, default=None)
args = parser.parse_args()

keys = sorted(vars(args).keys())
excluded = ('gpu', 'report_every', 'n_iterations')
run_id = 'cex-' + '-'.join('%s-%s' % (key, str(getattr(args, key))) for key in keys if key not in excluded)
writer = tb.SummaryWriter('runs/' + run_id)


# In[ ]:


if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)

labelling = {} if args.labelling == '' else {(0, 9) : 0, (9, 10) : 1}
rbg = args.actor in ('lenet', 'resnet')
train_x, train_y, test_x, test_y = getattr(data, 'load_%s' % args.dataset)(labelling, rbg, torch=True)

if args.sampler:
    a, b = args.sampler.split('.')
    sampler = getattr(importlib.import_module('imblearn.' + a), b)()
    train_x, train_y = sampler.fit_sample(train_x, train_y)
    train_x, train_y = th.from_numpy(train_x), th.from_numpy(train_y)

train_set = utils.data.TensorDataset(train_x, train_y)
train_loader = utils.data.DataLoader(train_set, 4096, drop_last=False)
test_set = utils.data.TensorDataset(test_x, test_y)
test_loader = utils.data.DataLoader(test_set, 4096, drop_last=False)

loader = data.BalancedDataLoader(train_x, train_y, args.batch_size, cuda)

n_classes = int(train_y.max() - train_y.min() + 1)


# In[ ]:


def global_scores(c, loader):
    key_list = ['accuracy', 'precision', 'recall', 'f1']
    score_list = [
        metrics.accuracy_score,
        lambda y, y_bar: metrics.precision_recall_fscore_support(y, y_bar, average=args.average)
    ]
    accuracy, (precision, recall, f1, _) = my.global_scores(c, loader, score_list)
    return collections.OrderedDict({
        'accuracy'  : accuracy,
        'precision' : precision,
        'recall'    : recall,
        'f1'        : f1,
    })

def report(actor, i):
    train_scores = global_scores(actor, train_loader)
    test_scores = global_scores(actor, test_loader)

    prefix = '0' * (len(str(args.n_iterations)) - len(str(i + 1)))
    print('[iteration %s%d]' % (prefix, i + 1) +           ' | '.join('%s %0.3f/%0.3f' % (key, value, test_scores[key]) for key, value in train_scores.items()))

    for key, value in train_scores.items():
        writer.add_scalar('train-' + key, value, i + 1)

    for key, value in test_scores.items():
        writer.add_scalar('test-' + key, value, i + 1)


# In[ ]:


n_channels = 1 if args.dataset == 'mnist' else 3
size = 28 if args.dataset == 'mnist' else 32
actor = {
    'linear' : nn.Linear(n_channels * size ** 2, n_classes),
    'lenet'  : lenet.LeNet(3, n_classes, size),
    'resnet' : resnet.ResNet(depth=18, n_classes=n_classes),
}[args.actor]

if cuda:
    actor.cuda()
    
optimizer = optim.Adam(actor.parameters(), lr=args.lr, amsgrad=True)

report(actor, -1)


# In[ ]:


for i in range(args.n_iterations):
    x, y = next(loader)
    ce = loss.CrossEntropyLoss()(actor(x), y)
    optimizer.zero_grad()
    ce.backward()
    optimizer.step()

    if (i + 1) % args.report_every == 0:
        report(actor, i)
