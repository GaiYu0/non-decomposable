
# coding: utf-8

# In[ ]:


import argparse
import collections
import copy
import sklearn.metrics as metrics
import tensorboardX as tb
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import data
import lenet
import my
import resnet


# In[ ]:


'''
args = argparse.Namespace()
args.actor = 'linear'
args.avg = 'binary' # average
args.bsa = 100 # batch size of actor
args.ckpt_every = 0
args.ds = 'cifar10' # data set
args.gpu = 0
args.lbl = '91' # labelling
args.lra = 1e-1 # learning rate of actor
args.ni = 100 # number of iterations
args.np = 25 # number of perturbations
args.report_every = 1
args.resume = 0
args.std = 1
args.tensorboard = True
'''

parser = argparse.ArgumentParser()
parser.add_argument('--actor', type=str, default='linear')
parser.add_argument('--avg', type=str, default='binary')
parser.add_argument('--bsa', type=int, default=100)
parser.add_argument('--ckpt-every', type=int, default=1000)
parser.add_argument('--ds', type=str, default='cifar10')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--lbl', type=str, default='91')
parser.add_argument('--lra', type=float, default=None)
parser.add_argument('--ni', type=int, default=10000)
parser.add_argument('--np', type=int, default=100)
parser.add_argument('--report-every', type=int, default=100)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--std', type=float, default=None)
parser.add_argument('--tensorboard', type=bool, default=True)
args = parser.parse_args()
'''

keys = sorted(vars(args).keys())
excluded = ('ckpt_every', 'gpu', 'report_every', 'ni', 'resume', 'tensorboard', 'verbose')
experiment_id = 'es#' + '#'.join('%s:%s' % (key, str(getattr(args, key))) for key in keys if key not in excluded)
if args.tensorboard:
    writer = tb.SummaryWriter('runs/' + experiment_id)


# In[ ]:


if args.gpu < 0:
    cuda = False
    new_tensor = th.FloatTensor
else:
    cuda = True
    new_tensor = th.cuda.FloatTensor
    th.cuda.set_device(args.gpu)

labelling = {} if args.lbl == '' else {(0, 9) : 0, (9, 10) : 1}
rbg = args.actor in ('lenet', 'resnet')
train_x, train_y, test_x, test_y = getattr(data, 'load_%s' % args.ds)(labelling, rbg, torch=True)

train_set = utils.data.TensorDataset(train_x, train_y)
train_loader = utils.data.DataLoader(train_set, 4096, drop_last=False)
test_set = utils.data.TensorDataset(test_x, test_y)
test_loader = utils.data.DataLoader(test_set, 4096, drop_last=False)

loader = data.BalancedDataLoader(train_x, train_y, args.bsa, cuda)

n_classes = int(train_y.max() - train_y.min() + 1)


# In[ ]:


def ckpt(actor, critic, actor_optim, critic_optim, i):
    th.save(actor.state_dict(), 'ckpt/%s-actor-%d' % (experiment_id, i + 1))
    th.save(critic.state_dict(), 'ckpt/%s-critic-%d' % (experiment_id, i + 1))
    th.save(actor_optim.state_dict(), 'ckpt/%s-actor_optim-%d' % (experiment_id, i + 1))
    th.save(critic_optim.state_dict(), 'ckpt/%s-critic_optim-%d' % (experiment_id, i + 1))

def global_scores(c, loader):
    key_list = ['accuracy', 'precision', 'recall', 'f1']
    score_list = [
        metrics.accuracy_score,
        lambda y, y_bar: metrics.precision_recall_fscore_support(y, y_bar, average=args.avg)
    ]
    accuracy, (precision, recall, f1, _) = my.global_scores(c, loader, score_list)
    return collections.OrderedDict({
        'accuracy'  : accuracy,
        'precision' : precision,
        'recall'    : recall,
        'f1'        : f1,
    })

def log_stats(tensor, tag, i):
    writer.add_scalar('th.min(%s)' % tag, th.min(tensor), i + 1)
    writer.add_scalar('th.max(%s)' % tag, th.max(tensor), i + 1)
    writer.add_scalar('th.mean(%s)' % tag, th.mean(tensor), i + 1)
    
def report(actor, i):
    train_scores = global_scores(actor, train_loader)
    test_scores = global_scores(actor, test_loader)

    prefix = '0' * (len(str(args.ni)) - len(str(i + 1)))
    print('[iteration %s%d]' % (prefix, i + 1) +           ' | '.join('%s %0.3f/%0.3f' % (key, value, test_scores[key]) for key, value in train_scores.items()))

    if args.tensorboard:
        for key, value in train_scores.items():
            writer.add_scalar('train-' + key, value, i + 1)
        for key, value in test_scores.items():
            writer.add_scalar('test-' + key, value, i + 1)


# In[ ]:


th.random.manual_seed(1)
if cuda:
    th.cuda.manual_seed_all(1)

n_channels = 1 if args.ds == 'mnist' else 3
size = 28 if args.ds == 'mnist' else 32
actor = {
    'linear' : nn.Linear(n_channels * size ** 2, n_classes),
    'lenet'  : lenet.LeNet(3, n_classes, size),
    'resnet' : resnet.ResNet(depth=18, n_classes=n_classes),
}[args.actor]

if cuda:
    actor.cuda()

actor_optim = optim.Adam(actor.parameters(), lr=args.lra, amsgrad=True)

my.set_requires_grad(actor, False)
actor_bar = copy.deepcopy(actor)

if args.resume > 0:
    c.load_state_dict(th.load('ckpt/%s-actor-%d' % (experiment_id, args.resume)))
    c_optim.load_state_dict(th.load('ckpt/%s-actor_optim-%d' % (experiment_id, args.resume)))

report(actor, -1)


# In[ ]:


for i in range(args.resume, args.resume + args.ni):
    x, y = next(loader)
    
    epsilon_list, fscore_list = [], []
    z_list = []
    for j in range(args.np):
        epsilon_list.append([])
        for p in actor_bar.parameters():
            epsilon_list[-1].append(args.std * th.randn(p.shape, device=p.device))
            p.data += epsilon_list[-1][-1]
        
        y_bar = th.max(actor_bar(x), 1)[1]
        fscore_list.append(metrics.f1_score(y, y_bar, average=args.avg))
        z_list.append(actor_bar(x))
        
        for p, p_bar in zip(actor.parameters(), actor_bar.parameters()):
            p_bar.data[:] = p.data
    
    epsilon_zip = zip(*epsilon_list)
    for p, epsilon_tuple, fscore in zip(actor.parameters(), epsilon_zip, fscore_list):
        epsilon_tensor = th.stack(epsilon_tuple)
        shape = [-1] + [1] * (epsilon_tensor.dim() - 1)
        fscore_tensor = th.tensor(fscore_list, device=p.device).view(*shape)
        p.grad = -1.0 / args.std * th.mean(fscore_tensor * epsilon_tensor, 0)
    actor_optim.step()
    
    for p, p_bar in zip(actor.parameters(), actor_bar.parameters()):
        p_bar.data[:] = p.data
    
    if args.report_every > 0 and (i + 1) % args.report_every == 0:
        report(actor, i)
        
    if args.ckpt_every > 0 and (i + 1) % args.ckpt_every == 0:
        ckpt(actor, critic, actor_optim, critic_optim, i)

