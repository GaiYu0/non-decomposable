
# coding: utf-8

# In[ ]:


import argparse
import copy
import collections
import time
import sklearn.metrics as metrics
import tensorboardX as tb
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import data
import guided_es
import lenet
import my
import resnet
import rn


# In[ ]:


'''
args = argparse.Namespace()
args.actor = 'linear'
# args.actor = 'lenet'
# args.actor = 'resnet'
args.alpha = 0.5
args.avg = 'binary' # average
args.bsa = 100 # batch size of actor
args.bscx = 1 # batch size of critic (x)
args.bscy = 1 # batch size of critic (y)
args.ckpt_every = 0
# args.ds = 'MNIST'
args.ds = 'CIFAR10' # data set
args.ges = True # guided es
args.gpu = 0
args.iw = 'none'
# args.iw = 'sqrt'
# args.iw = 'linear'
# args.iw = 'quadratic'
args.post = '91-under'
# args.post = '91-over'
args.lra = 1e-3 # learning rate of actor
args.lrc = 1e-3 # learning rate of critic
args.ni = 100 # number of iterations
args.nia = 1 # number of iterations (actor)
args.nic = 25 # number of iterations (critic)
args.np = 25 # number of perturbations
args.np_ges = 5 # number of perturbations for guided es
args.report_every = 1
args.resume = 0
args.ssc = 1 # sample size of critic
args.std = 0.1
args.std_ges = 0.1
args.tau = 0.1
args.tensorboard = True
args.verbose = -1
'''

parser = argparse.ArgumentParser()
parser.add_argument('--actor', type=str, default='linear')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='binary')
parser.add_argument('--bsa', type=int, default=100)
parser.add_argument('--bscx', type=int, default=1)
parser.add_argument('--bscy', type=int, default=1)
parser.add_argument('--ckpt-every', type=int, default=1000)
parser.add_argument('--ds', type=str, default='CIFAR10')
parser.add_argument('--ges', type=bool, default=False)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--iw', type=str, default='none')
parser.add_argument('--post', type=str, default='91-under')
parser.add_argument('--lra', type=float, default=1e-4)
parser.add_argument('--lrc', type=float, default=1e-3)
parser.add_argument('--ni', type=int, default=10000)
parser.add_argument('--nia', type=int, default=1)
parser.add_argument('--nic', type=int, default=25)
parser.add_argument('--np', type=int, default=25)
parser.add_argument('--np-ges', type=int, default=5)
parser.add_argument('--report-every', type=int, default=100)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--ssc', type=int, default=1)
parser.add_argument('--std', type=float, default=None)
parser.add_argument('--std-ges', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--tensorboard', type=bool, default=True)
parser.add_argument('--verbose', type=int, default=-1)
args = parser.parse_args()

keys = sorted(vars(args).keys())
excluded = ('ckpt_every', 'gpu', 'report_every', 'ni', 'resume', 'tensorboard', 'verbose')
experiment_id = 'parameter#' + '#'.join('%s:%s' % (key, str(getattr(args, key))) for key in keys if key not in excluded)
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

rbg = args.actor in ('lenet', 'resnet')
train_x, train_y, test_x, test_y = data.load_dataset(args.ds, rbg)
train_x, test_x = data.normalize(train_x, test_x)
if args.post == '91-under':
    label2ratio = {0 : 0.9, 1 : 0.1}
    train_x, train_y, test_x, test_y = data.random_subset(train_x, train_y, test_x, test_y, label2ratio)
elif args.post == '91-over':
    label2label = {9 : 1}
    label2label.update({i : 0 for i in range(9)})
    train_x, train_y, test_x, test_y = data.relabel(train_x, train_y, test_x, test_y, label2label)

train_set = utils.data.TensorDataset(train_x, train_y)
train_loader = utils.data.DataLoader(train_set, 4096, drop_last=False)
test_set = utils.data.TensorDataset(test_x, test_y)
test_loader = utils.data.DataLoader(test_set, 4096, drop_last=False)

loader = data.BalancedDataLoader(train_x, train_y, args.bsa, cuda)

n_classes = int(train_y.max() - train_y.min() + 1)


# In[ ]:


def batch(tensor, bsx, bsy):
    """
    Parameters
    ----------
    tensor : (x, y, z)
    """
    shapex, shapey, shapez = tensor.shape
    nx, ny = int(shapex / bsx), int(shapey / bsy)
    x_list = th.chunk(tensor, nx, 0)
    return sum([[y.view(-1, shapez).contiguous() for y in th.chunk(x, ny, 1)] for x in x_list], [])

def forward(actor, batch_list, yz=True, L=True):
    x_tuple, y_tuple = zip(*batch_list)
    x_tensor, y_tensor = th.cat(x_tuple), th.cat(y_tuple)
    z_tensor = actor(x_tensor)
    ret = []
    if yz:
        ret.append(th.cat([my.onehot(y_tensor, n_classes), F.softmax(z_tensor, 1)], 1).view(len(batch_list), -1))
    if L:
        z_list = th.chunk(z_tensor, len(batch_list))
        ret.append(new_tensor([L_batch(y, z) for y, z in zip(y_tuple, z_list)]).unsqueeze(1))
    return ret
    
def L_batch(y, z):
    y_bar = th.max(z, 1)[1]
    return metrics.f1_score(y, y_bar, average=args.avg)

iw = {
    'none' : lambda x: th.zeros_like(x),
    'quadratic' : lambda x: x * x,
}[args.iw]


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
unary = [2 * n_classes, 64]
binary = [2 * unary[-1], 64]
terminal = [64, 32, 16, 1]
critic = rn.RN(args.bsa, 2 * n_classes, unary, binary, terminal, F.relu, triu=True)

if cuda:
    actor.cuda()
    critic.cuda()

actor_bar = copy.deepcopy(actor)
my.set_requires_grad(actor_bar, False)
 
actor_optim = optim.Adam(actor.parameters(), lr=args.lra, amsgrad=True)
critic_optim = optim.Adam(critic.parameters(), lr=args.lrc, amsgrad=True)

if args.resume > 0:
    c.load_state_dict(th.load('ckpt/%s-actor-%d' % (experiment_id, args.resume)))
    critic.load_state_dict(th.load('ckpt/%s-critic-%d' % (experiment_id, args.resume)))
    c_optim.load_state_dict(th.load('ckpt/%s-actor_optim-%d' % (experiment_id, args.resume)))
    critic_optim.load_state_dict(th.load('ckpt/%s-critic_optim-%d' % (experiment_id, args.resume)))

report(actor, -1)


# In[ ]:


for i in range(args.resume, args.resume + args.ni):
    batch_list = [next(loader) for j in range(args.ssc)]
    
    my.set_requires_grad(actor, False)
    yz, L = forward(actor, batch_list)

    yzbar_list, Lbar_list, delta_list = [], [], []
    for j in range(args.np):
        my.perturb(actor_bar, args.std)
        yz_bar, L_bar = forward(actor_bar, batch_list)
        yzbar_list.append(yz_bar)
        Lbar_list.append(L_bar)
        delta_list.append(L - L_bar)
    if args.tensorboard:
        log_stats(th.cat(Lbar_list, 1), 'L_bar', i)
        
    yzbar_tensor = th.cat([yz_bar.unsqueeze(1) for yz_bar in yzbar_list], 1)
    delta_tensor = th.cat(delta_list, 1)
    weight_tensor = F.softmax(iw(delta_tensor), 1)
    entropy = th.sum(weight_tensor * th.log(weight_tensor)) / args.ssc
    if args.tensorboard:
        writer.add_scalar('entropy', entropy, i + 1)
    
    delta_tensor, weight_tensor = delta_tensor.unsqueeze(2), weight_tensor.unsqueeze(2)
    lambda_batch = lambda tensor: batch(tensor, args.bscx, args.bscy)
    yzbar_list, delta_list, weight_list = list(map(lambda_batch, [yzbar_tensor, delta_tensor, weight_tensor]))
    
    my.set_requires_grad(critic, True)
    for j in range(args.nic):
        for yz_bar, delta, weight in zip(yzbar_list, delta_list, weight_list):
            mse = th.sum(weight * (delta - (critic(yz) - critic(yz_bar))) ** 2)
            critic_optim.zero_grad()
            mse.backward()
            critic_optim.step()
        if args.tensorboard:
            writer.add_scalar('mse', mse, i * args.nic + j + 1)

    my.set_requires_grad(actor, True)
    my.set_requires_grad(critic, False)
    for j in range(args.nia):
        yz, L = forward(actor, batch_list)
        if args.tensorboard:
            log_stats(L, 'L', i * args.nia + j)
        
        objective = -th.mean(critic(yz))
        actor_optim.zero_grad()
        objective.backward()
        
        if args.ges:
            eval = lambda actor: forward(actor, batch_list, yz=False)[0]
            guided_es.guided_es(actor, eval, args.np_ges, args.std_ges, args.alpha)
            
        actor_optim.step()
    
    for p, p_bar in zip(actor.parameters(), actor_bar.parameters()):
        p_bar.data[:] = p.data
        
    if args.report_every > 0 and (i + 1) % args.report_every == 0:
        report(actor, i)

    if args.ckpt_every > 0 and (i + 1) % args.ckpt_every == 0:
        ckpt(actor, critic, actor_optim, critic_optim, i)

