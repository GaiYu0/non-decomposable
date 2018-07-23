
# coding: utf-8

# In[ ]:


import argparse
import copy
import collections
import functools
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
import l1
import lenet
import mlp
import my
import nll
import resnet
import rn


# In[ ]:


'''
args = argparse.Namespace()
args.actor = 'mlp'
# args.actor = 'lenet'
# args.actor = 'resnet'
args.alpha = 0.5
args.avg = 'binary' # average
args.bsa = 100 # batch size of actor
args.bscx = 1 # batch size of critic (x)
args.bscy = 1 # batch size of critic (y)
args.ckpt_every = 0
args.cos = False
args.critic = 'nll'
# args.ds = 'MNIST'
# args.ds = 'CIFAR10'
args.ds = 'covtype'
args.ges = False # guided es
args.gpu = 0
args.iw = 'none'
# args.iw = 'sqrt'
# args.iw = 'linear'
# args.iw = 'quadratic'
args.lra = 1e-3 # learning rate of actor
args.lrc = 1e-3 # learning rate of critic
args.ni = 1000 # number of iterations
args.nia = 1 # number of iterations (actor)
args.nic = 25 # number of iterations (critic)
args.np = 25 # number of perturbations
args.np_ges = 1 # number of perturbations for guided es
args.post = 'covtype'
# args.post = '91-under'
# args.post = '91-over'
args.report_every = 1
args.resume = 0
args.sn = 'p' # source of noise
args.ssc = 1 # sample size of critic
args.std = 1
args.std_ges = 0.1
args.tau = 0.1
args.tb = True
'''

parser = argparse.ArgumentParser()
parser.add_argument('--actor', type=str, default='mlp')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='binary')
parser.add_argument('--bsa', type=int, default=None)
parser.add_argument('--bscx', type=int, default=1)
parser.add_argument('--bscy', type=int, default=1)
parser.add_argument('--ckpt-every', type=int, default=100)
parser.add_argument('--cos', type=bool, default=False)
parser.add_argument('--critic', type=str, default='nll')
parser.add_argument('--ds', type=str, default='covtype')
parser.add_argument('--ges', type=bool, default=False)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--iw', type=str, default='none')
parser.add_argument('--lra', type=float, default=None)
parser.add_argument('--lrc', type=float, default=None)
parser.add_argument('--ni', type=int, default=100)
parser.add_argument('--nia', type=int, default=1)
parser.add_argument('--nic', type=int, default=25)
parser.add_argument('--np', type=int, default=25)
parser.add_argument('--np-ges', type=int, default=None)
parser.add_argument('--post', type=str, default='covtype')
parser.add_argument('--report-every', type=int, default=1)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--sn', type=str, default='p')
parser.add_argument('--ssc', type=int, default=1)
parser.add_argument('--std', type=float, default=None)
parser.add_argument('--std-ges', type=float, default=None)
parser.add_argument('--tau', type=float, default=None)
parser.add_argument('--tb', type=bool, default=True)
args = parser.parse_args()

keys = sorted(vars(args).keys())
excluded = ('ckpt_every', 'gpu', 'report_every', 'ni', 'resume', 'tb')
experiment_id = 'ac#' + '#'.join('%s:%s' % (key, str(getattr(args, key))) for key in keys if key not in excluded)
if args.tb:
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
elif args.post == 'covtype':
    label2label = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 1, 5 : 0, 6 : 0}
    train_x, train_y, test_x, test_y = data.relabel(train_x, train_y, test_x, test_y, label2label)

bsl = {
    'MNIST'   : 4096,
    'CIFAR10' : 4096,
    'covtype' : 65536,
}[args.ds] # batch size of loader
train_set = utils.data.TensorDataset(train_x, train_y)
train_loader = utils.data.DataLoader(train_set, bsl, drop_last=False)
test_set = utils.data.TensorDataset(test_x, test_y)
test_loader = utils.data.DataLoader(test_set, bsl, drop_last=False)

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

def forward(actor, xyy, z=None, yz=True, softmax=None, J=None):
    xx, yy = zip(*xyy)
    x, y = th.cat(xx), th.cat(yy)

    ret = []
    if z is None:
        z = actor(x)
        ret.append(z)
    if yz:
        onehot_x = my.one_hot(y, n_classes)
        softmax_x = softmax(z, 1)
        cat_x = th.cat([onehot_x, softmax_x], 1)
        ret.append(cat_x.view(len(xyy), -1))
    if J:
        zz = th.chunk(z, len(xyy))
        J_x = new_tensor([J(y, z) for y, z in zip(yy, zz)])
        ret.append((J_x).unsqueeze(1))
    return ret

def J(y, z, average=args.avg):
    y_bar = th.max(z, 1)[1]
    return metrics.f1_score(y, y_bar, average)

def surrogate(yz):
    yz = th.chunk(yz, int(yz.size(1) / n_classes), 1)
    y, z = th.cat(yz[::2]), th.cat(yz[1::2])
    return th.norm(y - z, 1)
        
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
    accuracy, [precision, recall, f1, _] = my.global_scores(c, loader, score_list)
    return collections.OrderedDict({
        'accuracy'  : accuracy,
        'precision' : precision,
        'recall'    : recall,
        'f1'        : f1,
    })

def log_critic_stats(critic, i):
    if args.critic in ['l1', 'nll']:
        w_t = F.softmax(critic.w, 0)
        for j, w in enumerate(w_t):
            writer.add_scalar('class %d' % j, w, i + 1)
            
def log_stats(tensor, tag, i):
    writer.add_scalar('th.min(%s)' % tag, th.min(tensor), i + 1)
    writer.add_scalar('th.max(%s)' % tag, th.max(tensor), i + 1)
    writer.add_scalar('th.mean(%s)' % tag, th.mean(tensor), i + 1)
    
def report(actor, i):
    train_scores = global_scores(actor, train_loader)
    test_scores = global_scores(actor, test_loader)

    prefix = '0' * (len(str(args.ni)) - len(str(i + 1)))
    print('[iteration %s%d]' % (prefix, i + 1) +           ' | '.join('%s %0.3f/%0.3f' % (key, value, test_scores[key]) for key, value in train_scores.items()))

    if args.tb:
        for key, value in train_scores.items():
            writer.add_scalar('train-' + key, value, i + 1)
        for key, value in test_scores.items():
            writer.add_scalar('test-' + key, value, i + 1)


# In[ ]:


th.random.manual_seed(1)
if cuda:
    th.cuda.manual_seed_all(1)

if args.ds in ['MNIST', 'CIFAR10']:
    n_channels = {
        'MNIST'   : 1,
        'CIFAR10' : 3,
    }[args.ds]
    size = {
        'MNIST'   : 28,
        'CIFAR10' : 32,
    }[args.ds]
    actor = {
        'linear' : nn.Linear(n_channels * size ** 2, n_classes),
        'lenet'  : lenet.LeNet(3, n_classes, size),
        'resnet' : resnet.ResNet(depth=18, n_classes=n_classes),
    }[args.actor]
elif args.ds in ['covtype']:
    n_features = train_x.size(1)
    actor = {
        'linear' : nn.Linear(n_features, n_classes),
        'mlp'    : mlp.MLP([n_features, 60, 60, 80, n_classes], th.tanh)
    }[args.actor]

if args.critic == 'l1':
    critic = l1.WeightedL1Loss(n_classes)
elif args.critic == 'nll':
    critic = nll.WeightedNLLLoss(n_classes)
elif args.critic == 'rn':
    unary = [2 * n_classes, 256]
    binary = [2 * unary[-1], 256]
    terminal = [256, 1]
    critic = rn.RN(args.bsa, 2 * n_classes, unary, binary, terminal, F.relu, triu=True)

if cuda:
    actor.cuda()
    critic.cuda()

actor_bar = copy.deepcopy(actor)
my.set_requires_grad(actor_bar, False)
 
actor_opt = optim.Adam(actor.parameters(), lr=args.lra, amsgrad=True)
critic_opt = optim.Adam(critic.parameters(), lr=args.lrc, amsgrad=True)

if args.resume > 0:
    c.load_state_dict(th.load('ckpt/%s-actor-%d' % (experiment_id, args.resume)))
    critic.load_state_dict(th.load('ckpt/%s-critic-%d' % (experiment_id, args.resume)))
    c_opt.load_state_dict(th.load('ckpt/%s-actor_opt-%d' % (experiment_id, args.resume)))
    critic_opt.load_state_dict(th.load('ckpt/%s-critic_opt-%d' % (experiment_id, args.resume)))

report(actor, -1)


# In[ ]:


kwargs = {'softmax' : F.log_softmax, 'J' : J} if args.critic == 'nll' else {'softmax' : F.softmax, 'J' : J}
for i in range(args.resume, args.resume + args.ni):
    xyy = [next(loader) for j in range(args.ssc)]
    
    my.set_requires_grad(actor, False)
    z, yz, J_x = forward(actor, xyy, **kwargs)

    yz_barr, jx_barr, deltaa = [], [], []
    for j in range(args.np):
        if args.sn == 'a':
            epsilon = args.std * th.randn(z.shape, device=z.device)
            yz_bar, jx_bar = forward(actor_bar, xyy, z + epsilon, **kwargs)
        elif args.sn == 'p':
            my.perturb(actor_bar, args.std)
            z_bar, yz_bar, jx_bar = forward(actor_bar, xyy, **kwargs)

        yz_barr.append(yz_bar)
        jx_barr.append(jx_bar)
        deltaa.append(J_x - jx_bar)
    
    if args.tb:
        log_stats(th.cat(jx_barr, 1), 'J_bar', i)
    
    yz_bar = th.cat([yz_bar.unsqueeze(1) for yz_bar in yz_barr], 1)
    delta = th.cat(deltaa, 1)
    w = F.softmax(iw(delta), 1)
    entropy = th.sum(w * th.log(w)) / args.ssc
    
    if args.tb:
        writer.add_scalar('entropy', entropy, i + 1)
    
    delta, w = delta.unsqueeze(2), w.unsqueeze(2)
    partial = functools.partial(batch, bsx=args.bscx, bsy=args.bscy)
    yz_barr, deltaa, ww = map(partial, [yz_bar, delta, w])
    
    my.set_requires_grad(critic, True)
    for j in range(args.nic):
        for x_yzbar, x_delta, x_w in zip(yz_barr, deltaa, ww):
            mse = th.sum(x_w * (x_delta - (critic(yz) - critic(x_yzbar))) ** 2)
            critic_opt.zero_grad()
            mse.backward()
            critic_opt.step()
        if args.tb:
            writer.add_scalar('mse', mse, i * args.nic + j + 1)
    log_critic_stats(critic, i)

    my.set_requires_grad(actor, True)
    my.set_requires_grad(critic, False)
    for j in range(args.nia):
        z, yz, J_x = forward(actor, xyy, **kwargs)
        
        if args.cos:
            def hook(g):
                g = th.chunk(g, int(g.size(1) / n_classes), 1)
                y, z = [th.zeros(g[0].size(), device=g[0].device)] * int(len(g) / 2), g[1::2]
                g = th.cat(sum(zip(y, z), tuple()), 1)
                globals()['yz_grad'] = g
            yz.register_hook(hook)
        
        if args.tb:
            log_stats(J_x, 'J_x', i * args.nia + j)
        
        objective = -th.mean(critic(yz))
        actor_opt.zero_grad()
        objective.backward()
        
        if args.cos:
            yz = yz.detach()
            yz.requires_grad = True
            objective = surrogate(yz)
            yz.grad = None
            objective.backward()
            F.cosine_similarity(yz_grad, yz.grad)
        
        if args.ges:
            partial = lambda actor: forward(actor, xyy, yz=False)[0]
            guided_es.guided_es(actor, partial, args.np_ges, args.std_ges, args.alpha)
            
        actor_opt.step()
    
    for p, p_bar in zip(actor.parameters(), actor_bar.parameters()):
        p_bar.data[:] = p.data
        
    if args.ckpt_every > 0 and (i + 1) % args.ckpt_every == 0:
        ckpt(actor, critic, actor_opt, critic_opt, i)
        
    if args.report_every > 0 and (i + 1) % args.report_every == 0:
        report(actor, i)

