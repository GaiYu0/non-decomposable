
# coding: utf-8

# In[ ]:


import argparse
import collections
import sklearn.metrics as metrics
import tensorboardX as tb
import torch as th
import torch.nn.functional as F
import torch.nn.modules.loss as loss
import torch.optim as optim
import torch.utils as utils
import data
import my
import lenet
import resnet


# In[ ]:


args = argparse.Namespace()
args.batch_size = 50
args.gpu = 0
args.n_iterations = 500000

keys = sorted(vars(args).keys())
excluded = ('gpu',)
run_id = 'ce-' + '-'.join('%s-%s' % (key, str(getattr(args, key))) for key in keys if key not in excluded)
writer = tb.SummaryWriter('runs/' + run_id)


# In[ ]:


if args.gpu < 0:
    cuda = False
else:
    cuda = True
    th.cuda.set_device(args.gpu)

train_x, train_y, test_x, test_y = data.load_cifar10(rbg=True, torch=True)

train_set = utils.data.TensorDataset(train_x, train_y)
train_loader = utils.data.DataLoader(train_set, 4096, drop_last=False)
test_set = utils.data.TensorDataset(test_x, test_y)
test_loader = utils.data.DataLoader(test_set, 4096, drop_last=False)

def TrainLoader():
    dataset = utils.data.TensorDataset(train_x, train_y)
    new_loader = lambda: iter(utils.data.DataLoader(dataset, args.batch_size, shuffle=True))
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


# In[ ]:


# c = lenet.LeNet(3, n_classes)
c = resnet.ResNet(18, n_classes)

if cuda:
    c.cuda()
    
# optimizer = optim.SGD(c.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
optimizer = optim.Adam(c.parameters())

for key, value in global_scores(c, test_loader).items():
    print(key, value)


# In[ ]:


n_iterations = 0
for i in range(args.n_iterations):
    x, y = next(loader)
    ce = loss.CrossEntropyLoss()(c(x), y)
    optimizer.zero_grad()
    ce.backward()
    optimizer.step()

    train_scores = global_scores(c, train_loader)
    test_scores = global_scores(c, test_loader)

    prefix = '0' * (len(str(args.n_iterations)) - len(str(i + 1)))
    print('[iteration %s%d]' % (prefix, i + 1) +           ' | '.join('%s %0.3f/%0.3f' % (key, value, test_scores[key]) for key, value in train_scores.items()))

    for key, value in train_scores.items():
        writer.add_scalar('train-' + key, value, i)

    for key, value in test_scores.items():
        writer.add_scalar('test-' + key, value, i)


# In[ ]:


# for pg in optimizer.param_groups:
#     pg['lr'] *= 0.1

