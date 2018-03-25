from __future__ import print_function
import argparse
import copy
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset
import my

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--classifier-path', type=str, default='')
parser.add_argument('--critic-path', type=str, default='')
parser.add_argument('gpu', type=int)
parser.add_argument('--inner-actor', type=int, default=5)
parser.add_argument('--inner-critic', type=int, default=5)
parser.add_argument('--n-perturbations', type=int, default=25)
parser.add_argument('--n-test', type=int, default=0)
parser.add_argument('--n-train', type=int, default=0)
parser.add_argument('--optimizer-path', type=str, default='')
parser.add_argument('outer', type=int)
parser.add_argument('--sample-size', type=int, default=64)
parser.add_argument('--std', type=float, default=0.1)
args = parser.parse_args()

train_data, train_labels, test_data, test_labels = \
    my.unbalanced_cifar10(args.n_train, args.n_test, p=[])

print('data loaded')

n_features = train_data.shape[1]
n_classes = int(train_labels.max() - train_labels.min() + 1)

train_data_np, train_labels_np, test_data_np, test_labels_np = \
    train_data, train_labels, test_data, test_labels
    
train_data = th.from_numpy(train_data).float()
train_labels = th.from_numpy(train_labels).long()
test_data = th.from_numpy(test_data).float()
test_labels = th.from_numpy(test_labels).long()

train_loader = DataLoader(TensorDataset(train_data, train_labels), 1024, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_labels), 1024)

cuda = True
if cuda:
    th.cuda.set_device(args.gpu)

class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 2, 1)
        self.linear = nn.Linear(8, n_classes)
    
    def forward(self, x):
        if x.dim() != 4:
            x = x.view(-1, 3, 32, 32)
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, 8)
        x = self.linear(x.view(-1, 8))
        return x

nd_stats = [my.accuracy] + [my.nd_curry(stat, n_classes)
                            for stat in (my.nd_precision, my.nd_recall, my.nd_f_beta)]
L = lambda c, loader: my.global_stats(c, loader, my.nd_curry(my.nd_f_beta, n_classes))

def forward(classifier, pair):
    X, y = pair
    y = my.onehot(y, n_classes)
    y_bar = F.softmax(classifier(X), 1)
    return th.cat((y, y_bar), 1).view(1, -1)

def sample(sample_size, batch_size):
    samples = [my.sample_subset(train_data_np, train_labels_np, sample_size)
               for k in range(batch_size)]
    if cuda:
        samples = [(X.cuda(), y.cuda()) for (X, y) in samples]
    return [(Variable(X), Variable(y)) for (X, y) in samples]

th.random.manual_seed(1)
th.cuda.manual_seed_all(1)

# c = nn.Linear(n_features, n_classes)
# c = my.MLP((n_features,) + (512,) * 1 + (n_classes,), F.relu)
c = CNN(n_classes)
critic = my.RN(args.sample_size, 2 * n_classes, (512,) * 3 + (1,), F.relu)

if cuda:
    c.cuda()
    critic.cuda()

# c_optim = SGD(c.parameters(), 0.1, momentum=0.5)
# critic_optim = SGD(critic.parameters(), 0.1, momentum=0.5)
c_optim = Adam(c.parameters(), 1e-3)
critic_optim = Adam(critic.parameters(), 1e-3)

accuracy, precision, recall, f1 = my.global_stats(c, test_loader, nd_stats)
print('accuracy: %f, precision: %f, recall: %f, f1: %f' %
      tuple(map(float, (accuracy, precision, recall, f1))))
# TODO unusual precision

# target_mean, target_std = [], []
for i in range(args.outer):
    for p in c.parameters():
        p.requires_grad = False
    L_c = L(c, train_loader)
#   L_c = L(c, test_loader)
    c_bar_list, target_list = [], []
    for j in range(args.n_perturbations):
        c_bar_list.append(my.perturb(c, args.std))
        target = L_c - L(c_bar_list[-1], train_loader)
#       target = L_c - L(c_bar_list[-1], test_loader)
        target_list.append(target[0])
#   target_mean.append(float(th.mean(th.cat(target_list))))
#   target_std.append(float(th.std(th.cat(target_list))))
    tau = th.std(th.cat(target_list))
    weight_list = [th.exp(-target / tau) for target in target_list]
    
    s = sample(args.sample_size, args.batch_size)
    y = th.cat([forward(c, x) for x in s], 0).detach()
    for j in range(args.inner_critic): # TODO mini-batch
        for c_bar, target, weight in zip(c_bar_list, target_list, weight_list):
            y_bar = th.cat([forward(c_bar, x) for x in s], 0).detach()
            delta = th.mean(critic(y) - critic(y_bar), 0)
            mse = weight * MSELoss()(delta, target)
            critic_optim.zero_grad()
            mse.backward()
            critic_optim.step()

    for p in c.parameters():
        p.requires_grad = True
    c_parameters = copy.deepcopy(tuple(c.parameters()))
    for j in range(args.inner_actor):
        y = th.cat([forward(c, x) for x in s], 0)
        objective = -th.mean(critic(y))
        c_optim.zero_grad()
        objective.backward()
        c_optim.step()
        if any(float(th.max(th.abs(p - q))) > args.std
               for p, q in zip(c_parameters, c.parameters())): break

    if (i + 1) % 1 == 0:
        f1 = my.global_stats(c, test_loader, my.nd_curry(my.nd_f_beta, n_classes))
        print('[iteration %d]f1: %f' % (i + 1, f1))
        if float(f1) > 0.2:
            if args.classifier_path:
                th.save(c.state_dict(), args.classifier_path)
            if args.classifier_path:
                th.save(critic.state_dict(), args.critic_path)
            if args.optimizer_path:
                th.save(c_optim.state_dict(), args.optimizer_path + 'classifier')
                th.save(critic_optim.state_dict(), args.optimizer_path + 'critic')
            break
