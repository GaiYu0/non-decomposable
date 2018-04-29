# https://github.com/kuangliu/pytorch-cifar

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes,
                     kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

def config(depth):
    return {
        18  : (BasicBlock, [2, 2, 2, 2]),
        34  : (BasicBlock, [3, 4, 6, 3]),
        50  : (Bottleneck, [3, 4, 6, 3]),
        101 : (Bottleneck, [3, 4, 23, 3]),
        152 : (Bottleneck, [3, 8, 36, 3]),
    } [depth]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(z))
        z += self.shortcut(x)
        z = F.relu(z)
        return z

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = self.bn3(self.conv3(z))
        z += self.shortcut(x)
        z = F.relu(z)
        return z

class ResNet(nn.Module):
    def __init__(self, depth, n_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block, num_blocks = config(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = F.avg_pool2d(z, 8)
        z = z.view(z.size(0), -1)
        z = self.linear(z)
        return z
