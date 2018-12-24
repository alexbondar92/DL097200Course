import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.nn.init as init
import math

# Model 1
class model1(nn.Module):
    """ConvNet"""
    def __init__(self):
        super(model1, self).__init__()
        self.name = 'Model 1 - ConvNet'

        self.conv1 = nn.Conv2d(3, 8, 5)
        init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(8, 16, 5)
        init.xavier_uniform_(self.conv2.weight)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.loss = nn.NLLLoss()

    def forward(self, x):
        out = Func.max_pool2d(Func.relu(self.conv1(x)), (2, 2))
        out = Func.max_pool2d(Func.relu(self.conv2(out)), (2, 2))
        out = out.view(-1, 16 * 5 * 5)
        out = Func.relu(self.fc1(out))
        out = Func.relu(self.fc2(out))
        out = self.fc3(out)
        out = Func.log_softmax(out, dim=0)
        return out

# Model 2
class DenseNet(nn.Module):
    def __init__(self, growthRate, depth):
        super(DenseNet, self).__init__()

        self.name = 'Model 2 - DenseNet'


        self.loss = nn.NLLLoss()


        nDenseBlocks = (depth-4) // 3
        nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels//2))
        self.trans1 = Transition(nChannels, nOutChannels)

#        self.BatchNorm2d = nn.BatchNorm2d(nChannels)
#        self.relu = nn.ReLU()
#        self.Conv2d = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
#        self.avg_pool2d = Func.avg_pool2d(2)
#        self.trans1 = nn.Sequential(
#            nn.BatchNorm2d(nChannels),
#            nn.ReLU(),
#            nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False),
#            Func.avg_pool2d(2))
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.MaxPool2d(2))

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels//2))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, 10)

        #TODO: To check if it is needed....
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(Func.avg_pool2d(Func.relu(self.bn1(out)), 8))
        out = Func.log_softmax(self.fc(out))
        return out


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.BatchNorm1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(Func.relu(self.BatchNorm1(x)))
        out = self.conv2(Func.relu(self.BatchNorm2(out)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(Func.relu(self.bn1(x)))
        out = Func.avg_pool2d(out, 2)
        return out


def model2():
    return DenseNet(growthRate=12, depth=21)

