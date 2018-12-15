import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.nn.init as init

# Model
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