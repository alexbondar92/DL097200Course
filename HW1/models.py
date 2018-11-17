import torch.nn as nn
import torch.nn.functional as Func


# Model
class model1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(model1, self).__init__()

        self.name = 'FC [78, 39, 10], wd=1e-5'
        hidden_size = 78

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)
        self.linear3 = nn.Linear(hidden_size//2, num_classes)
        self.relu = nn.ReLU()
        self.loss = nn.NLLLoss()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = Func.log_softmax(out)

        return out
