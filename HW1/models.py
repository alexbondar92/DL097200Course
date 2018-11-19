import torch
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

class model2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(model2, self).__init__()

        self.name = 'FC [50x11, 10], wd=1e-5'
        hidden_size = 50

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.loss = nn.NLLLoss()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.linear3(out)
        out = Func.log_softmax(out)

        return out

class model3(nn.Module):
    def __init__(self, input_size, num_classes):
        super(model3, self).__init__()

        self.name = 'Split image-4, wd=1e-5'
        hidden_size = 210

        self.linear1 = nn.Linear(input_size//4, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)
        self.linear3 = nn.Linear(hidden_size//2, num_classes)
        self.linear4 = nn.Linear(4*num_classes, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.loss = nn.NLLLoss()

    def forward(self, x):
        x = x.view(-1, 28, 28)
        x_u, x_d = torch.split(x, 14, dim=1)
        x_ul, x_ur = torch.split(x_u, 14, dim=2)
        x_dl, x_dr = torch.split(x_d, 14, dim=2)

        x_ul = x_ul.contiguous().view(-1, 14 * 14)
        x_ur = x_ur.contiguous().view(-1, 14 * 14)
        x_dl = x_dl.contiguous().view(-1, 14 * 14)
        x_dr = x_dr.contiguous().view(-1, 14 * 14)


        out1 = self.linear1(x_ul)
        out1 = self.relu(out1)
        out1 = self.linear2(out1)
        out1 = self.relu(out1)
        out1 = self.linear3(out1)
        out1 = self.relu(out1)

        out2 = self.linear1(x_ur)
        out2 = self.relu(out2)
        out2 = self.linear2(out2)
        out2 = self.relu(out2)
        out2 = self.linear3(out2)
        out2 = self.relu(out2)

        out3 = self.linear1(x_dl)
        out3 = self.relu(out3)
        out3 = self.linear2(out3)
        out3 = self.relu(out3)
        out3 = self.linear3(out3)
        out3 = self.relu(out3)

        out4 = self.linear1(x_dr)
        out4 = self.relu(out4)
        out4 = self.linear2(out4)
        out4 = self.relu(out4)
        out4 = self.linear3(out4)
        out4 = self.relu(out4)

        #out = torch.cat((torch.cat((out1, out2),1), torch.cat((out3, out4),1)), 1)
        out = torch.cat((out1, out2, out3, out4), 1)
        out = self.linear4(out)
        out = Func.log_softmax(out)

        return out