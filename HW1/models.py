import torch
import torch.nn as nn
import torch.nn.functional as Func

# Model
class model1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(model1, self).__init__()

        self.name = 'Model 1 - FC [78, 39, 10]'
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

        self.name = 'Model 2 - FC [50x11, 10]'
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

        self.name = 'Model 3 - Split image-4, FC [52, 26, 10, concat, 10]'
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

class model4(nn.Module):
    def __init__(self, input_size, num_classes):
            super(model4, self).__init__()

            self.name = 'Model 4 - Split image-16'
            hidden_size = 36

            self.linear1 = nn.Linear(input_size // 16, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
            self.linear3 = nn.Linear(16 * (hidden_size // 2), 8 * (hidden_size // 2))
            self.linear4 = nn.Linear(8 * (hidden_size // 2), 4 * (hidden_size // 2))
            self.linear5 = nn.Linear(4 * (hidden_size // 2), 2 * (hidden_size // 2))
            self.linear6 = nn.Linear(2 * (hidden_size // 2), num_classes)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()
            self.loss = nn.NLLLoss()

    def forward(self, x):
            x = x.view(-1, 28, 28)
            x_u, x_d = torch.split(x, 14, dim=1)
            x_ul, x_ur = torch.split(x_u, 14, dim=2)
            x_dl, x_dr = torch.split(x_d, 14, dim=2)

            x_ulu, x_uld = torch.split(x_ul, 7, dim=1)
            x_11, x_12 = torch.split(x_ulu, 7, dim=2)
            x_21, x_22 = torch.split(x_uld, 7, dim=2)

            x_uru, x_urd = torch.split(x_ur, 7, dim=1)
            x_13, x_14 = torch.split(x_uru, 7, dim=2)
            x_23, x_24 = torch.split(x_urd, 7, dim=2)

            x_dlu, x_dld = torch.split(x_dl, 7, dim=1)
            x_31, x_32 = torch.split(x_dlu, 7, dim=2)
            x_41, x_42 = torch.split(x_dld, 7, dim=2)

            x_dru, x_drd = torch.split(x_dr, 7, dim=1)
            x_33, x_34 = torch.split(x_dru, 7, dim=2)
            x_43, x_44 = torch.split(x_drd, 7, dim=2)

            x_11 = x_11.contiguous().view(-1, 7 * 7)
            x_12 = x_12.contiguous().view(-1, 7 * 7)
            x_13 = x_13.contiguous().view(-1, 7 * 7)
            x_14 = x_14.contiguous().view(-1, 7 * 7)
            x_21 = x_21.contiguous().view(-1, 7 * 7)
            x_22 = x_22.contiguous().view(-1, 7 * 7)
            x_23 = x_23.contiguous().view(-1, 7 * 7)
            x_24 = x_24.contiguous().view(-1, 7 * 7)
            x_31 = x_31.contiguous().view(-1, 7 * 7)
            x_32 = x_32.contiguous().view(-1, 7 * 7)
            x_33 = x_33.contiguous().view(-1, 7 * 7)
            x_34 = x_34.contiguous().view(-1, 7 * 7)
            x_41 = x_41.contiguous().view(-1, 7 * 7)
            x_42 = x_42.contiguous().view(-1, 7 * 7)
            x_43 = x_43.contiguous().view(-1, 7 * 7)
            x_44 = x_44.contiguous().view(-1, 7 * 7)

            out11 = self.linear1(x_11)
            out11 = self.relu(out11)
            out11 = self.linear2(out11)
            out11 = self.relu(out11)

            out12 = self.linear1(x_12)
            out12 = self.relu(out12)
            out12 = self.linear2(out12)
            out12 = self.relu(out12)

            out13 = self.linear1(x_13)
            out13 = self.relu(out13)
            out13 = self.linear2(out13)
            out13 = self.relu(out13)

            out14 = self.linear1(x_14)
            out14 = self.relu(out14)
            out14 = self.linear2(out14)
            out14 = self.relu(out14)

            out21 = self.linear1(x_21)
            out21 = self.relu(out21)
            out21 = self.linear2(out21)
            out21 = self.relu(out21)

            out22 = self.linear1(x_22)
            out22 = self.relu(out22)
            out22 = self.linear2(out22)
            out22 = self.relu(out22)

            out23 = self.linear1(x_23)
            out23 = self.relu(out23)
            out23 = self.linear2(out23)
            out23 = self.relu(out23)

            out24 = self.linear1(x_24)
            out24 = self.relu(out24)
            out24 = self.linear2(out24)
            out24 = self.relu(out24)

            out31 = self.linear1(x_31)
            out31 = self.relu(out31)
            out31 = self.linear2(out31)
            out31 = self.relu(out31)

            out32 = self.linear1(x_32)
            out32 = self.relu(out32)
            out32 = self.linear2(out32)
            out32 = self.relu(out32)

            out33 = self.linear1(x_33)
            out33 = self.relu(out33)
            out33 = self.linear2(out33)
            out33 = self.relu(out33)

            out34 = self.linear1(x_34)
            out34 = self.relu(out34)
            out34 = self.linear2(out34)
            out34 = self.relu(out34)

            out41 = self.linear1(x_41)
            out41 = self.relu(out41)
            out41 = self.linear2(out41)
            out41 = self.relu(out41)

            out42 = self.linear1(x_42)
            out42 = self.relu(out42)
            out42 = self.linear2(out42)
            out42 = self.relu(out42)

            out43 = self.linear1(x_43)
            out43 = self.relu(out43)
            out43 = self.linear2(out43)
            out43 = self.relu(out43)

            out44 = self.linear1(x_44)
            out44 = self.relu(out44)
            out44 = self.linear2(out44)
            out44 = self.relu(out44)

            # out = torch.cat((torch.cat((out1, out2),1), torch.cat((out3, out4),1)), 1)
            out = torch.cat((out11, out12, out13, out14, out21, out22, out23, out24, out31, out32, out33, out34, out41, out42, out43, out44), 1)
            out = self.linear3(out)
            out = self.relu(out)
            out = self.linear4(out)
            out = self.relu(out)
            out = self.linear5(out)
            out = self.relu(out)
            out = self.linear6(out)
            out = Func.log_softmax(out)
            return out

class model5(nn.Module):
    def __init__(self, input_size, num_classes):
            super(model5, self).__init__()

            self.name = 'Model 5 - Split image-16 11'
            hidden_size = 120

            self.linear1 = nn.Linear(input_size // 16, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size*3//4)
            self.linear3 = nn.Linear(hidden_size*3//4, hidden_size//2)
            self.linear4 = nn.Linear(hidden_size//2, hidden_size//4)

            self.linear5 = nn.Linear(hidden_size, hidden_size*3//4)
            self.linear6 = nn.Linear(hidden_size*3//4, hidden_size//2)
            self.linear7 = nn.Linear(hidden_size//2, hidden_size//4)

            self.linear8 = nn.Linear(hidden_size, hidden_size*3//4)
            self.linear9 = nn.Linear(hidden_size*3//4, hidden_size//2)
            self.linear10 = nn.Linear(hidden_size//2, hidden_size//4)
            self.linear11 = nn.Linear(hidden_size//4, num_classes)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()
            self.loss = nn.NLLLoss()
    def forward(self, x):
            x = x.view(-1, 28, 28)
            x_u, x_d = torch.split(x, 14, dim=1)
            x_ul, x_ur = torch.split(x_u, 14, dim=2)
            x_dl, x_dr = torch.split(x_d, 14, dim=2)

            x_ulu, x_uld = torch.split(x_ul, 7, dim=1)
            x_11, x_12 = torch.split(x_ulu, 7, dim=2)
            x_21, x_22 = torch.split(x_uld, 7, dim=2)

            x_uru, x_urd = torch.split(x_ur, 7, dim=1)
            x_13, x_14 = torch.split(x_uru, 7, dim=2)
            x_23, x_24 = torch.split(x_urd, 7, dim=2)

            x_dlu, x_dld = torch.split(x_dl, 7, dim=1)
            x_31, x_32 = torch.split(x_dlu, 7, dim=2)
            x_41, x_42 = torch.split(x_dld, 7, dim=2)

            x_dru, x_drd = torch.split(x_dr, 7, dim=1)
            x_33, x_34 = torch.split(x_dru, 7, dim=2)
            x_43, x_44 = torch.split(x_drd, 7, dim=2)

            x_11 = x_11.contiguous().view(-1, 7 * 7)
            x_12 = x_12.contiguous().view(-1, 7 * 7)
            x_13 = x_13.contiguous().view(-1, 7 * 7)
            x_14 = x_14.contiguous().view(-1, 7 * 7)
            x_21 = x_21.contiguous().view(-1, 7 * 7)
            x_22 = x_22.contiguous().view(-1, 7 * 7)
            x_23 = x_23.contiguous().view(-1, 7 * 7)
            x_24 = x_24.contiguous().view(-1, 7 * 7)
            x_31 = x_31.contiguous().view(-1, 7 * 7)
            x_32 = x_32.contiguous().view(-1, 7 * 7)
            x_33 = x_33.contiguous().view(-1, 7 * 7)
            x_34 = x_34.contiguous().view(-1, 7 * 7)
            x_41 = x_41.contiguous().view(-1, 7 * 7)
            x_42 = x_42.contiguous().view(-1, 7 * 7)
            x_43 = x_43.contiguous().view(-1, 7 * 7)
            x_44 = x_44.contiguous().view(-1, 7 * 7)

            out11 = self.linear1(x_11)
            out11 = self.relu(out11)
            out11 = self.linear2(out11)
            out11 = self.relu(out11)
            out11 = self.linear3(out11)
            out11 = self.relu(out11)
            out11 = self.linear4(out11)
            out11 = self.relu(out11)

            out12 = self.linear1(x_12)
            out12 = self.relu(out12)
            out12 = self.linear2(out12)
            out12 = self.relu(out12)
            out12 = self.linear3(out12)
            out12 = self.relu(out12)
            out12 = self.linear4(out12)
            out12 = self.relu(out12)

            out13 = self.linear1(x_13)
            out13 = self.relu(out13)
            out13 = self.linear2(out13)
            out13 = self.relu(out13)
            out13 = self.linear3(out13)
            out13 = self.relu(out13)
            out13 = self.linear4(out13)
            out13 = self.relu(out13)

            out14 = self.linear1(x_14)
            out14 = self.relu(out14)
            out14 = self.linear2(out14)
            out14 = self.relu(out14)
            out14 = self.linear3(out14)
            out14 = self.relu(out14)
            out14 = self.linear4(out14)
            out14 = self.relu(out14)

            out21 = self.linear1(x_21)
            out21 = self.relu(out21)
            out21 = self.linear2(out21)
            out21 = self.relu(out21)
            out21 = self.linear3(out21)
            out21 = self.relu(out21)
            out21 = self.linear4(out21)
            out21 = self.relu(out21)

            out22 = self.linear1(x_22)
            out22 = self.relu(out22)
            out22 = self.linear2(out22)
            out22 = self.relu(out22)
            out22 = self.linear3(out22)
            out22 = self.relu(out22)
            out22 = self.linear4(out22)
            out22 = self.relu(out22)

            out23 = self.linear1(x_23)
            out23 = self.relu(out23)
            out23 = self.linear2(out23)
            out23 = self.relu(out23)
            out23 = self.linear3(out23)
            out23 = self.relu(out23)
            out23 = self.linear4(out23)
            out23 = self.relu(out23)

            out24 = self.linear1(x_24)
            out24 = self.relu(out24)
            out24 = self.linear2(out24)
            out24 = self.relu(out24)
            out24 = self.linear3(out24)
            out24 = self.relu(out24)
            out24 = self.linear4(out24)
            out24 = self.relu(out24)

            out31 = self.linear1(x_31)
            out31 = self.relu(out31)
            out31 = self.linear2(out31)
            out31 = self.relu(out31)
            out31 = self.linear3(out31)
            out31 = self.relu(out31)
            out31 = self.linear4(out31)
            out31 = self.relu(out31)

            out32 = self.linear1(x_32)
            out32 = self.relu(out32)
            out32 = self.linear2(out32)
            out32 = self.relu(out32)
            out32 = self.linear3(out32)
            out32 = self.relu(out32)
            out32 = self.linear4(out32)
            out32 = self.relu(out32)

            out33 = self.linear1(x_33)
            out33 = self.relu(out33)
            out33 = self.linear2(out33)
            out33 = self.relu(out33)
            out33 = self.linear3(out33)
            out33 = self.relu(out33)
            out33 = self.linear4(out33)
            out33 = self.relu(out33)

            out34 = self.linear1(x_34)
            out34 = self.relu(out34)
            out34 = self.linear2(out34)
            out34 = self.relu(out34)
            out34 = self.linear3(out34)
            out34 = self.relu(out34)
            out34 = self.linear4(out34)
            out34 = self.relu(out34)

            out41 = self.linear1(x_41)
            out41 = self.relu(out41)
            out41 = self.linear2(out41)
            out41 = self.relu(out41)
            out41 = self.linear3(out41)
            out41 = self.relu(out41)
            out41 = self.linear4(out41)
            out41 = self.relu(out41)

            out42 = self.linear1(x_42)
            out42 = self.relu(out42)
            out42 = self.linear2(out42)
            out42 = self.relu(out42)
            out42 = self.linear3(out42)
            out42 = self.relu(out42)
            out42 = self.linear4(out42)
            out42 = self.relu(out42)

            out43 = self.linear1(x_43)
            out43 = self.relu(out43)
            out43 = self.linear2(out43)
            out43 = self.relu(out43)
            out43 = self.linear3(out43)
            out43 = self.relu(out43)
            out43 = self.linear4(out43)
            out43 = self.relu(out43)

            out44 = self.linear1(x_44)
            out44 = self.relu(out44)
            out44 = self.linear2(out44)
            out44 = self.relu(out44)
            out44 = self.linear3(out44)
            out44 = self.relu(out44)
            out44 = self.linear4(out44)
            out44 = self.relu(out44)

            # out = torch.cat((torch.cat((out1, out2),1), torch.cat((out3, out4),1)), 1)
            x_ul = torch.cat((out11, out12, out21, out22), 1)
            x_ur = torch.cat((out13, out14, out23, out24), 1)
            x_dl = torch.cat((out31, out32, out41, out42), 1)
            x_dr = torch.cat((out33, out34, out43, out44), 1)

            out1 = self.linear5(x_ul)
            out1 = self.relu(out1)
            out1 = self.linear6(out1)
            out1 = self.relu(out1)
            out1 = self.linear7(out1)
            out1 = self.relu(out1)

            out2 = self.linear5(x_ur)
            out2 = self.relu(out2)
            out2 = self.linear6(out2)
            out2 = self.relu(out2)
            out2 = self.linear7(out2)
            out2 = self.relu(out2)

            out3 = self.linear5(x_dl)
            out3 = self.relu(out3)
            out3 = self.linear6(out3)
            out3 = self.relu(out3)
            out3 = self.linear7(out3)
            out3 = self.relu(out3)

            out4 = self.linear5(x_dr)
            out4 = self.relu(out4)
            out4 = self.linear6(out4)
            out4 = self.relu(out4)
            out4 = self.linear7(out4)
            out4 = self.relu(out4)

            # out = torch.cat((torch.cat((out1, out2),1), torch.cat((out3, out4),1)), 1)
            out = torch.cat((out1, out2, out3, out4), 1)
            out = self.linear8(out)
            out = self.relu(out)
            out = self.linear9(out)
            out = self.relu(out)
            out = self.linear10(out)
            out = self.relu(out)
            out = self.linear11(out)
            out = Func.log_softmax(out)

            return out

class model6(nn.Module):
    def __init__(self, input_size, num_classes):
            super(model6, self).__init__()

            self.name = 'Model 6 - Split image-16 36'
            hidden_size = 18

            self.linear1 = nn.Linear(input_size // 16, hidden_size)
            self.linear2 = nn.Linear(16 * (hidden_size), 8 * (hidden_size))
            self.linear3 = nn.Linear(8 * (hidden_size), 4 * (hidden_size ))
            self.linear4 = nn.Linear(4 * (hidden_size), 2 * (hidden_size))
            self.linear5 = nn.Linear(2 * (hidden_size ), hidden_size )
            self.linear6 = nn.Linear(hidden_size, num_classes)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()
            self.loss = nn.NLLLoss()

    def forward(self, x):
            x = x.view(-1, 28, 28)
            x_u, x_d = torch.split(x, 14, dim=1)
            x_ul, x_ur = torch.split(x_u, 14, dim=2)
            x_dl, x_dr = torch.split(x_d, 14, dim=2)

            x_ulu, x_uld = torch.split(x_ul, 7, dim=1)
            x_11, x_12 = torch.split(x_ulu, 7, dim=2)
            x_21, x_22 = torch.split(x_uld, 7, dim=2)

            x_uru, x_urd = torch.split(x_ur, 7, dim=1)
            x_13, x_14 = torch.split(x_uru, 7, dim=2)
            x_23, x_24 = torch.split(x_urd, 7, dim=2)

            x_dlu, x_dld = torch.split(x_dl, 7, dim=1)
            x_31, x_32 = torch.split(x_dlu, 7, dim=2)
            x_41, x_42 = torch.split(x_dld, 7, dim=2)

            x_dru, x_drd = torch.split(x_dr, 7, dim=1)
            x_33, x_34 = torch.split(x_dru, 7, dim=2)
            x_43, x_44 = torch.split(x_drd, 7, dim=2)

            x_11 = x_11.contiguous().view(-1, 7 * 7)
            x_12 = x_12.contiguous().view(-1, 7 * 7)
            x_13 = x_13.contiguous().view(-1, 7 * 7)
            x_14 = x_14.contiguous().view(-1, 7 * 7)
            x_21 = x_21.contiguous().view(-1, 7 * 7)
            x_22 = x_22.contiguous().view(-1, 7 * 7)
            x_23 = x_23.contiguous().view(-1, 7 * 7)
            x_24 = x_24.contiguous().view(-1, 7 * 7)
            x_31 = x_31.contiguous().view(-1, 7 * 7)
            x_32 = x_32.contiguous().view(-1, 7 * 7)
            x_33 = x_33.contiguous().view(-1, 7 * 7)
            x_34 = x_34.contiguous().view(-1, 7 * 7)
            x_41 = x_41.contiguous().view(-1, 7 * 7)
            x_42 = x_42.contiguous().view(-1, 7 * 7)
            x_43 = x_43.contiguous().view(-1, 7 * 7)
            x_44 = x_44.contiguous().view(-1, 7 * 7)

            out11 = self.linear1(x_11)
            out11 = self.relu(out11)

            out12 = self.linear1(x_12)
            out12 = self.relu(out12)

            out13 = self.linear1(x_13)
            out13 = self.relu(out13)

            out14 = self.linear1(x_14)
            out14 = self.relu(out14)

            out21 = self.linear1(x_21)
            out21 = self.relu(out21)

            out22 = self.linear1(x_22)
            out22 = self.relu(out22)

            out23 = self.linear1(x_23)
            out23 = self.relu(out23)

            out24 = self.linear1(x_24)
            out24 = self.relu(out24)

            out31 = self.linear1(x_31)
            out31 = self.relu(out31)

            out32 = self.linear1(x_32)
            out32 = self.relu(out32)

            out33 = self.linear1(x_33)
            out33 = self.relu(out33)

            out34 = self.linear1(x_34)
            out34 = self.relu(out34)

            out41 = self.linear1(x_41)
            out41 = self.relu(out41)

            out42 = self.linear1(x_42)
            out42 = self.relu(out42)

            out43 = self.linear1(x_43)
            out43 = self.relu(out43)

            out44 = self.linear1(x_44)
            out44 = self.relu(out44)

            # out = torch.cat((torch.cat((out1, out2),1), torch.cat((out3, out4),1)), 1)
            out = torch.cat((out11, out12, out13, out14, out21, out22, out23, out24, out31, out32, out33, out34, out41, out42, out43, out44), 1)
            out = self.linear2(out)
            out = self.relu(out)
            out = self.linear3(out)
            out = self.relu(out)
            out = self.linear4(out)
            out = self.relu(out)
            out = self.linear5(out)
            out = self.relu(out)
            out = self.linear6(out)
            out = Func.log_softmax(out)
            return out


class model7(nn.Module):
    def __init__(self, input_size, num_classes):
            super(model7, self).__init__()

            self.name = 'Model 7 - Split image-16'
            hidden_size = 285

            self.linear1 = nn.Linear(input_size // 16, hidden_size)
            self.linear6 = nn.Linear(16*hidden_size, num_classes)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout()
            self.loss = nn.NLLLoss()

    def forward(self, x):
            x = x.view(-1, 28, 28)
            x_u, x_d = torch.split(x, 14, dim=1)
            x_ul, x_ur = torch.split(x_u, 14, dim=2)
            x_dl, x_dr = torch.split(x_d, 14, dim=2)

            x_ulu, x_uld = torch.split(x_ul, 7, dim=1)
            x_11, x_12 = torch.split(x_ulu, 7, dim=2)
            x_21, x_22 = torch.split(x_uld, 7, dim=2)

            x_uru, x_urd = torch.split(x_ur, 7, dim=1)
            x_13, x_14 = torch.split(x_uru, 7, dim=2)
            x_23, x_24 = torch.split(x_urd, 7, dim=2)

            x_dlu, x_dld = torch.split(x_dl, 7, dim=1)
            x_31, x_32 = torch.split(x_dlu, 7, dim=2)
            x_41, x_42 = torch.split(x_dld, 7, dim=2)

            x_dru, x_drd = torch.split(x_dr, 7, dim=1)
            x_33, x_34 = torch.split(x_dru, 7, dim=2)
            x_43, x_44 = torch.split(x_drd, 7, dim=2)

            x_11 = x_11.contiguous().view(-1, 7 * 7)
            x_12 = x_12.contiguous().view(-1, 7 * 7)
            x_13 = x_13.contiguous().view(-1, 7 * 7)
            x_14 = x_14.contiguous().view(-1, 7 * 7)
            x_21 = x_21.contiguous().view(-1, 7 * 7)
            x_22 = x_22.contiguous().view(-1, 7 * 7)
            x_23 = x_23.contiguous().view(-1, 7 * 7)
            x_24 = x_24.contiguous().view(-1, 7 * 7)
            x_31 = x_31.contiguous().view(-1, 7 * 7)
            x_32 = x_32.contiguous().view(-1, 7 * 7)
            x_33 = x_33.contiguous().view(-1, 7 * 7)
            x_34 = x_34.contiguous().view(-1, 7 * 7)
            x_41 = x_41.contiguous().view(-1, 7 * 7)
            x_42 = x_42.contiguous().view(-1, 7 * 7)
            x_43 = x_43.contiguous().view(-1, 7 * 7)
            x_44 = x_44.contiguous().view(-1, 7 * 7)

            out11 = self.linear1(x_11)
            out11 = self.relu(out11)

            out12 = self.linear1(x_12)
            out12 = self.relu(out12)

            out13 = self.linear1(x_13)
            out13 = self.relu(out13)

            out14 = self.linear1(x_14)
            out14 = self.relu(out14)

            out21 = self.linear1(x_21)
            out21 = self.relu(out21)

            out22 = self.linear1(x_22)
            out22 = self.relu(out22)

            out23 = self.linear1(x_23)
            out23 = self.relu(out23)

            out24 = self.linear1(x_24)
            out24 = self.relu(out24)

            out31 = self.linear1(x_31)
            out31 = self.relu(out31)

            out32 = self.linear1(x_32)
            out32 = self.relu(out32)

            out33 = self.linear1(x_33)
            out33 = self.relu(out33)

            out34 = self.linear1(x_34)
            out34 = self.relu(out34)

            out41 = self.linear1(x_41)
            out41 = self.relu(out41)

            out42 = self.linear1(x_42)
            out42 = self.relu(out42)

            out43 = self.linear1(x_43)
            out43 = self.relu(out43)

            out44 = self.linear1(x_44)
            out44 = self.relu(out44)

            out = torch.cat((out11, out12, out13, out14, out21, out22, out23, out24, out31, out32, out33, out34, out41, out42, out43, out44), 1)
            out = self.linear6(out)
            out = Func.log_softmax(out)
            return out