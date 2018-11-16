import torch
import torch.nn as nn
import torch.nn.functional as Func
import torchvision
import torchvision.datasets as DSet
import torchvision.transforms as transforms

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.0015
hidden_size = 78

transform = transforms.Compose(
         [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = DSet.MNIST(root='./data',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = DSet.MNIST(root='./data',
                            train=False,
                            transform=transform)

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

# Model
class LogisticRegration(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegration, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size//2)
        self.linear3 = nn.Linear(hidden_size//2, num_classes)

        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        out = self.linear1(x)
        out = self.leakyRelu(out)
        out = self.linear2(out)
        out = self.leakyRelu(out)
        out = self.linear3(out)
        out = self.dropout(out)
        out = Func.log_softmax(out)
        return out


model = LogisticRegration(input_size, num_classes)


if torch.cuda.is_available():
    print('===============Cuda works!!')
    model = model.cuda()

# Loss and Optimizer
# Softmax is internally computed
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

print('============== Model\'s number of parameters = %d'
      %(sum(p.numel() for p in model.parameters())))

# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28*28)
        labels = labels

        # Forward + Backward + Optimize
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i+1, len(train_loader), loss.data[0]))
#                 % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, loss.data[0]))

# Testing the Model

correct = 0
total = 0

for images, labels in test_loader:
    images = images.view(-1,28*28)
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the model on the 10000 test images: %d %%'
      % (100*correct / total))

