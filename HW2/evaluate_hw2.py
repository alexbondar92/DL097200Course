import torch
import torch.optim.lr_scheduler as LRscheduler
import datasets
import utils

# Parameters
input_size = 784
num_classes = 10

# Model you wish to evaluate
file_path = r'./saved models/Model 5 - DenseNet, lr=0.001, ss=1, gm=0.1 , bs=64.pkl'
model_name = file_path.split('/')[1]
model_name = model_name.split('.pkl')[0]

state = torch.load(file_path,lambda storage, loc: storage)

model = utils.initialize_model(3)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Decay LR by a factor of 0.1 every 7 epochs
# lr_scheduler = LRscheduler.StepLR(optimizer, step_size=ss, gamma=gm)
# TODO: CHECK OTHER TYPES OF SCHEDULER!!  https://pytorch.org/docs/stable/optim.html
lr_scheduler = LRscheduler.ReduceLROnPlateau(optimizer, eps=1e-3)
model, optimizer = utils.load_checkpoint(model, optimizer, lr_scheduler, file_path)


if torch.cuda.is_available():
    print('GPU detected - Enabling Cuda!')
    model = model.cuda()
else:
    print('No GPU detected!')

# Dataset
test_dataset = datasets.test_dataset()

# Dataset Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=False)
# Test the Model
correct = 0
total = 0

for images, labels in test_loader:
    # Convert the images and labels to cuda
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()


    # Change the model to prediction mode
    model.eval()

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()


print('Accuracy of the model - {} on the test set: {:.2f}'.format(model_name, 100 * float(correct)/total))
