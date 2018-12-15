#TODO: THIS FILE NEEDS TO BE UPDATED!!

import torch

import datasets
import models

# Parameters
input_size = 784
num_classes = 10

# Model you wish to evaluate
file_path = r'./saved models/best model/Model 4 - Split image-16, lr=0.001, wd=0.0001, bs=64.pkl'
model_name = file_path.split('best model/')[1]
model_name = model_name.split('.pkl')[0]

state = torch.load(file_path,lambda storage, loc: storage)
model = models.model4(input_size, num_classes)
model.load_state_dict(state)


if torch.cuda.is_available():
    print('GPU detected - Enabling Cuda!')
    model = model.cuda()
else:
    print('No GPU detected!')

# Dataset
test_dataset = datasets.test_dataset()

# Dataset Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=256,
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

print('Accuracy of the model - {} on the test set: {}'.format(model_name, 100 * int(correct) / total))