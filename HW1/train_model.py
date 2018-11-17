import numpy as np
import matplotlib.pyplot as plt
import torch
from time import time

import datasets
import models
import utils

#TODO: Run validation to the models
#TODO: Train the best model on the whole training dataset
#TODO: save model parameters

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 50
batch_size = [16, 32, 64, 128, 256]
learning_rate = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

# Datasets
train_dataset = datasets.train_dataset()
validation_dataset = datasets.validation_dataset()
test_dataset = datasets.test_dataset()

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size[3],
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size[3],
                                                shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size[3],
                                          shuffle=False)

# Define the model of the network
model = models.model1(input_size, num_classes)

if torch.cuda.is_available():
    print('GPU detected - Enabling Cuda!')
    model = model.cuda()
else:
    print('No GPU detected!')

# Loss function
criterion = model.loss

# Optimizing method
# TODO: Adjust lr and other parameters for validation
optimizers = {'ADAM': torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5),
             'SGD': torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)}
optimizer = optimizers['ADAM']

print('Model\'s number of parameters = %d'
      %(sum(p.numel() for p in model.parameters())))

# Train the Model

train_loss_log = np.zeros(num_epochs)
train_error_log = np.zeros(num_epochs)
validation_loss_log = np.zeros(num_epochs)
validation_error_log = np.zeros(num_epochs)

for epoch in range(num_epochs):

    start_time = time()

    # Initialize errors and losses
    epoch_train_loss = 0.0
    epoch_train_error = 0.0
    epoch_validation_loss = 0.0
    epoch_validation_error = 0.0

    model.train()
    for images, labels in train_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        images = images.view(-1, 28*28)

        # Forward + Backward + Optimize
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_train_loss += outputs.shape[0] * loss.item()

        # Accumulate error
        _, predictions = torch.max(outputs.data, 1)
        epoch_train_error += (predictions != labels).sum()

    # Mean error and loss
    epoch_train_loss = epoch_train_loss / len(train_dataset)
    epoch_train_error = epoch_train_error.type(torch.FloatTensor) / len(train_dataset) * 100
    train_loss_log[epoch] = epoch_train_loss
    train_error_log[epoch] = epoch_train_error

    # Validation
    model.eval()
    for images, labels in validation_loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        images = images.view(-1, 28*28)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Accumulate loss
        epoch_validation_loss += outputs.shape[0] * loss.item()

        # Accumulate error
        _, predictions = torch.max(outputs.data, 1)
        epoch_validation_error += (predictions != labels).sum()

    # Mean error and loss
    epoch_validation_loss = epoch_validation_loss / len(validation_dataset)
    epoch_validation_error = epoch_validation_error.type(torch.FloatTensor) / len(validation_dataset) * 100
    validation_loss_log[epoch] = epoch_validation_loss
    validation_error_log[epoch] = epoch_validation_error

    duration = time() - start_time

    print('Epoch [{}/{}], Time: {:.1f} [s], Training speed: {:.2f} [images/s]'.format(epoch + 1, num_epochs, duration,
                                                                                     len(train_dataset) / duration))
    print('Train loss: {:.4f}, Train error: {:.2f}%'.format(epoch_train_loss, epoch_train_error))
    print('Validation loss: {:.4f}, Validation error: {:.2f}%'.format(epoch_validation_loss, epoch_validation_error))



# Test the Model
correct = 0
total = 0

for images, labels in test_loader:
    # Convert the images and labels to cuda
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

    images = images.view(-1,28*28)

    # Change the model to prediction mode
    model.eval()

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the model on the 10000 test images: %d %%'
      % (100*correct / total))

# Export the results to .npy file
results = {'Name': model.name, 'Train loss': train_loss_log, 'Validation loss': validation_loss_log,
           'Train error': train_error_log, 'Validation error': validation_error_log}
np.save('./results/' + model.name + '.npy', results)

# Plot the results
fig = utils.fig_plot(results, export_plot=True)
plt.show()