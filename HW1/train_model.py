import numpy as np
import matplotlib.pyplot as plt
import torch
from time import time
import os

import datasets
import models
import utils

#TODO: Change the mode - Alon 1, Bar 2, Alex 3
mode = 1
save_dir = './saved models/'

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100

batch_size = [32, 64, 128]
learning_rate = [1e-2, 1e-3, 1e-4]
weight_decay = [1e-4,1e-5, 1e-6]

# Datasets
train_dataset = datasets.train_dataset()
validation_dataset = datasets.validation_dataset()


# Define the model of the network
models = [models.model1(input_size, num_classes), models.model2(input_size, num_classes), models.model3(input_size, num_classes),
          models.model4(input_size, num_classes), models.model5(input_size, num_classes), models.model6(input_size, num_classes),
          models.model7(input_size, num_classes)]

if mode == 1:
    models = models[0:2]
elif mode == 2:
    models = models[2:5]
elif mode == 3:
    models = models[5:]

for model in models:

    if torch.cuda.is_available():
        print('GPU detected - Enabling Cuda!')
        model = model.cuda()
    else:
        print('No GPU detected!')

    # Calculate number of model parameters
    print('Model: {}, number of parameters = {}'.format(model.name, sum(p.numel() for p in model.parameters())))

    for bs in batch_size:

        # Dataset Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=bs,
                                                   shuffle=True)

        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                        batch_size=bs,
                                                        shuffle=False)
        for lr in learning_rate:
            for wd in weight_decay:

                run_name = "{}, lr={}, wd={}, bs={}".format(model.name, lr, wd, bs)
                file_path = os.path.join(save_dir,run_name + '.pkl')

                criterion = model.loss
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                # Load model parameters and optimizer condition if available
                if os.path.exists(file_path):
                    model, optimizer = utils.load_checkpoint(model, optimizer, file_path)

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

                        images = images.view(-1, 28 * 28)

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

                        images = images.view(-1, 28 * 28)

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

                    print('Model: {}, Epoch [{}/{}], Time: {:.1f} [s], Training speed: {:.2f} [images/s]'.format(run_name, epoch+1 , num_epochs,
                                                                                                                 duration, len(train_dataset) / duration))
                    print('Train loss: {:.4f}, Train error: {:.2f}%'.format(epoch_train_loss, epoch_train_error))
                    print('Validation loss: {:.4f}, Validation error: {:.2f}%'.format(epoch_validation_loss,
                                                                                      epoch_validation_error))
                # Save model weights optimizer state and current epoch
                print('Saving model - {}'.format(run_name))
                utils.save_checkpoint(model, optimizer, file_path)

                # Export the results to .npy file
                results = {'Name': run_name, 'Train loss': train_loss_log, 'Validation loss': validation_loss_log,
                           'Train error': train_error_log, 'Validation error': validation_error_log}
                np.save('./results/' + run_name + '.npy', results)

                # Plot the results
                fig = utils.fig_plot(results, export_plot=True)

                print('Done!')

plt.show()