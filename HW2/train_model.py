import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim.lr_scheduler as LRscheduler
from time import time
import os

import datasets
import utils

save_dir = './saved models/'

#TODO: CHANGE THE MODEL NUMBER TO THE ONE YOU NEED
model_num = 1

# Hyper Parameters
num_epochs = 10 #TODO: I THINK WE NEED TO GO UP TO SEVERAL THOUSANDS (~5K)...

#TODO: TO-BE-DECIDED
batch_size = [256] #[256, 128]
learning_rate = [1e-3] #[1e-3, 1e-4]

# Datasets
#TODO: NOTICE THAT YOU SHOULD PASS THE TRANSFORMATION YOU WANT TO APPLY TO THE TRAINING DATASET (1-4)!!!
train_dataset = datasets.train_dataset()
test_dataset = datasets.test_dataset()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


for bs in batch_size:
    # Dataset Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs,
                                              shuffle=False)

    for lr in learning_rate:
        model = utils.initialize_model(model_num)

        if torch.cuda.is_available():
            print('GPU detected - Enabling Cuda!')
            model = model.cuda()
        else:
            print('No GPU detected!')

        # Calculate number of model parameters
        print('Model: {}, number of parameters = {}'.format(model.name, sum(p.numel() for p in model.parameters())))

        run_name = "{}, lr={}, bs={}".format(model.name, lr, bs)
        file_path = os.path.join(save_dir,run_name + '.pkl')

        criterion = model.loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #TODO: CHECK OTHER TYPES OF SCHEDULER!!  https://pytorch.org/docs/stable/optim.html
        lr_scheduler = LRscheduler.ReduceLROnPlateau(optimizer)

        # Load model parameters and optimizer condition if available
        if os.path.exists(file_path):
            model, optimizer, lr_scheduler = utils.load_checkpoint(model, optimizer, lr_scheduler, file_path)


        # Train the Model
        train_loss_log = np.zeros(num_epochs)
        train_error_log = np.zeros(num_epochs)
        test_loss_log = np.zeros(num_epochs)
        test_error_log = np.zeros(num_epochs)

        for epoch in range(num_epochs):

            start_time = time()

            # Initialize errors and losses
            epoch_train_loss = 0.0
            epoch_train_error = 0.0
            epoch_test_loss = 0.0
            epoch_test_error = 0.0

            model.train()
            for images, labels in train_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

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

            # Update the scheduler step
            lr_scheduler.step(epoch_train_loss)

            # Validation
            model.eval()
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate loss
                epoch_test_loss += outputs.shape[0] * loss.item()

                # Accumulate error
                _, predictions = torch.max(outputs.data, 1)
                epoch_test_error += (predictions != labels).sum()

            # Mean error and loss
            epoch_test_loss = epoch_test_loss / len(test_dataset)
            epoch_test_error = epoch_test_error.type(torch.FloatTensor) / len(test_dataset) * 100
            test_loss_log[epoch] = epoch_test_loss
            test_error_log[epoch] = epoch_test_error
            duration = time() - start_time

            print('Model: {}, Epoch [{}/{}], Time: {:.1f} [s], Training speed: {:.2f} [images/s]'.format(run_name, epoch+1 , num_epochs,
                                                                                                         duration, len(train_dataset) / duration))
            print('Train loss: {:.4f}, Train error: {:.2f}%'.format(epoch_train_loss, epoch_train_error))
            print('Test loss: {:.4f}, Test error: {:.2f}%'.format(epoch_test_loss, epoch_test_error))

        # Save model weights optimizer state and current epoch
        print('Saving model - {}'.format(run_name))
        utils.save_checkpoint(model, optimizer, lr_scheduler, file_path)

        # Export the results to .npy file
        results = {'Name': run_name, 'Train loss': train_loss_log, 'Test loss': test_loss_log,
                   'Train error': train_error_log, 'Test error': test_error_log}
        np.save('./results/' + run_name + '.npy', results)

        # Plot the results
        fig = utils.fig_plot(results, export_plot=True)

        print('Done!')

plt.show()