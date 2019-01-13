import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as LRscheduler
from time import time
import os

import utils
import models

# Hyper Parameters
embed_size = 85
num_epochs = 150
num_samples = 1000  # number of words to be sampled
temperature = 1.0
batch_size = [20]
seq_lengths = [30]
learning_rate = [0.001]

#TODO: CHANGE THE MODEL NUMBER TO THE ONE YOU NEED
model_num = 4

# Load Penn Treebank Dataset
train_path = './data/train.txt'
valid_path = './data/valid.txt'
sample_path = './sample.txt'
save_dir = './saved models/'

for bs in batch_size:
    # Create corpuses
    train_corpus = utils.Corpus()
    ids_train = train_corpus.get_data(train_path, bs)
    ids_valid = train_corpus.get_data(valid_path, bs)
    train_vocab_size = len(train_corpus.dictionary)

    for seq_len in seq_lengths:
        num_train_batches = ids_train.size(1) // seq_len
        num_valid_batches = ids_valid.size(1) // seq_len

        for lr in learning_rate:
            model = utils.initialize_model(model_num, train_vocab_size, embed_size)
            model = utils.use_cuda(model)
            print('Training vocabulary size: {}'.format(train_vocab_size))
            print('Model: {}'.format(model.name))
            print('Number of parameters = {}'.format(sum(p.numel() for p in model.parameters())))

            run_name = "{}, seq_len={}, lr={}, bs={}".format(model.name, seq_len, lr, bs)
            file_path = os.path.join(save_dir,run_name + '.pkl')

            # Loss and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            #TODO: CHANGE PARAMETERS - EPS, PATIENCE, etc.. CHECK OTHER TYPES OF SCHEDULER!!  https://pytorch.org/docs/stable/optim.html
            lr_scheduler = LRscheduler.ReduceLROnPlateau(optimizer,eps= 1e-7)

            # Load model parameters and optimizer condition if available
            if os.path.exists(file_path):
                model, optimizer, lr_scheduler = utils.load_checkpoint(model, optimizer, lr_scheduler, file_path)

            # Train the Model
            train_loss_log = np.zeros(num_epochs)
            train_perp_log = np.zeros(num_epochs)
            valid_loss_log = np.zeros(num_epochs)
            valid_perp_log = np.zeros(num_epochs)

            for epoch in range(num_epochs):
                start_time = time()

                # Initialize loss and perplexity for each epoch
                epoch_train_loss = 0.0
                epoch_train_perp = 0.0

                # Set the model into training mode
                model.train()

                # Initial hidden and memory states
                states = (utils.use_cuda(torch.zeros(model.num_layers, bs, model.hidden_size)),
                          utils.use_cuda(torch.zeros(model.num_layers, bs, model.hidden_size)))

                for i in range(0, ids_train.size(1) - seq_len, seq_len):
                    # Get batch inputs and targets
                    inputs = utils.use_cuda(ids_train[:, i:i + seq_len])
                    targets = utils.use_cuda(ids_train[:, (i + 1):(i + 1) + seq_len].contiguous())

                    # Forward + Backward + Optimize
                    model.zero_grad()
                    states = [state.detach() for state in states]   # Truncated Back propagation
                    outputs, states = model(inputs, states)
                    train_loss = criterion(outputs, targets.view(-1))
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    # step = (i + 1) // seq_len
                    # if step % 100 == 0:
                    #     print('Epoch [%d/%d], Step[%d/%d], Momentary Loss: %.3f, Momentary Perplexity: %5.2f' %
                    #           (epoch + 1, num_epochs, step, num_train_batches, train_loss.item(), np.exp(train_loss.item())))

                  # Accumulate loss and perplexity
                    epoch_train_loss += train_loss.item()
                    epoch_train_perp += np.exp(train_loss.item())

                # Mean loss and perplexity during the epoch
                epoch_train_loss = epoch_train_loss / num_train_batches
                epoch_train_perp = epoch_train_perp / num_train_batches
                train_loss_log[epoch] = epoch_train_loss
                train_perp_log[epoch] = epoch_train_perp

                # Update the scheduler step
                lr_scheduler.step(epoch_train_loss)

                # Validation
                epoch_valid_loss = 0.0
                epoch_valid_perp = 0.0

                model.eval()

                # Initial hidden and memory states
                states = (utils.use_cuda(torch.zeros(model.num_layers, bs, model.hidden_size)),
                          utils.use_cuda(torch.zeros(model.num_layers, bs, model.hidden_size)))

                #TODO: Guy said theres no need to create a valid corpus
                # for i in range(0, ids_valid.size(1) - seq_len, seq_len):
                #     # Get batch inputs and targets
                #     inputs = utils.use_cuda(ids_valid[:, i:i + seq_len])
                #     targets = utils.use_cuda(ids_valid[:, (i + 1):(i + 1) + seq_len].contiguous())

                for i in range(0, ids_valid.size(1) - seq_len, seq_len):
                    # Get batch inputs and targets
                    inputs = utils.use_cuda(ids_valid[:, i:i + seq_len])
                    targets = utils.use_cuda(ids_valid[:, (i + 1):(i + 1) + seq_len].contiguous())

                    # Forward pass
                    # states = [state.detach() for state in states]  # Truncated Back propagation
                    outputs, states = model(inputs, states)
                    # outputs, _ = model(inputs, states)
                    valid_loss = criterion(outputs, targets.view(-1))

                    # Accumulate loss and perplexity
                    epoch_valid_loss += valid_loss.item()
                    epoch_valid_perp += np.exp(valid_loss.item())

                # Mean loss and perplexity during the epoch
                epoch_valid_loss = epoch_valid_loss / num_valid_batches
                epoch_valid_perp = epoch_valid_perp / num_valid_batches
                valid_loss_log[epoch] = epoch_valid_loss
                valid_perp_log[epoch] = epoch_valid_perp

                duration = time() - start_time

                print('Model: {}, Epoch [{}/{}], Time: {:.1f} [s]'.format(run_name,epoch + 1, num_epochs, duration))
                print('Average train loss: {:.4f}, Average train perplexity: {:.2f}'.format(epoch_train_loss, epoch_train_perp))
                print('Average valid loss: {:.4f}, Average valid perplexity: {:.2f}'.format(epoch_valid_loss, epoch_valid_perp))

            # Save model weights optimizer state and current epoch
            print('Saving model - {}'.format(run_name))
            utils.save_checkpoint(model, optimizer, lr_scheduler, file_path)

            # Export the results to .npy file
            results = {'Name': run_name, 'Train loss': train_loss_log, 'Valid loss': valid_loss_log,
                       'Train perplexity': train_perp_log, 'Valid perplexity': valid_perp_log}
            np.save('./results/' + run_name + '.npy', results)

            # Plot the results
            fig = utils.fig_plot(results, export_plot=True)

            print('Done!')

        plt.show()

# Sampling
with open(sample_path, 'w') as f:
    # Set intial hidden ane memory states
    state = (utils.use_cuda(torch.zeros(model.num_layers, 1, model.hidden_size)),
             utils.use_cuda(torch.zeros(model.num_layers, 1, model.hidden_size)))

    # Select one word id randomly
    prob = torch.ones(train_vocab_size)
    input = utils.use_cuda(torch.multinomial(prob, num_samples=1).unsqueeze(1))

    for i in range(num_samples):
        # Forward propagate rnn
        output, state = model(input, state)

        # Sample a word id
        word_weights = output.squeeze().div(temperature).exp().cpu()
        word_id = torch.multinomial(word_weights, 1).item()

        # Feed sampled word id to next time step
        input.data.fill_(word_id)

        # File write
        word = train_corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word + ' '
        f.write(word)

        if (i + 1) % 20 == 0:
            print('Sampled [%d/%d] words and save to %s' % (i + 1, num_samples, sample_path))