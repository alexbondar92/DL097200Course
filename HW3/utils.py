import os
import matplotlib.pyplot as plt
import torch

import models

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path='./data'):
        self.dictionary = Dictionary()
        # self.train = os.path.join(path, 'train.txt')
        # self.test = os.path.join(path, 'test.txt')

    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)

def use_cuda(x):
    if torch.cuda.is_available():
        x=x.cuda()
    return x

def fig_plot(results, export_plot=False):
    # Plot results
    x = range(1, 1+len(results['Train loss']))
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x, results['Train loss'], label='Training-set')
    ax[0].plot(x, results['Valid loss'], label='Validation-set')
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='best')
    ax[0].grid(axis='both', which='both')
    ax[0].set_title('Loss vs. Epochs')
    ax[1].plot(x, results['Train perplexity'], label='Training-set')
    ax[1].plot(x, results['Valid perplexity'], label='Validation-set')
    ax[1].set_ylabel('Perplexity')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(loc='best')
    ax[1].grid(axis='both', which='both')
    ax[1].set_title('Perplexity vs. Epochs')

    fig.get_axes()[0].annotate(results['Name'], (0.5, 0.95),
                               xycoords='figure fraction', ha='center',
                               fontsize=16
                               )

    if export_plot:
        fig.savefig('./results/' + results['Name']+'.png')

    return fig

def save_checkpoint(model, optimizer, scheduler, filepath):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict()
    }
    torch.save(state, filepath)

def load_checkpoint(model, optimizer, scheduler, filepath):
    # "lambda" allows to load the model on cpu in case it is saved on gpu
    state = torch.load(filepath, lambda storage, loc: storage)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['lr_scheduler'])

    return model, optimizer, scheduler

def initialize_model(model_num, vocab_size, embed_size):
    if model_num == 1:
        return models.model1(vocab_size, embed_size)
    elif model_num == 2:
        return models.model2(vocab_size, embed_size)
    elif model_num == 3:
        return models.model3(vocab_size, embed_size)
    elif model_num == 4:
        return models.model4(vocab_size, embed_size)
    elif model_num == 5:
        return models.model5(vocab_size, embed_size)
    elif model_num == 6:
        return models.model6(vocab_size, embed_size)
    elif model_num == 7:
        return models.model7(vocab_size, embed_size)
