import torch
import os

import utils

# Hyper Parameters
embed_size = 249
num_samples = 30  # number of words to be sampled
temperature = [0.5, 1.0, 2]


# Load Penn Treebank Dataset
train_path = './data/train.txt'
valid_path = './data/valid.txt'
sample_path = './sample.txt'
save_dir = './saved models/'
model_name = 'LSTM - 249 hidden cells, 1 layers'

# Create the corpus
corpus = utils.Corpus()
ids_train = corpus.get_data(train_path)
ids_valid = corpus.get_data(valid_path)
vocab_size = len(corpus.dictionary)

# Load best model
model_num = 4
model = utils.initialize_model(model_num, vocab_size, embed_size)
model = utils.use_cuda(model)
file_path = os.path.join(save_dir, model_name + '.pkl')
load_state = torch.load(file_path, lambda storage, loc: storage)
model.load_state_dict(load_state['state_dict'])
model.eval()    # Turn to eval mode - so there won't be any dropouts!

# Sampling
with open(sample_path, 'w') as f:
    for t in temperature:

        f.write('Sentence wit temperature = {}:\n'.format(t))

        # Set initial hidden and memory states
        state = (utils.use_cuda(torch.zeros(model.num_layers, 1, model.hidden_size)),
                 utils.use_cuda(torch.zeros(model.num_layers, 1, model.hidden_size)))

        # Pre-defined words to generate a sentence
        words_input = torch.tensor([corpus.dictionary.word2idx['buy'], corpus.dictionary.word2idx['low'],
                                    corpus.dictionary.word2idx['cell'], corpus.dictionary.word2idx['high'],
                                    corpus.dictionary.word2idx['is'], corpus.dictionary.word2idx['the']])

        # Sample words
        for i in range(num_samples):
            if i <= 5:
                # Get next pre-defined word
                if i == 0:
                    input = utils.use_cuda(torch.tensor([[words_input[i]]]))
                else:
                    input.data.fill_(words_input[i])

                word_id = words_input[i].item()
            else:
                # Sample a word id
                word_weights = output.squeeze().div(t).exp().cpu()
                word_id = torch.multinomial(word_weights, 1).item()

                # Feed sampled word id to next time step
                input.data.fill_(word_id)

            # Forward propagate RNN
            output, state = model(input, state)

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

        f.write('\n\n')
