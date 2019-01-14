import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.nn.init as init





class model1(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(model1, self).__init__()
        # Parameters
        self.hidden_size = 122
        self.num_layers = 4
        self.type = 'LSTM'
        self.name = '{} - {} hidden cells, {} layers'.format(self.type, str(self.hidden_size), str(self.num_layers))

        # Word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Structure
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.num_layers, dropout=0.55, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate RNN
        out, h = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time step
        out = self.linear(out)
        return out, h

class model2(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(model2, self).__init__()
        # Parameters
        self.hidden_size = 130
        self.num_layers = 3
        self.type = 'LSTM'
        self.name = '{} - {} hidden cells, {} layers'.format(self.type, str(self.hidden_size), str(self.num_layers))

        # Word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Structure
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate RNN
        out, h = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time step
        out = self.linear(out)
        return out, h

class model3(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(model3, self).__init__()
        # Parameters
        self.hidden_size = 140
        self.num_layers = 2
        self.type = 'LSTM'
        self.name = '{} - {} hidden cells, {} layers'.format(self.type, str(self.hidden_size), str(self.num_layers))

        # Word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Structure
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate RNN
        out, h = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time step
        out = self.linear(out)
        return out, h


class model4(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(model4, self).__init__()
        # Parameters
        self.hidden_size = embed_size
        self.num_layers = 1
        self.type = 'LSTM'
        self.name = '{} - {} hidden cells, {} layers - 50'.format(self.type, str(self.hidden_size), str(self.num_layers))

        # Word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Structure
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.drop = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.linear.weight = self.embed.weight
        self.init_weights()


    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.drop(self.embed(x))

        # Forward propagate RNN
        out, h = self.lstm(x, h)
        out = self.drop(out)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        #out = self.linear(out.view(out.size(0) * out.size(1), out.size(2)))
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time step
        out = self.linear(out)

        return out, h


class model5(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(model5, self).__init__()
        # Parameters
        self.hidden_size = 250
        self.num_layers = 1
        self.type = 'GRU'
        self.name = '{} - {} hidden cells, {} layers'.format(self.type, str(self.hidden_size), str(self.num_layers))

        # Word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Structure
        self.gru = nn.GRU(embed_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.init_weights()

        weights = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
        self.embd = nn.Embedding.from_pretrained(weights)
        input = torch.LongTensor([1])
        self.embd(input)


    def init_weights(self):
        #self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate RNN
        out, h = self.gru(x)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))

        # Decode hidden states of all time step
        out = self.linear(out)

        return out, h
