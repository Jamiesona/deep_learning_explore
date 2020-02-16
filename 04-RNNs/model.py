import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.rnn = nn.GRU(input_size, hidden_size)

    def forward(self, x, h0=None):
        x = self.embedding(x)
        _, hn = self.rnn(x, h0)
        return hn

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.rnn = nn.GRU(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0):
        x = self.embedding(x)
        outs, hn = self.rnn(x, h0)
        outs = self.dense(outs)
        return outs, hn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        encoder_hidden = self.encoder(x)
        outs, _ = self.decoder(y, encoder_hidden)
        return outs


