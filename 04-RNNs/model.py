import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re
from util import train_dl, vocab_eng, vocab_chi, DEVICE


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
input_size, hidden_size = 100, 100
encoder = Encoder(input_size, hidden_size, len(vocab_eng))
decoder = Decoder(input_size, hidden_size, len(vocab_chi))
enc_dec = EncoderDecoder(encoder, decoder)
enc_dec.to(DEVICE)

epochs = 20
loss_fn = nn.CrossEntropyLoss()
optim_fn = optim.Adam(enc_dec.parameters(), lr=0.1)

losses = torch.zeros(epochs)
for epoch in range(epochs):
    lv = 0
    idx = 0
    for x, y in train_dl:
        x, y = x.transpose(0, 1), y.transpose(0, 1)
        y_hat = enc_dec(x, y[:-1])
        loss = loss_fn(y_hat.transpose(1, 2), y[1:])
        loss.backward()
        optim_fn.step()
        optim_fn.zero_grad()
        lv += loss.detach().item()
        idx += 1
    losses[epoch] = lv / idx

plt.plot(losses, 'r-h')
plt.show()

def translate_with_encoder_decoder(line, enc_dec, vocab_eng, vocab_chi):
    enc_dec.train(False)
    line = re.sub("[^a-z]+", " ", line.lower())
    line = line.split(" ")
    x = torch.tensor(vocab_eng[line], dtype=torch.long).unsqueeze(1).to(DEVICE)
    hidden = enc_dec.encoder(x)
    y = torch.tensor(vocab_chi['<bos>'], dtype=torch.long).unsqueeze(1).to(DEVICE)
    outs = []
    for _ in range(10):
        out, hidden = enc_dec.decoder(y, hidden)
        out_idx = torch.argmax(out, 2)
        outs.append(vocab_chi.idx2word[out_idx])
        if outs[-1] == '<eos>':
            break
    return ' '.join(outs)

print(translate_with_encoder_decoder("where are you from", enc_dec, vocab_eng, vocab_chi))
