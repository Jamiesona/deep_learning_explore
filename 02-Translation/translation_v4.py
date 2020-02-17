import re
from collections import Counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 模型配置
config = {
    "input_size": 100,
    "hidden_size": 100,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


class MyEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        """
        构造方法.
        :param input_size: 词向量大小
        :param hidden_size: 隐藏状态大小
        :param vocab_size: 词表大小
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.rnn = nn.GRU(input_size, hidden_size)  # 层数, 是否双向之后考虑

    def forward(self, x, h0=None):
        x = self.embedding(x)
        outs, hn = self.rnn(x, h0)
        # outs记录了encoder每个时间布的输出, hn记录最后一个时间步的隐藏状态
        return outs, hn


class MyDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.rnn = nn.GRU(input_size, hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0):
        x = self.embedding(x)
        outs, hn = self.rnn(x, h0)
        outs = self.dense(outs)
        return outs, hn


class MyEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, seq0, seq1):
        _, hn = self.enc(seq0)
        outs, _ = self.dec(seq1, hn)
        return outs


class Vocab:
    def __init__(self, lines):
        seq = []
        for line in lines:
            seq.extend(line)
        counter = Counter(seq)
        self.idx2word = ['<bos>', '<eos>', '<unk>', '<pad>'] + [item[0]
                                                                for item in counter.most_common()]
        self.bos, self.eos, self.unk, self.pad = self.idx2word[0:4]
        self.word2idx = {}
        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, index):
        if not isinstance(index, (list, tuple)):
            return [self.word2idx.get(index, self.word2idx['<unk>'])]
        else:
            res = []
            for idx in index:
                res.append(self.word2idx.get(idx, self.word2idx['<unk>']))
            return res


def get_list_from_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.read()

    lines = lines.split('\n')
    src_list, tgt_list = [], []
    for line in lines:
        if line == '':
            break
        pair = line.split('\t')
        pair[0] = re.sub('[^a-z]+', ' ', pair[0].lower())
        src_list.append(pair[0].strip().split(' '))
        tgt_list.append(list(pair[1]))

    return src_list, tgt_list


def build_array(lines: list, vocab: Vocab, is_source=True, device=config["device"]):
    rl = []
    for line in lines:
        res = vocab[line]
        if not is_source:
            res = vocab[vocab.bos] + res + vocab[vocab.eos]
        rl.append(torch.tensor(res, dtype=torch.long).unsqueeze(1).to(device))

    return rl


src_list, tgt_list = get_list_from_file("eng2chi.txt")
vocab1 = Vocab(src_list)
vocab2 = Vocab(tgt_list)
x, y = build_array(src_list, vocab1), build_array(tgt_list, vocab2, is_source=False)

enc = MyEncoder(config["input_size"], config["hidden_size"], len(vocab1))
dec = MyDecoder(config["input_size"], config["hidden_size"], len(vocab2))
enc_dec = MyEncoderDecoder(enc, dec).to(config["device"])

loss_fn = nn.CrossEntropyLoss()
optim_fn = optim.Adam(enc_dec.parameters(), lr=0.05)


def train_model(model, loss_fn, optim_fn, x, y, epochs=10):
    losses = torch.zeros(epochs)
    for epoch in range(epochs):
        for idx in range(len(x)):
            xx, yy = x[idx], y[idx]
            optim_fn.zero_grad()
            y_hat = model(xx, yy[:-1])
            y_hat = y_hat.transpose(1, 2)
            loss = loss_fn(y_hat, yy[1:])
            loss.backward()
            optim_fn.step()
            losses[epoch] += loss.detach().item()

    return losses / len(x)


def predict(model: MyEncoderDecoder, x, max_len: int):
    model.train(False)
    x = torch.tensor(vocab1[x], device=config["device"]).unsqueeze(1)
    _, hn = model.enc(x)
    dec_x = torch.tensor(vocab2[vocab2.bos], dtype=torch.long, device=config["device"]).unsqueeze(1)
    res = []
    for idx in range(max_len):
        outs, hn = model.dec(dec_x, hn)
        res.append(vocab2.idx2word[torch.argmax(outs, 2).item()])
        dec_x = outs.argmax(2)
        if res[-1] == vocab2.eos:
            break
    return "".join(res)


losses = train_model(enc_dec, loss_fn, optim_fn, x, y, 20)
plt.plot(losses, 'r-o')
plt.show()

print(src_list[0])
print(predict(enc_dec, src_list[0], 10))
