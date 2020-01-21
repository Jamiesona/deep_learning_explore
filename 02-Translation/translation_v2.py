import torch
import torch.nn as nn
import re
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


class Lang:
    """表示一种语言."""

    def __init__(self):
        self.word2idx = {SOS_TOKEN: 0, EOS_TOKEN: 1}
        self.idx2word = {0: SOS_TOKEN, 1: EOS_TOKEN}
        self.count = 2

    def add_sentence(self, sentence):
        raise NotImplementedError("未实现的方法.")

    def to_tensor(self, sentence):
        raise NotImplementedError("未实现的方法.")


class English(Lang):
    def __init__(self):
        super(English, self).__init__()

    def add_sentence(self, sentence):
        words = re.split(r"\s", sentence)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.count
                self.idx2word[self.count] = word
                self.count += 1

    def to_tensor(self, sentence):
        words = re.split(r"\s", sentence)
        lst = [self.word2idx[SOS_TOKEN]] + [self.word2idx[word]
                                            for word in words] + [self.word2idx[EOS_TOKEN]]
        return torch.tensor(lst)


class Chinese(Lang):
    def __init__(self):
        super(Chinese, self).__init__()

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.word2idx[word] = self.count
                self.idx2word[self.count] = word
                self.count += 1

    def to_tensor(self, sentence):
        lst = [self.word2idx[SOS_TOKEN]] + [self.word2idx[word]
                                            for word in sentence] + [self.word2idx[EOS_TOKEN]]
        return torch.tensor(lst, dtype=torch.long)


MAX_SEQ_LEN = 128


class Translator(nn.Module):
    """使用transformer实现翻译器."""

    def __init__(self, d_model, input_lang, output_lang):
        super(Translator, self).__init__()
        self.d_model = d_model
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pe = Translator.positional_encoding(
            d_model, MAX_SEQ_LEN).clone().detach().requires_grad_(False)
        # 输入和输出对应的word embedding
        self.input_embedding = nn.Embedding(input_lang.count, d_model)
        self.output_embedding = nn.Embedding(output_lang.count, d_model)
        # transformer模型
        self.tf = nn.Transformer(
            d_model, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=1024)
        # 输出的分类器
        self.linear = nn.Linear(d_model, self.output_lang.count)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # 损失函数和优化器
        self.loss_fn = nn.NLLLoss()
        self.optim_fn = optim.SGD(self.parameters(), lr=0.02)

    @staticmethod
    def positional_encoding(d_model, max_len=128):
        pe = torch.zeros(max_len, d_model)
        pos = np.arange(0, max_len).reshape(max_len, 1)
        pe[:, 0::2] = torch.sin(torch.tensor(
            pos/(10000.0 ** (np.arange(0, d_model, 2)/d_model))))
        pe[:, 1::2] = torch.cos(torch.tensor(
            pos/(10000.0 ** (np.arange(0, d_model, 2)/d_model))))
        return pe

    def format_sentence(self, sentence, is_input=True):
        """
        把句子格式化成需要word embedding + positional encoding的形式.
        """
        lang = self.input_lang if is_input else self.output_lang
        embedding = self.input_embedding if is_input else self.output_embedding

        x = lang.to_tensor(sentence)
        em = embedding(x)
        pe = self.pe[x, :]
        return (em + pe).unsqueeze(1)

    def train_with_one_setence_pair(self, src_sentence, tgt_sentence):
        self.optim_fn.zero_grad()
        tgt = self.output_lang.to_tensor(tgt_sentence)
        T = tgt.size(0)-1
        mask = np.triu(np.ones((T, T)), k=0)
        mask[mask == 1] = -1e9
        tgt_mask = torch.from_numpy(mask)
        src_sentence = self.format_sentence(src_sentence)
        tgt_sentence = self.format_sentence(tgt_sentence, False)
        output = self.tf(src_sentence, tgt_sentence[0:-1], tgt_mask=tgt_mask)
        output = self.linear(output.view(-1, self.d_model))
        output = self.logsoftmax(output)
        loss_val = self.loss_fn(output, tgt[1:])
        loss_val.backward()
        self.optim_fn.step()
        return loss_val.detach().numpy()

    def forward(self, input_sentence):
        self.train(False)
        # 处理输入句子为word embedding + positional encoding的形式
        src_sentence = self.format_sentence(input_sentence)
        tgt = [self.output_lang.word2idx[SOS_TOKEN]]
        res = ""
        for _ in range(MAX_SEQ_LEN):
            em = self.output_embedding(torch.tensor(tgt))
            pe = self.pe[torch.tensor(tgt)]
            tgt_sentence = (em + pe).unsqueeze(1)
            output = self.tf(src_sentence, tgt_sentence)
            output = self.linear(output.view(-1, self.d_model))
            output = self.logsoftmax(output)
            output = torch.argmax(output, dim=1)
            output = output[-1]
            res += self.output_lang.idx2word[int(output.numpy())]
            if output == self.output_lang.word2idx[EOS_TOKEN]:
                break
            else:
                tgt.append(output)
        return res


english = English()
chinese = Chinese()

english.add_sentence("hello world")
chinese.add_sentence("你好世界")

ts = Translator(512, english, chinese)
lvs = []
for i in range(100):
    lv = ts.train_with_one_setence_pair("hello world", "你好世界")
    lvs.append(lv)

plt.plot(lvs)
print(ts("hello world"))
plt.show()
