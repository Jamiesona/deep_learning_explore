import torch
import torch.nn as nn
import re
import torch.optim as optim

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
        return torch.tensor(lst)


class Translator(nn.Module):
    """使用transformer实现翻译器."""

    def __init__(self, input_lang, target_lang, hidden_size):
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.input_embedding = nn.Embedding(input_lang.count, hidden_size)
        self.target_embedding = nn.Embedding(target_lang.count, hidden_size)
        self.trsfm = nn.Transformer(hidden_size, 3, 3)
        self.loss = nn.MSELoss()
