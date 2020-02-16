import re
from collections import Counter
import torch


class Vocab:
    def __init__(self, lines):
        seq = []
        for line in lines:
            seq.extend(line)
        counter = Counter(seq)
        self.idx2word = ['<bos>', '<eos>', '<unk>', '<pad>'] + [item[0]
                                                                for item in counter.most_common()]
        self.word2idx = {}
        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx

    def __getitem__(self, index):
        if not isinstance(index, (list, tuple)):
            return self.word2idx[index]
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


def build_array(vocab, lines, max_len):
    res = []
    for line in lines:
        if len(line) < max_len:
            line += "<pad>" * (max_len - len(line))
        res.append(vocab[line[0: max_len]])
    return torch.tensor(res, dtype=torch.long)


s, t = get_list_from_file("04-RNNs/eng2chi.txt")
vocab = Vocab(s)

print(build_array(vocab, s, 10))
