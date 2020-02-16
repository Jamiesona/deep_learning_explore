import re
from collections import Counter
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


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


def build_array(vocab, lines, max_len, is_target=False):
    res = []
    for line in lines:
        if is_target:
            line = ['<bos>'] + line + ['<eos>']
        if len(line) < max_len:
            line += "<pad>" * (max_len - len(line))
        res.append(vocab[line[0: max_len]])
    return torch.tensor(res, dtype=torch.long)


src_list, tgt_list = get_list_from_file("04-RNNs/eng2chi.txt")
vocab_eng = Vocab(src_list)
vocab_chi = Vocab(tgt_list)
max_len = 10
eng, chi = build_array(vocab_eng, src_list, max_len), build_array(vocab_chi, tgt_list, max_len, True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eng, chi = eng.to(DEVICE), chi.to(DEVICE)

train_data = TensorDataset(eng, chi)
train_dl = DataLoader(train_data, batch_size=1, shuffle=True)

