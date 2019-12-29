import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """编码器.
    每次输入一个字符, 得到相应的输出和隐含状态.
    """

    def __init__(self, out_size, hidden_size):
        """构造函数.
        :param out_size 词汇表大小
        :param hidden_size 隐含状态的大小
        """
        # 首先调用父类构造函数
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(out_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        """前向传播.
        这里每次只能输入一个字符.
        """
        x = self.embedding(x).view(1, 1, -1)  # GRU的输入要求(seq_len, batch_size, feature_size)
        out, hidden = self.gru(x, hidden)
        out = self.relu(out)
        return out[0, 0], hidden

    def init_hidden(self):
        """初始化隐含状态."""
        return torch.zeros(1, 1, self.hidden_size, dtype=torch.float32)


class DecoderRNN(nn.Module):
    """解码器.
    使用Attention机制, 加权求和Encoder的历史输出.
    """

    def __init__(self, out_size, hidden_size, max_sentence_length):
        """构造函数.
        :param out_size 输出语言的词汇表大小
        :param hidden_size 隐含状态大小
        :param max_sentence_length 输入句子的最大长度
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(out_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size * 2, max_sentence_length)  #
        self.softmax_1 = nn.Softmax(dim=0)
        self.linear_2 = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_size)
        self.softmax_2 = nn.Softmax(dim=0)

    def forward(self, x, hidden, encoder_outputs):
        """前向传播, 不使用Attention时, Encoder的处理结果都有最后的hidden表达, 会有信息的丢失.
        :param x 整数, 表示上一个时间步Decoder的输出, 数值上等于输出单词在输出语言表词汇表上的索引
        :param hidden 上一个时间步的隐藏状态, 初始时为Encoder最后一个时间步的hidden状态
        :param encoder_outputs 输入序列每个时间步经过Encoder处理之后的输出
        """
        embeded = self.embedding(x).view(1, -1)  # size = (1, hidden_size)
        x = torch.cat((embeded, hidden[0]), dim=1)
        x = self.linear_1(x.squeeze())
        weights = self.softmax_1(x).unsqueeze(0)  # shape = (1, max_sequrnce_length)
        x = torch.matmul(weights, encoder_outputs)
        input = torch.cat((embeded, x), dim=1).squeeze()
        input = self.linear_2(input).unsqueeze(dim=0).unsqueeze(dim=0)
        y, h = self.gru(input, hidden)
        y = self.linear_3(y[0, 0])
        y = self.softmax_2(y)
        return y.topk(1)[1][0], h
