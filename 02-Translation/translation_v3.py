import torch.nn as nn
import torch.cuda as cuda
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    """
    翻译系统的编码器.
    """

    def __init__(self, input_size, vocab_size):
        """
        构造函数.
        :param input_size: 词向量大小
        :param vocab_size: 词汇表大小
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm_cell = nn.LSTMCell(input_size, input_size)
        self.input_size = input_size
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)

    def init_hidden_and_cell_state(self):
        """
        初始化第一个hidden state和cell state.
        :return: hidden state和cell state
        """
        c0 = torch.zeros(1, self.input_size, dtype=torch.float32).to(self.device)
        h0 = torch.zeros(1, self.input_size, dtype=torch.float32).to(self.device)
        return h0, c0

    def forward(self, x):
        x = x.to(self.device)
        seq_len = len(x)
        encoder_outputs = torch.empty(seq_len, self.input_size, dtype=torch.float32, device=self.device)
        h0, c0 = self.init_hidden_and_cell_state()

        for idx, item in enumerate(x):
            h0, c0 = self.forward0(item, h0, c0)
            encoder_outputs[idx] = h0[0]

        return h0, encoder_outputs

    def forward0(self, x1, h0, c0):
        """
        处理一个时间步.
        :param x1:
        :param h0:
        :param c0:
        :return:
        """
        embed = self.embedding(x1)
        embed = embed.view(1, -1)
        h1, c1 = self.lstm_cell(embed, (h0, c0))
        return h1, c1


class SimpleDecoder(nn.Module):
    """
    翻译系统的解码器.不使用注意力机制.
    """

    def __init__(self, input_size, vocab_size):
        """
        构造函数.
        :param input_size:
        :param vocab_size:
        """
        super().__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru_cell = nn.GRUCell(input_size, input_size)
        self.linear = nn.Linear(input_size, vocab_size)
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x1, h0):
        """
        前向传播.
        :param x1:
        :param h0:
        :return: 预测出的下一个字符, hidden state
        """
        x1 = x1.to(self.device)
        embed = self.embedding(x1)
        embed = embed.view(1, -1)
        h1 = self.gru_cell(embed, h0)
        y = self.linear(h1)
        return self.logSoftMax(y), h1


class Translator(nn.Module):
    """
    翻译系统.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = nn.NLLLoss()
        self.optim_fn = optim.SGD(self.parameters(), lr=0.05, momentum=0.5)

    def forward(self, x, y=None):
        """
        模型训练或者预测, y是None是为预测, 否则为训练.
        :param x: 源语言句子
        :param y: 目标语言句子
        :return: 预测结果, 损失函数值
        """
        if y is None:
            self.train(False)
        else:
            self.train(True)
            self.optim_fn.zero_grad()

        hidden, encoder_outs = self.encoder(x)
        decoder_outs, decoder_outputs_len = self.decoder_forward(hidden, y)
        loss = .0
        if y is not None:
            loss = self.loss_fn(decoder_outs[0:len(y)], y)
            loss.backward()
            self.optim_fn.step()
            loss = loss.cpu().detach().numpy()
        y_predict = decoder_outs[:decoder_outputs_len].argmax(1)
        return y_predict, loss

    def decoder_forward(self, h0, y=None):
        """
        计算decoder的输出
        :param h0: encoder的输出结果
        :param y: not None, 预测; else 训练
        :return: decoder输出预测结果, 预测的序列长度
        """
        max_len = 128
        steps = max_len if y is None else len(y)
        # decoder的第一个输入, 即句子的开始
        x1 = torch.tensor(0)
        # 记录全部输出
        outs = torch.empty(steps, self.decoder.vocab_size, dtype=torch.float32)
        for idx in range(steps):
            x1, h0 = self.decoder(x1, h0)
            outs[idx] = x1
            if y is None:
                x1 = torch.argmax(x1, 1)
            else:
                x1 = y[idx]
            # 输出了句子结束符号, 跳出循环
            if x1 == 1:
                break
        return outs, idx + 1


x = [torch.tensor([2, 3, 4, 5]), torch.tensor([3, 3, 2, 7, 5])]
y = [torch.tensor([5, 3, 8, 2, 1]), torch.tensor([9, 7, 8, 2, 1])]

encoder = Encoder(100, 10)
decoder = SimpleDecoder(100, 10)
translator = Translator(encoder, decoder)

lvs = np.empty(120, dtype=np.float)
for i in range(120):
    lv = 0.0
    for j in range(len(x)):
        y_pre, lv_tmp = translator(x[j], y[j])
        lv += lv_tmp
    lvs[i] = lv / len(x)

plt.plot(range(0, 120, 5), lvs[0::5], "k-->")
plt.show()

print(translator(x[0]))
