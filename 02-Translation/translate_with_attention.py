import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 句子开始标志
SOS_TOKEN = "<SOS>"
# 句子结束标志
EOS_TOKEN = "<EOS>"
# 未知单词标志
UNKNOW_TOKEN = "<UNKNOW>"
# 句子最大长度
MAX_SENTENCE_LEN = 128


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
        return out[:, 0, :], hidden

    def init_hidden(self):
        """初始化隐含状态."""
        return torch.zeros(1, 1, self.hidden_size, dtype=torch.float32)


class DecoderRNN(nn.Module):
    """解码器.
    使用Attention机制, 加权求和Encoder的历史输出.
    """

    def __init__(self, out_size, hidden_size, max_sentence_length=MAX_SENTENCE_LEN):
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
        self.log_softmax = nn.LogSoftmax(dim=0)

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
        y = self.log_softmax(y)
        return y.unsqueeze(0), h


class Language:
    """代表一种语言.
    """

    def __init__(self):
        self.word2idx = {"<SOS>": 0, "<EOS>": 1, "<UNKNOW>": 2}
        self.idx2word = {0: "<SOS>", 1: "<EOS>", 2: "<UNKNOW>"}
        # 最后一个单词的位置
        self.count = 3

    def add_sentence(self, sentence):
        """
        将一个句子包含的单词添加到语言之中, 不处理重复的单词.
        :param sentence: 某种语言的句子
        :return:
        """
        raise NotImplementedError

    def add_sentences(self, sentences):
        """向语言中添加一批句子."""
        for sentence in sentences:
            self.add_sentence(sentence)

    def sentence2tensor(self, sentence, max_length=MAX_SENTENCE_LEN):
        """
        将句子转换成Tensor对象.
        :param sentence: 句子
        :param max_length: 句子最大长度
        :return: 转换结果
        """
        raise NotImplementedError


class English(Language):
    """英语."""

    def __init__(self):
        super(English, self).__init__()
        self.word2idx[" "] = self.count
        self.idx2word[self.count] = " "
        self.count += 1

    def add_sentence(self, sentence):
        # 划分出每个单词
        words = sentence.split()
        for word in words:
            # 添加没有出现过的单词到语言中
            if word not in self.word2idx:
                self.word2idx[word] = self.count
                self.idx2word[self.count] = word
                self.count += 1

    def sentence2tensor(self, sentence, max_length=MAX_SENTENCE_LEN):
        res = torch.zeros(1, 1, max_length, dtype=torch.long)
        words = sentence.split()
        # 加上句子开始标志
        res[0, 0, 0] = self.word2idx[SOS_TOKEN]
        for i, word in enumerate(words):
            if word in self.word2idx:
                res[0, 0, i + 1] = self.word2idx[word]
            else:
                res[0, 0, i + 1] = self.word2idx[UNKNOW_TOKEN]
        # 加上句子结束标志
        res[0, 0, i + 2] = self.word2idx[EOS_TOKEN]
        return res


class Chinese(Language):
    """中文."""

    def __init__(self):
        super(Chinese, self).__init__()

    def add_sentence(self, sentence):
        # 划分出每个单词
        for word in sentence:
            # 添加没有出现过的单词到语言中
            if word not in self.word2idx:
                self.word2idx[word] = self.count
                self.idx2word[self.count] = word
                self.count += 1

    def sentence2tensor(self, sentence, max_length=MAX_SENTENCE_LEN):
        res = torch.zeros(1, 1, max_length, dtype=torch.long)
        # 加上句子开始标志
        res[0, 0, 0] = self.word2idx[SOS_TOKEN]
        for i, word in enumerate(sentence):
            if word in self.word2idx:
                res[0, 0, i + 1] = self.word2idx[word]
            else:
                res[0, 0, i + 1] = self.word2idx[UNKNOW_TOKEN]
        # 加上句子结束标志
        res[0, 0, i + 2] = self.word2idx[EOS_TOKEN]
        return res


class Translator:
    """使用Encoder和Decoder组成翻译器."""

    def __init__(self, from_lang, to_lang, *file_names):
        """
        构造翻译器.
        :param from_lang: 源语言
        :param to_lang: 目标语言
        """
        self._from_lang = from_lang
        self._to_lang = to_lang
        # 读取数据文件, 初始化源语言和目标语言
        self._prepare_data(*file_names)
        self._encoder = EncoderRNN(self._from_lang.count, 100)
        self._decoder = DecoderRNN(self._to_lang.count, 150)
        # 创建编码器和解码器
        self._encoder = EncoderRNN(self._from_lang.count, 100)
        self._decoder = DecoderRNN(self._to_lang.count, 100)
        # 优化器
        self._encoder_optimizer = optim.SGD(self._encoder.parameters(), lr=0.01, momentum=0.9)
        self._decoder_optimizer = optim.SGD(self._decoder.parameters(), lr=0.01, momentum=0.9)
        # 目标函数
        self._loss = nn.NLLLoss()
        # 是否已经训练完成
        self.is_trained = False

    def _prepare_data(self, *file_names):
        """
        从文件初始化源语言和目标语言
        :param file_names: 两个源语言和目标语言的文件
        :return: None
        """
        with open(file_names[0], "r") as f:
            lines = f.readlines()
            for line in lines:
                self._from_lang.add_sentence(line)
        with open(file_names[1], "r") as f:
            lines = f.readlines()
            for line in lines:
                self._to_lang.add_sentence(line)

    def _train(self, input_tensor, target_tensor):
        """
        输入一个句子对, 进行训练.
        :param input_tensor: 源语言句子
        :param target_tensor: 目标语言句子
        :return: 损失函数值
        """
        # 1.梯度清零
        self._encoder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()

        # 2.encoder对输入句子进行编码
        encoder_hidden = self._encoder.init_hidden()
        encoder_outputs = []
        for word in input_tensor[0, 0]:
            o, encoder_hidden = self._encoder(word, encoder_hidden)
            encoder_outputs.append(o)

        # 3.使用解码器处理编码器的处理结果
        encoder_outputs = torch.cat(encoder_outputs)
        use_teaching = np.random.rand() >= 0.5
        decoder_input = torch.tensor(self._to_lang.word2idx[SOS_TOKEN])
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        for target_word in target_tensor[0, 0]:
            if use_teaching:
                decoder_output, decoder_hidden = self._decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = target_word
                decoder_outputs.append(decoder_output)
            else:
                decoder_output, decoder_hidden = self._decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output.topk(1)[1][0]
                decoder_outputs.append(decoder_output)
        # 4.计算损失函数
        decoder_outputs = torch.cat(decoder_outputs)
        loss = self._loss(decoder_outputs, target_tensor[0, 0])
        loss.backward()
        # 5.更新参数
        self._encoder_optimizer.step()
        self._decoder_optimizer.step()
        return loss.detach().numpy()

    def fit(self, epochs=10):
        """
        模型训练.
        :param epochs: 训练回合数.
        :return:
        """
        self.is_trained = True

    def translate(self, sentence):
        """
        训练好的模型进行翻译.
        :param sentence: 待翻译句子.
        :return: 翻译结果
        """
        if self.is_trained:
            pass
        else:
            raise Exception("The translator has not been trained yet, please call the fit method first.")


translater = Translator(English(), Chinese(), "english.txt", "chinese.txt")
translater.fit()
