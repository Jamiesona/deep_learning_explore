import torch
import torch.nn as nn

# 常量定义
HIDDEN_SIZE = 100


class Encoder(nn.Module):
    """
    编码器.
    """

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):
        """
        构造函数.
        :param input_size: 输入包含的单词数量.
        :param hidden_size: 隐含状态大小.
        """
        super(Encoder, self).__init__()
        # 运行使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, x, hidden):
        pass

    def init_hidden(self):
        pass
