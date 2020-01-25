import torch
import torch.nn as nn


class MyRNNCell(nn.Module):
    """
    Simple RNN implement.
    """

    def __init__(self, input_size, hidden_size):
        """
        Constructor function.
        :param input_size: input feature size
        :param hidden_size: hidden state size
        """
        super(MyRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size, True)
        self.activation = nn.Tanh()

    def forward(self, x, hidden_pre):
        x = torch.cat((x, hidden_pre), 0)
        x = self.linear(x)
        return self.activation(x)

