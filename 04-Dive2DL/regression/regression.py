import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.Sigmoid(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        y = self.model(x)
        return y

def train_net(model, data_iter, trainner, loss, epochs):
    model.train()
    losses = torch.zeros(epochs)
    n = len(data_iter)

    for epoch in range(epochs):
        for x, y in data_iter:
            y_hat = model(x)
            loss_val = loss(y_hat, y)
            loss_val.backward()
            trainner.step()
            trainner.zero_grad()
            losses[epoch] += loss_val.detach().item()
    plt.plot(range(epochs), losses, 'k--o')
    plt.xlabel('epochs')
    plt.ylabel('Losses')
    return losses

if __name__ == '__main__':

    # 原始数据
    x = torch.linspace(0, 2*np.pi, 200).unsqueeze(1)
    y = 0.2 * x**2 + 0.1
    data_iter = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=20, shuffle=True)

    # 构造模型
    net = Net()
    loss = nn.MSELoss()
    trainer = torch.optim.Adam(net.parameters(), lr=0.05)
    
    # 训练模型
    losses = train_net(net, data_iter, trainer, loss, 30)
    
    # 预测数据
    net.eval()
    y_hat = net(x).detach()
    plt.figure()
    plt.plot(x, y, 'k--', x, y_hat, 'r--')
    plt.show()
    print(losses[-1])
