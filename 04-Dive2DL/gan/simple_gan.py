import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class Net_D(nn.Module):
    def __init__(self):
        super(Net_D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 5),
            nn.Tanh(),
            nn.Linear(5, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



class Net_G(nn.Module):
    def __init__(self):
        super(Net_G, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2)
        )

    def forward(self, x):
        return self.model(x)

def train_d(net_d, net_g, x, y, z, trainer, loss):
    net_d.train()
    net_g.eval()
    trainer.zero_grad()
    y_hat = net_d(x).squeeze()
    y_hat2 = net_d(net_g(z)).squeeze()
    zeros = torch.zeros(x.shape[0]).to(device)
    ls = (loss(y_hat, y) + loss(y_hat2, zeros))/2
    ls.backward()
    trainer.step()
    return ls.cpu().detach().item()

def train_g(net_g, net_d, z, trainer, loss):
    net_d.eval()
    net_g.train()
    trainer.zero_grad()
    y_hat = net_d(net_g(z)).squeeze()
    ones = torch.ones(z.shape[0]).to(device)
    ls = loss(y_hat, ones)
    ls.backward()
    trainer.step()
    return ls.cpu().detach().item()

def train_gan(net_d, net_g, data_iter, trainer_d, trainer_g, epochs, loss):
    
    d_losses = torch.zeros(epochs)
    g_losses = torch.zeros(epochs)
    n = len(data_iter)

    # 单独训练判别器
    for epoch in range(epochs):
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            z = torch.normal(0, 1, (x.shape[0], 2)).to(device)
            d_loss = train_d(net_d, net_g, x, y, z, trainer_d, loss)
            g_loss = train_g(net_g, net_d, z, trainer_g, loss)
            d_losses[epoch] += d_loss
            g_losses[epoch] += g_loss
    return d_losses/n, g_losses/n



if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练数据
    x = torch.normal(0, 1, (1000, 2))
    data = torch.matmul(x, torch.tensor([[3, 2], [-0.1, 0.5]])) + torch.tensor([-1, 2])
    y = torch.ones(1000)
    data_iter = DataLoader(TensorDataset(data, y), batch_size=10, shuffle=True)
    # 数据线性变换, 并展示
    plt.scatter(x[::10, 0], x[::10, 1], color='blue')
    plt.scatter(data[::10, 0], data[::10, 1], color='red')
    plt.show()
    # 模型创建, 损失函数和优化器定义
    net_d, net_g = Net_D().to(device), Net_G().to(device)
    loss = nn.BCELoss()
    trainer_d = torch.optim.Adam(net_d.parameters(), 0.05)
    trainer_g = torch.optim.Adam(net_g.parameters(), 0.005)
    # 模型训练, 判别器和生成器的损失函数变化
    d_losses, g_losses = train_gan(net_d, net_g, data_iter, trainer_d, trainer_g, 30, loss)
    plt.figure()
    plt.plot(d_losses, 'r--', label='loss of D')
    plt.plot(g_losses, 'g--', label='loss of G')
    plt.legend()
    plt.show()
    # 模型评估
    net_d.eval()
    net_g.eval()
    # 噪声, 根据噪声生成数据
    z = torch.normal(0, 1, (100, 2)).to(device)
    x_fake = net_g(z).cpu().detach()
    plt.figure()
    # 绘制结果
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Real data')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='red', label='Fake data')
    plt.legend()
    plt.show()

    a = net_g.model[0].weight.cpu().detach()
    b = net_g.model[0].bias.cpu().detach()

    print(a)
    print(b)

    plt.figure()
    x = torch.matmul(z.cpu(), a.transpose(0, 1)) + b
    plt.scatter(data[:, 0], data[:, 1], color='blue', label='Real data')
    plt.scatter(x[:, 0], x[:, 1], color='red', label='Fake data')
    plt.show()