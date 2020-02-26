import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_data(root='./', batch_size=20, percent=0.2):
    ''''加载数据.'''
    data = MNIST(root=root, train=True, transform=ToTensor, download=True)
    data.data = data.data / 255.0
    n = int(len(data.train_data) * percent)
    # 不使用全部数据参加训练
    x = data.data[:n].reshape(-1, 1, 28, 28)
    data_iter = DataLoader(x, batch_size=batch_size, shuffle=True)
    return data_iter

class Net_D(nn.Module):
    '''判别器.'''
    def __init__(self):
        super(Net_D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1), # (N, 6, 28, 28)
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #(N, 6, 14, 14)
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1), #(N, 12, 14, 14)
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #(N, 12, 7, 7)
            nn.Flatten(), #(N, 12*7*7)
            nn.Linear(12*7*7, 12),
            nn.Dropout(0.2),
            nn.Linear(12, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

class Net_G(nn.Module):
    '''生成器.'''
    def __init__(self):
        super(Net_G, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 14*14),
            Reshape(),
            nn.Upsample(size=(20, 20), mode='bilinear', align_corners=True),
            nn.BatchNorm2d(num_features=1),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1), # (N, 6, 20, 20)
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.Upsample(size=(28, 28), mode='bilinear', align_corners=True),
            nn.BatchNorm2d(num_features=6),
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 14, 14)

def update_D(net_d, net_g, x, noise, trainer_d, loss, device):
    batch_size = x.shape[0]
    ones = torch.ones(batch_size, device=device)
    zeros = torch.zeros(noise.shape[0], device=device)
    fake_x = net_g(noise)
    ones_hat = net_d(x).squeeze()
    zeros_hat = net_d(fake_x).squeeze()
    d_loss = 0.5 * (loss(ones_hat, ones) + loss(zeros_hat, zeros))
    d_loss.backward()
    trainer_d.step()
    trainer_d.zero_grad()
    return d_loss.cpu().detach().item()

def update_G(net_g, net_d, noise, trainer_g, loss, device):
    batch_size = noise.shape[0]
    ones = torch.zeros(batch_size, device=device)
    fake_y = net_d(net_g(noise)).squeeze()
    g_loss = loss(fake_y, ones)
    trainer_g.step()
    trainer_g.zero_grad()
    return g_loss.cpu().detach().item()


def train_model(net_d, net_g, data_iter, device, epochs=10):
    net_d.to(device)
    net_g.to(device)

    trainer_d = torch.optim.Adam(net_d.parameters(), lr=0.1)
    trainer_g = torch.optim.Adam(net_g.parameters(), lr=0.001)
    loss = nn.BCELoss()

    d_losses = torch.zeros(epochs)
    g_losses = torch.zeros(epochs)

    n = len(data_iter)
    for epoch in range(epochs):
        for x in data_iter:
            x = x.to(device)
            noise = torch.randn(x.shape[0]//2, 64, device=device)
            d_loss = update_D(net_d, net_g, x, noise, trainer_d, loss, device)
            g_loss = update_G(net_g, net_d, noise, trainer_g, loss, device)
            d_losses[epoch] += d_loss
            g_losses[epoch] += g_loss

    return d_losses/n, g_losses/n

def evaluate_g(net_d, net_g, noise):
    net_d.eval()
    net_g.eval()
    fake_x = net_g(noise)
    y_hat = net_d(fake_x)
    imgs = fake_x.cpu().detach().squeeze()

    batch_size = imgs.shape[0]
    for idx in range(batch_size):
        plt.subplot(1, batch_size, idx+1)
        plt.imshow(imgs[idx])
        plt.axis('off')
    print(y_hat)
    plt.show()

if __name__=='__main__':
    net_d, net_g = Net_D(), Net_G()
    data_iter = load_data()
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_losses, g_losses = train_model(net_d, net_g, data_iter, device, epochs)

    plt.plot(range(epochs), d_losses, 'r--', label='d_losses')
    plt.plot(range(epochs), g_losses, 'g--', label='g_losses')
    plt.legend()
    plt.show()

    noise = torch.randn(5, 64, device=device)
    evaluate_g(net_d, net_g, noise)

    for x in data_iter:
        y = net_d(x.to(device)).cpu().detach()
        print(y)
        break