import torch.nn as nn
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(623)


class AutoEncoder(nn.Module):
    """AutoEncoder Class."""

    def __init__(self, width, height, in_channels=1):
        super(AutoEncoder, self).__init__()
        self.width = width
        self.height = height
        self.in_channels = in_channels

        # Encoder Part
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16 * in_channels, kernel_size=3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16 * in_channels, out_channels=8 * in_channels, kernel_size=3),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8 * in_channels, out_channels=in_channels, kernel_size=3),
            nn.ReLU()
        )

        # Decoder Part
        self.ups1 = nn.Upsample(size=(width, height))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        encoder_output = self.conv3(x)

        y = self.ups1(encoder_output)
        decoder_output = self.conv4(y)

        return decoder_output, encoder_output


def train(input_tensor, target_tensor, encoder, loss, optimizer):
    optimizer.zero_grad()
    decoder_output, _ = encoder(input_tensor)
    loss_val = loss(decoder_output, target_tensor)
    loss_val.backward()
    optimizer.step()
    return loss_val.cpu().detach().numpy()


mnist_data = torchvision.datasets.MNIST(root="./mnist", download=True)
autoencoder = AutoEncoder(28, 28, 1).cuda()
loss = nn.BCELoss()
optm = torch.optim.SGD(autoencoder.parameters(), lr=1e-3, momentum=0.9)

dataloader = torch.utils.data.DataLoader(mnist_data.data[:1000].cuda(), batch_size=64, shuffle=True)

epochs = 40
lvs = []
for epoch in range(epochs):

    for imgs in dataloader:
        input_tensor = imgs.unsqueeze(1) / 255.0
        lv = train(input_tensor, input_tensor, autoencoder, loss, optm)
        lvs.append(lv)

plt.plot(lvs)
plt.show()

batch = next(iter(dataloader))
batch.unsqueeze_(1)
batch = batch / 255.0

i = 1
out_img, _ = autoencoder(batch[0:5])
for x in batch[0:5]:
    plt.subplot(5, 2, 2 * i - 1)
    plt.imshow(x[0].cpu())
    plt.subplot(5, 2, 2 * i)
    plt.imshow(out_img.cpu().detach()[i - 1, 0])  # 拷贝到cpu并detach
    i += 1

plt.show()

print(autoencoder)
