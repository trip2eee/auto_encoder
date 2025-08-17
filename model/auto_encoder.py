import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # 5x5
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        # 7x7
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        # 9x9
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)


    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.act2(y)

        y = self.conv3(y)
        y = self.act3(y)

        y = self.conv4(y)
        
        return y

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, padding=0)
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.act2(y)

        y = self.conv3(y)
        y = self.act3(y)

        return y

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        y = self.encoder(x)
        y = self.decoder(y)

        return y
    
