import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.blocks(x)


class NormNetEncoder(nn.Module):  # Normal Appearance Network, handles monomodal normal image
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.extend([
            ConvBlock(1, 64),  # monomodal determines the first param to be 1
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512)
        ])
        self.downsample = nn.MaxPool2d(2)  # retain the largest number in 2x2 region, saves feature & reduce file size

    def forward(self, x):
        e0 = self.downsample(self.encoder[0](x))
        e1 = self.downsample(self.encoder[1](e0))
        e2 = self.downsample(self.encoder[2](e1))
        e3 = self.downsample(self.encoder[3](e2))

        return x, e1, e2, e3  # return Fr1,Fr2,Fr3,Fr4


class NormNetDecoder(nn.Module):  # Normal Appearance Network, handles monomodal normal image
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.ModuleList()
        self.decoder.extend([
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.ConvTranspose2d(768, 256, 4, 2, 1),
            nn.ConvTranspose2d(384, 128, 4, 2, 1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1)
        ])
        self.downsample = nn.MaxPool2d(2)  # retain the largest number in 2x2 region, saves feature & reduce file size
        self.conv_seg = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, e1, e2, e3):

        d0 = self.decoder[0](e3)
        d1 = self.decoder[1](torch.cat([d0, e2], dim=1))  # concatenate features before downsampling
        d2 = self.decoder[2](torch.cat([d1, e1], dim=1))
        d3 = self.decoder[3](d2)
        d4 = self.conv_seg(d3)

        return d4  # return Fr1,Fr2,Fr3,Fr4


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NormNetEncoder()
        self.decoder = NormNetDecoder()

    def forward(self, xs):
        e0, e1, e2, e3 = self.encoder(xs)
        d4 = self.decoder(e1, e2, e3)
        return e0, d4
