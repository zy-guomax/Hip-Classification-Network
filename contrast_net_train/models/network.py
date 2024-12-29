import torch
import torch.nn as nn


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


class CLFC(nn.Module):
    def __init__(self, nf) -> None:
        super().__init__()
        self.zeta = nn.Conv2d(nf, nf//2, 1, 1, 0)
        self.g = nn.Conv2d(nf, nf//2, 1, 1, 0)
        self.phi = nn.Conv2d(nf//2, 1, 1, 1, 0)

    def forward(self, ft, fr):
        z_ft = self.zeta(ft)
        g_fr = self.g(fr)
        a = torch.sigmoid(self.phi(torch.relu(-z_ft*g_fr)))
        if self.training:
            return ft*a, (z_ft, g_fr)
        else:
            return ft*a


class Projection(nn.Module):
    def __init__(self, nf) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(nf, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256)
        )

    def forward(self, x):
        x = x.permute(1, 0, 2, 3).flatten(1).T
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Prediction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(128, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class NormNet(nn.Module):  # Normal Appearance Network, handles monomodal normal image
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

        return [e0, e1, e2, e3]  # return Fr1,Fr2,Fr3,Fr4


# 加载预训练的模型权重
pretrained_path = '/root/autodl-fs/encoder.pt'
norm_params = torch.load(pretrained_path)
encoder_feature = NormNet()
encoder_contrast = NormNet()
encoder_feature.load_state_dict(norm_params, strict=False)
encoder_contrast.load_state_dict(norm_params, strict=False)
# 让第一个encoder的参数不支持梯度回传
for param in encoder_feature.parameters():
    param.requires_grad = False
# 让第二个encoder的参数支持梯度回传
for param in encoder_contrast.parameters():
    param.requires_grad = True


class SegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_normal = encoder_feature
        self.encoder_patient = encoder_feature
        self.encoder_contrast = encoder_contrast
        self.mlp = nn.Sequential(
            nn.Linear(61440, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)
        self.clfc = nn.ModuleList()
        self.clfc.extend([
            CLFC(64),
            CLFC(128),
            CLFC(256),
            CLFC(512)
        ])
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x, rec):
        nf = self.encoder_normal(rec)  # fr = [fr1,fr2,fr3,fr4]
        pf = self.encoder_patient(x)
        e0 = self.downsample(self.encoder_patient.encoder[0](x))
        d0 = self.clfc[0](e0, nf[0])
        e1 = self.downsample(self.encoder_patient.encoder[1](0.5 * pf[0] + 0.5 * d0))
        d1 = self.clfc[1](e1, nf[1])
        e2 = self.downsample(self.encoder_patient.encoder[2](0.5 * pf[1] + 0.5 * d1))
        d2 = self.clfc[2](e2, nf[2])
        e3 = self.downsample(self.encoder_patient.encoder[3](0.5 * pf[2] + 0.5 * d2))
        d3 = self.clfc[3](e3, nf[3])
        d3 = d3.flatten(start_dim=1, end_dim=3)  # [32, 6240]
        d3 = self.mlp(d3)  # [32, 4, 39, 40]
        d3 = self.softmax(d3)
        return d3

