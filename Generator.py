import torch
import torch.nn as nn

bias = True


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        self.pre_solu = Pre_solu(in_channels, out_channels)
        self.color_encoder = Color_encoder(in_channels, out_channels)
        self.conv1 = BasicCov(out_channels * 2, out_channels, 5, 1)
        self.conv2 = nn.Conv2d(out_channels, in_channels, 5, 1, 2)

    def init_colorpanel(self, colorpanel, input):
        norm = (colorpanel + 2) / 4
        self.color = torch.ones([input.size[-1], input.size[-2]]) * norm

    def forward(self, input):
        in_1 = self.pre_solu(input)
        in_2 = self.color_encoder(input, self.color)
        x = self.conv1(torch.cat([in_1, in_2], dim=1))
        x = self.conv2(x)
        return x


class Pre_solu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pre_solu, self).__init__()
        self.conv1 = BasicCov(in_channels, out_channels, 3, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Color_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Color_encoder, self).__init__()
        self.conv1 = BasicCov(in_channels + 1, out_channels, 5, 2)
        self.conv2 = BasicCov(out_channels * 2, out_channels * 4, 3, 1)
        self.conv3 = BasicCov(out_channels * 4, out_channels * 2, 3, 1)
        self.deconv = nn.ConvTranspose2d(out_channels * 2, out_channels, 5, 2, 2)

    def forward(self, input, color):
        x = torch.cat([input, color], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.deconv(x)

        return x


class BasicCov(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(BasicCov, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x
