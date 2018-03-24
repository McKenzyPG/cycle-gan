"""
This contains network architecture. 
some part of the code is obtained from : 
https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/models.py
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, in_planes)
        self.norm1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(in_planes, in_planes)
        self.norm2 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(self.norm1(x))

        x = self.conv2(x)
        x = self.norm2(x)

        return self.relu(residual + x)


class GenM2S(nn.Module):
    def __init__(self, nc=1, nf=64, nout=3):
        super(GenM2S, self).__init__()

        self.first_block = nn.Sequential(
            nn.Conv2d(nc, nf, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.ReLU(inplace=True)
        )

        # downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(nf, nf*2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf*4),
            nn.ReLU(inplace=True)
        )

        # residual blocks

        self.res_blocks = nn.Sequential(
            ResidualBlock(nf*4),
            ResidualBlock(nf*4)
        )

        # upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nout, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.first_block(x)
        x = self.downsample(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


class GenS2M(nn.Module):
    def __init__(self, nc=3, nf=64, nout=1):
        super(GenS2M, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(nc, nf, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf),
            nn.ReLU(inplace=True)
        )

        # downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(nf, nf*2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(nf*4),
            nn.ReLU(inplace=True)
        )

        # residual blocks

        self.res_blocks = nn.Sequential(
            ResidualBlock(nf*4),
            ResidualBlock(nf*4)
        )

        # upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(nf*4, nf*2, 3, stride=2, bias=False),
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nout, 3, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.first_block(x)
        x = self.downsample(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, nc, nf=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(nc, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, 4, padding=1),
            nn.InstanceNorm2d(nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, 1, 4, padding=1)
        )

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
