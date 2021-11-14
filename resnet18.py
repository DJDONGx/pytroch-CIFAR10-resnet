import torch
from torch import nn


def conv3x3(in_channels,out_channels,stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3, stride = stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, shotcut = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shotcut = shotcut

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shotcut:
            residual = self.shotcut(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layer, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3,16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 16, layer[0])
        self.layer2 = self.make_layer(block, 32, layer[1], 2)
        self.layer3 = self.make_layer(block, 64, layer[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride = 1):
        shotcut = None
        if(stride != 1) or (self.in_channels != out_channels):
            shotcut = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride = stride,padding=1),
                nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, shotcut))

        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x