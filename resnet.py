import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TinyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, relu:bool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        if self.relu is None:
            return self.bn(self.conv(x))
        return self.relu(self.bn(self.conv(x)))
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = TinyConvBlock(in_channels, mid_channels, stride=1, kernel_size=1, padding=0)
        if in_channels != out_channels:
            self.conv2 = TinyConvBlock(mid_channels, mid_channels, stride=2, kernel_size=3, padding=1)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        else:
            self.conv2 = TinyConvBlock(mid_channels, mid_channels, stride=1, kernel_size=3, padding=1)
            self.shortcut = None
        self.conv3 = TinyConvBlock(mid_channels, out_channels, stride=1, kernel_size=1, padding=0, relu=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return self.relu(x3 + x_s)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, layers:int = 5):
        super(ResnetBlock, self).__init__()
        self.layers = layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.ModuleList(
            [ConvBlock(in_channels, mid_channels, out_channels)] + 
            [ConvBlock(out_channels, mid_channels, out_channels) for _ in range(layers-1)]
        )
        
    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x
        

class MyResnetCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(MyResnetCNN, self).__init__()
        self.embedding = nn.Sequential(
            # N X 3 X 224 X 224
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3),
            # N X 64 X 112 X 112
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            # N X 64 X 56 X 56
        )
        self.layers = nn.ModuleList(
            [   ResnetBlock(64, 64, 256, layers=3),
                ResnetBlock(256, 128, 512, layers=8),
                ResnetBlock(512, 256, 1024, layers=36),
                ResnetBlock(1024, 512, 2048, layers=3), ]
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        result = self.fc(out)
        return result
    