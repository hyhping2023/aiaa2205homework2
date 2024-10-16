import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,dropout_prob=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout_prob = dropout_prob

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x))))
        x = self.bn2(self.pool(F.relu(self.conv2(x))))
        x = x.view(-1, 256 * 56 * 56)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_prob)
        x = self.fc2(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class inceptionA(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3x2red, ch3x3x2, pool_proj, 
                 stride=1, pool_type = 'avg'):
        super().__init__()
        if ch1x1 != 0:
            self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        else:
            self.branch1 = None
        
        if stride == 1:
            self.branch2 = nn.Sequential(
                BasicConv2d(in_channels, ch3x3red, kernel_size=1),
                BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
            )

            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, ch3x3x2red, kernel_size=1),
                BasicConv2d(ch3x3x2red, ch3x3x2, kernel_size=3, padding=1),
                BasicConv2d(ch3x3x2, ch3x3x2, kernel_size=3, padding=1)
            )
        else:
            self.branch2 = nn.Sequential(
                BasicConv2d(in_channels, ch3x3red, kernel_size=1, stride=1),
                BasicConv2d(ch3x3red, ch3x3, kernel_size=3, stride=2)
            )

            self.branch3 = nn.Sequential(
                BasicConv2d(in_channels, ch3x3x2red, kernel_size=1, stride=1),
                BasicConv2d(ch3x3x2red, ch3x3x2, kernel_size=3, padding=1),
                BasicConv2d(ch3x3x2, ch3x3x2, kernel_size=3, stride=2)
            )
        if pool_proj != 0:
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                BasicConv2d(in_channels, pool_proj, kernel_size=1)
            )
        else:
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        if self.branch1 is not None:
            branch1 = self.branch1(x)
            outputs = [branch1, branch2, branch3, branch4]
        else:
            outputs = [branch2, branch3, branch4]
        outputs = torch.cat(outputs, 1)

        return outputs
    

    
class inceptionB(nn.Module):
    def __init__(self, in_channels, ch1x1, ch7x7red, ch7x7, ch7x7dbl_red, dch7x7dbl, pool_proj,
                 conv_block=None, stride_num=1, pool_type='max'):
        super(inceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        if ch1x1 == 0:
            self.branch1 = None
        else:
            self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, stride=1, padding=0)

        if stride_num == 2:
            self.branch2 = nn.Sequential(
                conv_block(in_channels, ch7x7red, kernel_size=1),
                conv_block(ch7x7red, ch7x7, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(ch7x7, ch7x7, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(ch7x7, ch7x7, kernel_size=3, stride=2)
            )

            self.branch3 = nn.Sequential(
                conv_block(in_channels, ch7x7dbl_red, kernel_size=1),
                conv_block(ch7x7dbl_red, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=3, stride=2)
            )
        else:
            self.branch2 = nn.Sequential(
                conv_block(in_channels, ch7x7red, kernel_size=1),
                conv_block(ch7x7red, ch7x7, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(ch7x7, ch7x7, kernel_size=(7, 1), padding=(3, 0))
            )
            self.branch3 = nn.Sequential(
                conv_block(in_channels, ch7x7dbl_red, kernel_size=1),
                conv_block(ch7x7dbl_red, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(7, 1), padding=(3, 0)),
                conv_block(dch7x7dbl, dch7x7dbl, kernel_size=(1, 7), padding=(0, 3))
            )

        if pool_proj != 0:
            # avg pooling
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                conv_block(in_channels, pool_proj, kernel_size=1, stride=1)
            )
        else:
            # max pooling
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        if self.branch1 is not None:
            branch1 = self.branch1(x)
            outputs = [branch1, branch2, branch3, branch4]
        else:
            outputs = [branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class inceptionC(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch3x3dbl_red, dch3x3dbl, pool_proj,
                 conv_block=None, stride_num=1, pool_type='max'):
        super(inceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, stride=1, padding=0)

        self.branch3x3_1 = conv_block(in_channels, ch3x3red, kernel_size=1)
        self.branch3x3_2a = conv_block(ch3x3red, ch3x3, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(ch3x3red, ch3x3, kernel_size=(3, 1), padding=(1, 0))

        # double
        self.branch3x3dbl_1 = conv_block(in_channels, ch3x3dbl_red, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(ch3x3dbl_red, dch3x3dbl, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(dch3x3dbl, dch3x3dbl, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(dch3x3dbl, dch3x3dbl, kernel_size=(3, 1), padding=(1, 0))

        if pool_proj != 0:
            # avg pooling
            self.branch4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                conv_block(in_channels, pool_proj, kernel_size=1, stride=1, padding=0)
            )
        else:
            # max pooling
            self.branch4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _forward(self, x):
        branch1 = self.branch1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ] # double the number of channels
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ] # double the number of channels
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch4 = self.branch4(x)

        outputs = [branch1, branch3x3, branch3x3dbl, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class inceptionAux(nn.Module):
    def __init__(self, in_channels, avg_size=5, num_classes=10):
        super(inceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(3200, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((avg_size, avg_size))
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU(inplace=True)
        self.cateconv = BasicConv2d(in_channels, 128, kernel_size=1)

    def forward(self, x, categorize=True):
        if not categorize:
            self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
            self.relu = nn.ReLU()
            x = self.avgpool(x)
            x = self.cateconv(x)
            x = torch.flatten(x, 1)
            # N x 2048
            return x
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class PreConv(nn.Module):
    def __init__(self, ):
        super(PreConv, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 80, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(80, 192, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(192, 288, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        # x = N x 3 x 225 x 225
        x = self.conv1(x)
        # N x 32 x 112 x 112
        x = self.conv2(x)
        # N x 32 x 112 x 110
        x = self.conv3(x)
        # N x 64 x 110 x 110
        x = self.maxpool1(x)
        # N x 64 x 55 x 55
        x = self.conv4(x)
        # N x 80 x 53 x 53
        x = self.conv5(x)
        # N x 192 x 26 x 26
        x = self.conv6(x)
        # N x 288 x 26 x 26
        return x

class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, transform_input=False):
        super().__init__()
        self.transform_input = transform_input

        self.preconv = PreConv()
        self.aux1 = inceptionAux(768, 5, num_classes)
        self.aux2 = inceptionAux(1280, 5, num_classes)
        self.inception3a = inceptionA(288, 64, 64, 64, 64, 96, 64, pool_type='avg')
        self.inception3b = inceptionA(288, 64, 64, 96, 64, 96, 32, pool_type='avg')
        self.inception3c = inceptionA(288, 0, 128, 320, 64, 160, 0, pool_type='max', stride=2)

        self.inception5a = inceptionB(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5b = inceptionB(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5c = inceptionB(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5d = inceptionB(768, 192, 96, 160, 96, 160, 256, pool_type='avg')
        self.inception5e = inceptionB(768, 0, 128, 192, 128, 320, 0, pool_type='max', stride_num=2)

        self.inception2a = inceptionC(1280, 256, 128, 160, 128, 240, 224, pool_type='avg')
        self.inception2b = inceptionC(1280, 256, 96, 96, 96, 160, 0, pool_type='max')

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        self._initialize_weights()

    def forward(self, x):
        if self.transform_input:
            x = self._transform_input(x)
        # x = N x 3 x 225 x 225
        x = self.preconv(x)
        # N x 288 x 26 x 26
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        # N x 768 x 12 x 12

        aux1 = self.aux1(x)

        # N x 768 x 12 x 12
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.inception5c(x)
        x = self.inception5d(x)
        x = self.inception5e(x)
        # N x 1280 x 5 x 5

        aux2 = self.aux2(x)

        # N x 1280 x 5 x 5
        x = self.inception2a(x)
        x = self.inception2b(x)
        # N x 2048 x 5 x 5

        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x, aux1, aux2

    def forward_feature(self, x):
        if self.transform_input:
            x = self._transform_input(x)
        x = self.preconv(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        aux1 = self.aux1.forward(x, categorize=False)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.inception5c(x)
        x = self.inception5d(x)
        x = self.inception5e(x)

        aux2 = self.aux2(x, categorize=False)

        x = self.inception2a(x)
        x = self.inception2b(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        # N x 2048
        return x, aux1, aux2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

class PreRNN(nn.Module):
    def __init__(self, device='cuda', num_classes=10):
        super(PreRNN, self).__init__()
        self.CNN = GoogleNet().to(device)
        self.lstm = nn.LSTM(2048, 1024, num_layers=2, batch_first=True, dropout=0.5, device=device)
        self.multihead = nn.MultiheadAttention(2048, 8, dropout=0.5, batch_first=True, device=device)
        self.fc = nn.Linear(1024, num_classes, device=device)

    def _forward(self, x):
        # N X 30 X 2048
        x, _ = self.multihead(x, x, x)
        # N X 30 X 2048
        x, _ = self.lstm(x)

        # N X 1024
        x = x[:, -1, :]
        x = F.relu(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        #input will be N X seq x channel x height x width
        # N X 30 X 3 X 225 X 225
        results = [self.CNN.forward_feature(x[i]) for i in range(x.shape[0])]
        x, aux1, aux2 = torch.stack([results[i][0] for i in range(x.shape[0])], dim=0), \
                        torch.stack([results[i][1] for i in range(x.shape[0])], dim=0), \
                        torch.stack([results[i][2] for i in range(x.shape[0])], dim=0)
        x, aux1, aux2 = self._forward(x), self._forward(aux1), self._forward(aux2)
        return x, aux1, aux2
    
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=q.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
        
