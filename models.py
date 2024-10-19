import torch
from torch import nn
import torch.nn.functional as F
import torch.utils

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


    def judgementTraining(self, input_tensor):
        '''
        input should be : Batch x Depth xChannel x Height x Width
        '''
        batch_size = input_tensor.shape[0]
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-3], input_tensor.shape[-2], input_tensor.shape[-1])
        outputs, aux1, aux2 = self.forward(input_tensor)
        outputs = outputs.reshape(batch_size, -1, outputs.shape[-1]) # batchsize x depth x classes
        # print(outputs.shape)
        outputs = torch.mean(outputs, dim=1)
        aux1 = aux1.reshape(batch_size, -1, aux1.shape[-1])
        aux2 = aux2.reshape(batch_size, -1, aux2.shape[-1])
        aux1 = torch.mean(aux1, dim=1)
        aux2 = torch.mean(aux2, dim=1) # batchsize x classes

        return outputs, aux1, aux2


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
    
