import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(BasicBlock2D, self).__init__()
        mid_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SEBlock(nn.Module):  # Squeeze-and-Excitation block for attention
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class OutputPredictor(nn.Module):
    def __init__(self):
        super(OutputPredictor, self).__init__()
        self.init_conv = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.second_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res_block1 = BasicBlock2D(64, 64)
        self.se_block1 = SEBlock(64)  # Adding SE block
        self.res_block2 = BasicBlock2D(64, 128, stride=2)
        self.res_block3 = BasicBlock2D(128, 256, stride=2)
        self.res_block4 = BasicBlock2D(256, 512, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1_2d = nn.Linear(512, 1024)

        self.fc1_1d_real = nn.Linear(20, 512)
        self.fc1_1d_imag = nn.Linear(20, 512)

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 40)

    def forward(self, x2d, x1d_real, x1d_imag):
        x2d = x2d.unsqueeze(1)
        x2d = F.relu(self.init_conv(x2d))
        x2d = F.relu(self.second_conv(x2d))
        x2d = self.res_block1(x2d)
        x2d = self.se_block1(x2d)  # Apply SE block
        x2d = self.res_block2(x2d)
        x2d = self.res_block3(x2d)
        x2d = self.res_block4(x2d)
        x2d = self.avg_pool(x2d)
        x2d = x2d.view(-1, 512)
        x2d = F.relu(self.fc1_2d(x2d))
        x1d_real = F.relu(self.fc1_1d_real(x1d_real))
        x1d_imag = F.relu(self.fc1_1d_imag(x1d_imag))
        x = torch.cat((x2d, x1d_real, x1d_imag), dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
