from torch import nn
NUM_CLASSES = 7
# -----------------------------
# Model: CNN from scratch
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class EmotionCNN(nn.Module):
    """
    4 convolutional blocks (2 conv each) + global pooling + 2-layer classifier head.
    Works with any input HxW >= 16 because of AdaptiveAvgPool2d(1).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.features = nn.Sequential(
            ConvBlock(3, c, dropout=0.10),        # 2 conv
            ConvBlock(c, 2 * c, dropout=0.15),    # 2 conv
            ConvBlock(2 * c, 4 * c, dropout=0.20),# 2 conv
            ConvBlock(4 * c, 8 * c, dropout=0.25) # 2 conv
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * c, 4 * c),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.50),
            nn.Linear(4 * c, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# -------------------------------
# RESNET-16 (96x96 friendly)
# -------------------------------

# @title Moderl
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# -------------------------------
# RESNET-16 (96x96 friendly)
# -------------------------------
class ResNet16_96(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # Conv initiale (pas de downsampling)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Stages ResNet
        self.layer1 = self._make_layer(64,  2, stride=1)  # 96x96
        self.layer2 = self._make_layer(128, 2, stride=2)  # 48x48
        self.layer3 = self._make_layer(256, 2, stride=2)  # 24x24
        self.layer4 = self._make_layer(512, 1, stride=2)  # 12x12

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []

        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


