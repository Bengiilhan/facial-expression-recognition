import torch
import torch.nn as nn
from torchvision.models import resnet34
from timm.layers import SqueezeExcite

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = SqueezeExcite(channels, rd_ratio=1/reduction)

    def forward(self, x):
        return self.se(x)

class EmotionResNet34(nn.Module):
    def __init__(self, num_classes=7, in_channels=3):
        super().__init__()
        base = resnet34(weights=None)  # We'll train from scratch or load pretrained manually
        base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.se = SEBlock(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.se(x)
        x = self.pool(x)
        return self.head(x)