# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from timm.layers import SqueezeExcite

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = SqueezeExcite(channels, rd_ratio=1/reduction)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = resnet18(pretrained=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.cbam = CBAM(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        return self.head(x)
