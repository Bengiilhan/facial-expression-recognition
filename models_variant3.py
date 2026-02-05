import torch
import torch.nn as nn
from timm import create_model

class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)