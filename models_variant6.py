import torch.nn as nn
from timm import create_model

class SwinTinyClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)