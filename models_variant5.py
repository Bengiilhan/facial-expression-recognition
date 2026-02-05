import torch.nn as nn
from timm import create_model

class ConvNeXtTinyClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = create_model('convnext_tiny', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)