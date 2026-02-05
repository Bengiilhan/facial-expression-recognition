import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)