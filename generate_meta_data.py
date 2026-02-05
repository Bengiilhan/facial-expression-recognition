# src/generate_meta_data.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale, Resize
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from src.models import EmotionResNet
from src.models_variant2 import EmotionResNet34
from src.models_variant3 import EfficientNetB0Classifier
from src.models_variant4 import DenseNet121Classifier
from src.models_variant5 import ConvNeXtTinyClassifier
from timm import create_model
from src.utils import get_transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64

gray_dataset = ImageFolder("test", transform=Grayscale(num_output_channels=1))
rgb_dataset = ImageFolder("test")

class WrappedDataset(Dataset):
    def __init__(self, gray_ds, rgb_ds, transform):
        self.gray_ds = gray_ds
        self.rgb_ds = rgb_ds
        self.transform = transform

    def __len__(self):
        return len(self.gray_ds)

    def __getitem__(self, idx):
        gray_img, label1 = self.gray_ds[idx]
        rgb_img, label2 = self.rgb_ds[idx]
        assert label1 == label2
        return {
            'gray': self.transform(image=np.array(gray_img))['image'],
            'rgb': self.transform(image=np.array(rgb_img))['image'],
            'label': label1
        }

loader = DataLoader(
    WrappedDataset(gray_dataset, rgb_dataset, get_transforms('test')),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load models
model1 = EmotionResNet().to(DEVICE)
model1.load_state_dict(torch.load("models/emotion_resnet_cbam.pth", map_location=DEVICE))
model1.eval()

model2 = EmotionResNet34(in_channels=1).to(DEVICE)
model2.load_state_dict(torch.load("models/emotion_resnet34_se.pth", map_location=DEVICE))
model2.eval()

model3 = EfficientNetB0Classifier(num_classes=7).to(DEVICE)
model3.load_state_dict(torch.load("models/emotion_efficientnet_rgb_mixup.pth", map_location=DEVICE))
model3.eval()

model4 = DenseNet121Classifier(num_classes=7).to(DEVICE)
model4.load_state_dict(torch.load("models/emotion_densenet_rgb_mixup.pth", map_location=DEVICE))
model4.eval()

model5 = ConvNeXtTinyClassifier(num_classes=7).to(DEVICE)
model5.load_state_dict(torch.load("models/emotion_convnext_tiny.pth", map_location=DEVICE))
model5.eval()

model6 = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=7).to(DEVICE)
model6.load_state_dict(torch.load("models/swin_cutmix.pth", map_location=DEVICE))
model6.eval()

resize_224 = Resize((224, 224))

# Inference & Save softmax logits
all_logits, all_labels = [], []

with torch.no_grad():
    for batch in loader:
        gray = batch['gray'].to(DEVICE)
        rgb = batch['rgb'].to(DEVICE)
        labels = batch['label']

        out1 = torch.softmax(model1(gray), dim=1).cpu().numpy()
        out2 = torch.softmax(model2(gray), dim=1).cpu().numpy()
        out3 = torch.softmax(model3(rgb), dim=1).cpu().numpy()
        out4 = torch.softmax(model4(rgb), dim=1).cpu().numpy()
        out5 = torch.softmax(model5(rgb), dim=1).cpu().numpy()
        out6 = torch.softmax(model6(resize_224(rgb)), dim=1).cpu().numpy()

        # Stack shape: (batch, 6, 7)
        stacked = np.stack([out1, out2, out3, out4, out5, out6], axis=1)
        all_logits.append(stacked)
        all_labels.extend(labels.numpy())

logits = np.concatenate(all_logits, axis=0)      # shape: [N, 6, 7]
labels = np.array(all_labels)                    # shape: [N]

np.save("meta_logits.npy", logits)
np.save("meta_labels.npy", labels)

print("Saved meta_logits.npy and meta_labels.npy")
