import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm import create_model

from src.models import EmotionResNet
from src.models_variant2 import EmotionResNet34
from src.models_variant3 import EfficientNetB0Classifier
from src.models_variant4 import DenseNet121Classifier
from src.models_variant5 import ConvNeXtTinyClassifier

# === Config ===
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
CLASS_NAMES = ['angry','disgust','fear','happy','neutral','sad','surprise']

# === TTA transforms ===
TTA_TRANSFORMS = [
    A.Compose([A.Resize(48,48), A.HorizontalFlip(p=1.0), A.Normalize(), ToTensorV2()]),
    A.Compose([A.Resize(48,48), A.RandomBrightnessContrast(p=1.0), A.Normalize(), ToTensorV2()]),
    A.Compose([A.Resize(48,48), A.GaussianBlur(p=1.0), A.Normalize(), ToTensorV2()]),
    A.Compose([A.Resize(48,48), A.Rotate(limit=20,p=1.0), A.Normalize(), ToTensorV2()]),
    A.Compose([A.Resize(48,48), A.Normalize(), ToTensorV2()])
]

# === MetaLearner ===
class MetaLearner(nn.Module):
    def __init__(self): 
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(42,64), nn.ReLU(), nn.Linear(64,7)
        )
    def forward(self,x): return self.model(x)

# === Dataset ===
class WrappedDataset(Dataset):
    def __init__(self, folder): 
        self.gray = ImageFolder(folder)
        self.rgb  = ImageFolder(folder)
    def __len__(self): return len(self.gray)
    def __getitem__(self,idx):
        g,lab1 = self.gray[idx]
        r,lab2 = self.rgb[idx]
        assert lab1==lab2
        return np.array(g.convert('L')), np.array(r.convert('RGB')), lab1

# === DataLoader ===
loader = DataLoader(WrappedDataset("test"), batch_size=BATCH_SIZE, shuffle=False)

# === Load base models ===
model1 = EmotionResNet().to(DEVICE)
model1.load_state_dict(torch.load("models/emotion_resnet_cbam.pth",map_location=DEVICE)); model1.eval()
model2 = EmotionResNet34(in_channels=1).to(DEVICE)
model2.load_state_dict(torch.load("models/emotion_resnet34_se.pth",map_location=DEVICE)); model2.eval()
model3 = EfficientNetB0Classifier(num_classes=7).to(DEVICE)
model3.load_state_dict(torch.load("models/emotion_efficientnet_rgb_mixup.pth",map_location=DEVICE)); model3.eval()
model4 = DenseNet121Classifier(num_classes=7).to(DEVICE)
model4.load_state_dict(torch.load("models/emotion_densenet_rgb_mixup.pth",map_location=DEVICE)); model4.eval()
model5 = ConvNeXtTinyClassifier(num_classes=7).to(DEVICE)
model5.load_state_dict(torch.load("models/emotion_convnext_tiny.pth",map_location=DEVICE)); model5.eval()
model6 = create_model('swin_tiny_patch4_window7_224',pretrained=False,num_classes=7).to(DEVICE)
model6.load_state_dict(torch.load("models/swin_cutmix.pth",map_location=DEVICE)); model6.eval()

meta_model = MetaLearner().to(DEVICE)
meta_model.load_state_dict(torch.load("models/meta_learner2.pth", map_location=DEVICE))
meta_model.eval()

# === Inference with Hard TTA ===
all_preds, all_labels = [], []
with torch.no_grad():
    for gray_np, rgb_np, labels in loader:
        B = len(labels)
        logits_sum = torch.zeros((B,42),device=DEVICE)
        # gray_np: numpy array? Actually DataLoader will convert list of numpy to tensor of shape (B,H,W) dtype maybe object? 
        # We iterate manually:
        for tta in TTA_TRANSFORMS:
            gray_aug = torch.stack([
                tta(image=np.expand_dims(g,2))['image'] for g in gray_np.numpy()
            ]).to(DEVICE)
            rgb_aug = torch.stack([
                tta(image=r)           ['image'] for r in rgb_np.numpy()
            ]).to(DEVICE)
            out1 = torch.softmax(model1(gray_aug),dim=1)
            out2 = torch.softmax(model2(gray_aug),dim=1)
            out3 = torch.softmax(model3(rgb_aug),dim=1)
            out4 = torch.softmax(model4(rgb_aug),dim=1)
            out5 = torch.softmax(model5(rgb_aug),dim=1)
            rgb_swin = F.interpolate(rgb_aug, size=(224,224), mode='bilinear', align_corners=False)
            out6 = torch.softmax(model6(rgb_swin),dim=1)
            logits_sum += torch.cat([out1,out2,out3,out4,out5,out6],dim=1)
        avg_logits = logits_sum / len(TTA_TRANSFORMS)
        preds = torch.argmax(meta_model(avg_logits),dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

# === Report ===
print("\nMeta-Learner Final Evaluation with Hard TTA\n")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,cmap='Blues')
plt.title("Meta Confusion Matrix (TTA)")
plt.xlabel("Pred"); plt.ylabel("True")
plt.tight_layout(); plt.savefig("meta_confusion_matrix_tta.png"); plt.show()
