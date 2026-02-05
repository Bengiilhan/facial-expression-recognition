import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import sys
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from timm import create_model
from tqdm import tqdm

from src.utils import get_large_transforms

# ===== Config =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
BATCH_SIZE = 16
LR = 3e-5
NUM_CLASSES = 7
SAVE_PATH = 'models/swin_cutmix.pth'

# ===== Custom Dataset (real RGB + 224x224) =====
class RGBDataset(Dataset):
    def __init__(self, dataset, transform):
        self.samples = dataset.samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = np.array(img)
        transformed = self.transform(image=img)
        return transformed['image'], label

# ===== CutMix Function =====
def cutmix(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size()[0])
    target_a = labels
    target_b = labels[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

    return images, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# ===== Data Loaders =====
train_ds = ImageFolder("train_val_split/train")
val_ds = ImageFolder("train_val_split/val")

train_loader = DataLoader(RGBDataset(train_ds, get_large_transforms('train')), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RGBDataset(val_ds, get_large_transforms('val')), batch_size=BATCH_SIZE)

# ===== Model =====
model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ===== Training =====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Apply CutMix
        images, targets_a, targets_b, lam = cutmix(images, labels)
        outputs = model(images)
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# ===== Save Model =====
torch.save(model.state_dict(), SAVE_PATH)
print(f"\nSwin Transformer model saved to {SAVE_PATH}")
