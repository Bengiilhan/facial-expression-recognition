import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale
from tqdm import tqdm

from src.models import EmotionResNet
from src.utils import get_transforms
from src.losses import FocalLoss

# Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
BATCH_SIZE = 64

# Datasets
train_dataset = ImageFolder("train", transform=Grayscale(num_output_channels=1))
test_dataset = ImageFolder("test", transform=Grayscale(num_output_channels=1))

# Wrap with albumentations
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset
class WrappedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(image=np.array(img))['image']
        return img, label

import numpy as np
train_loader = DataLoader(WrappedDataset(train_dataset, get_transforms('train')), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(WrappedDataset(test_dataset, get_transforms('test')), batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer
model = EmotionResNet().to(DEVICE)
weights = torch.tensor([4.0, 5.5, 3.5, 1.0, 1.5, 2.0, 0.8]).to(DEVICE)
criterion = FocalLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

# Save final model
torch.save(model.state_dict(), "models/emotion_resnet_cbam.pth")