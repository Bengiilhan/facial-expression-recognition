import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.models_variant5 import ConvNeXtTinyClassifier
from src.utils import get_transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
BATCH_SIZE = 64
SAVE_PATH = "models/emotion_convnext_tiny.pth"

train_ds = ImageFolder("train_val_split/train")
val_ds = ImageFolder("train_val_split/val")

class RGBWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img, dtype=np.uint8)

        if img.ndim == 2:  # grayscale → (H, W)
            rgb = np.stack([img] * 3, axis=-1)  # → (H, W, 3)
        else:
            rgb = img  # already RGB

        rgb = rgb.astype(np.uint8)
        return self.transform(image=rgb)['image'], label

train_loader = DataLoader(RGBWrapper(train_ds, get_transforms('train')), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(RGBWrapper(val_ds, get_transforms('val')), batch_size=BATCH_SIZE)

model = ConvNeXtTinyClassifier(num_classes=7).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
