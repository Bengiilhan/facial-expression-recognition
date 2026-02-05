import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale
from src.models_variant2 import EmotionResNet34
from src.utils import get_transforms
from src.losses import FocalLoss
import numpy as np
from tqdm import tqdm

# ===== CONFIG =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
BATCH_SIZE = 64
GRAYSCALE = True  # Change to False for RGB
MODEL_PATH = "models/emotion_resnet34_se.pth"

# ===== DATASET SETUP =====
if GRAYSCALE:
    base_transform = Grayscale(num_output_channels=1)
    in_channels = 1
else:
    base_transform = None
    in_channels = 3

train_ds = ImageFolder("train", transform=base_transform)
test_ds = ImageFolder("test", transform=base_transform)

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
        import numpy as np
        img = self.transform(image=np.array(img))['image']
        return img, label

train_loader = DataLoader(WrappedDataset(train_ds, get_transforms('train')), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(WrappedDataset(test_ds, get_transforms('test')), batch_size=BATCH_SIZE, shuffle=False)

# ===== MODEL / LOSS / OPTIM =====
model = EmotionResNet34(num_classes=7, in_channels=in_channels).to(DEVICE)
weights = torch.tensor([4.0, 5.5, 3.5, 1.0, 1.5, 2.0, 0.8]).to(DEVICE)
criterion = FocalLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ===== TRAIN LOOP =====
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

# ===== SAVE IT =====
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
