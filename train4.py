import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from src.models_variant4 import DenseNet121Classifier
from src.utils import get_transforms
from src.losses import FocalLoss
from tqdm import tqdm

# ===== CONFIG =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
BATCH_SIZE = 64
MODEL_PATH = "models/emotion_densenet_rgb_mixup.pth"
MIXUP_ALPHA = 0.4

# ===== DATASET (RGB) =====
train_ds = ImageFolder("train")
test_ds = ImageFolder("test")

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

train_loader = DataLoader(WrappedDataset(train_ds, get_transforms('train')), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(WrappedDataset(test_ds, get_transforms('test')), batch_size=BATCH_SIZE, shuffle=False)

# ===== MODEL / LOSS / OPTIM =====
model = DenseNet121Classifier(num_classes=7).to(DEVICE)
weights = torch.tensor([4.0, 5.5, 3.5, 1.0, 1.5, 2.0, 0.8]).to(DEVICE)
criterion = FocalLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)

# ===== MIXUP FUNCTION =====
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===== TRAIN LOOP =====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, y_a, y_b, lam = mixup_data(images, labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(outputs, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

# ===== SAVE MODEL =====
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
