import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.models_variant4 import DenseNet121Classifier
from src.utils import get_transforms

# ===== CONFIG =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
MODEL_PATH = "models/emotion_densenet_rgb_mixup.pth"
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ===== DATASET (RGB) =====
test_raw = ImageFolder("test")

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

test_loader = DataLoader(
    WrappedDataset(test_raw, get_transforms('test')),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ===== Load Model =====
model = DenseNet121Classifier(num_classes=7).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== Evaluate =====
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ===== Report =====
print("\n🧾 DenseNet121 Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)

# ===== Plot Confusion Matrix =====
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.title("DenseNet121 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("densenet_confusion_matrix.png")
plt.show()
