# src/train_meta_learner.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, random_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
NUM_CLASSES = 7
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === Load meta logits and labels ===
logits = np.load("meta_logits.npy")  # shape: [N, 6, 7]
labels = np.load("meta_labels.npy")  # shape: [N]
X = logits.reshape(logits.shape[0], -1)
y = labels

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# === Split into train and validation ===
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# === Meta-learner definition ===
class MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )
    def forward(self, x):
        return self.model(x)

model = MetaLearner().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# === Training loop with validation ===
train_accuracies = []
val_accuracies = []
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    train_accuracies.append(train_acc)

    # === Validation ===
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(DEVICE)
            outputs = model(X_val)
            val_preds.extend(outputs.argmax(1).cpu().numpy())
            val_targets.extend(y_val.numpy())
    val_acc = accuracy_score(val_targets, val_preds)
    val_accuracies.append(val_acc)

    # === Logging ===
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/meta_learner2.pth")

# === Final evaluation on all data ===
model.load_state_dict(torch.load("models/meta_learner2.pth"))
model.eval()
with torch.no_grad():
    final_preds = model(X_tensor.to(DEVICE)).argmax(1).cpu().numpy()
    print("\nFinal Evaluation on All Data")
    print(classification_report(y, final_preds, target_names=EMOTIONS))
    overall_acc = accuracy_score(y, final_preds)
    print(f"\nOverall Accuracy on All Meta-Logits: {overall_acc * 100:.2f}%")

# === Print final summary ===
print("\nTraining Summary:")
print(f"   ➤ Final Train Accuracy   : {train_accuracies[-1]*100:.2f}%")
print(f"   ➤ Final Validation Accuracy: {val_accuracies[-1]*100:.2f}%")
print(f"   ➤ Best Validation Accuracy : {max(val_accuracies)*100:.2f}%")
