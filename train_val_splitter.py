import os
import shutil
import random
from tqdm import tqdm

SRC_DIR = "train"  # original train folder
DEST_DIR = "train_val_split"
VAL_RATIO = 0.1    # 10% for validation

# Create destination dirs
for split in ['train', 'val']:
    for cls in os.listdir(SRC_DIR):
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

# Split each class
for cls in os.listdir(SRC_DIR):
    class_dir = os.path.join(SRC_DIR, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]
    train_images = images[val_count:]

    for img in tqdm(val_images, desc=f"{cls} → val"):
        src = os.path.join(class_dir, img)
        dst = os.path.join(DEST_DIR, 'val', cls, img)
        shutil.copy(src, dst)

    for img in tqdm(train_images, desc=f"{cls} → train"):
        src = os.path.join(class_dir, img)
        dst = os.path.join(DEST_DIR, 'train', cls, img)
        shutil.copy(src, dst)

print("✅ Done. Safely created train/val split in 'train_val_split/'")
