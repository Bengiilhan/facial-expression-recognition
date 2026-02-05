# src/utils.py
from albumentations.pytorch import ToTensorV2
import albumentations as A

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(48, 48),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(48, 48),
            A.Normalize(),
            ToTensorV2()
        ])

def get_large_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(p=0.3),
            A.GaussianBlur(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=56, max_width=56, p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])