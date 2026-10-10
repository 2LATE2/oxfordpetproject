"""Shared dataset utilities for all training conditions."""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "Egyptian_Mau", "Ragdoll", "Russian_Blue", "Siamese",
]
CLASS_TO_IDX = {b: i for i, b in enumerate(BREEDS)}
MINORITY = "Egyptian_Mau"

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALIZE,
])

AUG_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])


class PetDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, images_dir: Path, transform=None):
        self.rows = rows.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.rows)

    def load(self, idx):
        row = self.rows.iloc[idx]
        img = Image.open(self.images_dir / row["image_id"]).convert("RGB")
        return img, CLASS_TO_IDX[row["breed"]]

    def __getitem__(self, idx):
        img, label = self.load(idx)
        return self.transform(img), label


def load_split(manifest: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    return df[df["split"] == split].copy()
