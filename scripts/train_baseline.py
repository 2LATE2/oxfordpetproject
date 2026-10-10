"""
Train one baseline condition and report metrics.

Conditions:
  imbalanced   -- raw training split, no adjustment
  oversampling -- repeat minority rows until budget is met
  augmentation -- add augmented minority copies until budget is met

Usage:
    py scripts/train_baseline.py --condition imbalanced
    py scripts/train_baseline.py --condition oversampling
    py scripts/train_baseline.py --condition augmentation
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import (AUG_TRANSFORM, BREEDS, CLASS_TO_IDX, EVAL_TRANSFORM,
                     MINORITY, TRAIN_TRANSFORM, PetDataset, load_split)

ADDED_BUDGET = 55  # = 70 - 15
TRAIN_SEED = 0
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_rows(base_rows: pd.DataFrame, condition: str, images_dir: Path) -> tuple[pd.DataFrame, object]:
    minority_rows = base_rows[base_rows["breed"] == MINORITY].copy()

    if condition == "imbalanced":
        return base_rows.copy(), TRAIN_TRANSFORM

    if condition == "oversampling":
        rng = random.Random(TRAIN_SEED)
        extra = [minority_rows.sample(frac=1, random_state=rng.randint(0, 9999))
                 for _ in range((ADDED_BUDGET // len(minority_rows)) + 2)]
        extra_df = pd.concat(extra).head(ADDED_BUDGET)
        return pd.concat([base_rows, extra_df], ignore_index=True), TRAIN_TRANSFORM

    if condition == "augmentation":
        extra = minority_rows.sample(ADDED_BUDGET, replace=True, random_state=TRAIN_SEED)
        combined = pd.concat([base_rows, extra], ignore_index=True)
        return combined, None  # handled per-row in dataset

    raise ValueError(f"Unknown condition: {condition}")


class MixedTransformDataset(PetDataset):
    """Applies augmentation transform only to the extra minority rows."""
    def __init__(self, rows, images_dir, base_len, aug_transform, base_transform):
        super().__init__(rows, images_dir, transform=None)
        self.base_len = base_len
        self.aug_transform = aug_transform
        self.base_transform = base_transform

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        row = self.rows.iloc[idx]
        img = PILImage.open(self.images_dir / row["image_id"]).convert("RGB")
        label = CLASS_TO_IDX[row["breed"]]
        t = self.aug_transform if idx >= self.base_len else self.base_transform
        return t(img), label


def build_model(num_classes: int = 8) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    minority_idx = CLASS_TO_IDX[MINORITY]
    mau_recall = recall_score(all_labels, all_preds, labels=[minority_idx], average="micro", zero_division=0)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report     = classification_report(all_labels, all_preds, target_names=BREEDS, zero_division=0)
    return {"egyptian_mau_recall": mau_recall, "macro_f1": macro_f1, "report": report}


def main(args):
    set_seed(TRAIN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest    = Path(args.manifest)
    images_dir  = Path(args.images_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split(manifest, "train")
    val_rows   = load_split(manifest, "val")

    combined_rows, transform = build_train_rows(train_rows, args.condition, images_dir)

    if args.condition == "augmentation":
        train_ds = MixedTransformDataset(
            combined_rows, images_dir,
            base_len=len(train_rows),
            aug_transform=AUG_TRANSFORM,
            base_transform=TRAIN_TRANSFORM,
        )
    else:
        train_ds = PetDataset(combined_rows, images_dir, transform=transform)

    val_ds = PetDataset(val_rows, images_dir, transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nCondition: {args.condition}")
    print(f"Train size: {len(train_ds)}  Val size: {len(val_ds)}")

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_recall = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch:2d} | loss {total_loss/len(train_loader):.3f} "
              f"| Mau-recall {metrics['egyptian_mau_recall']:.3f} "
              f"| Macro-F1 {metrics['macro_f1']:.3f}")

        if metrics["egyptian_mau_recall"] >= best_recall:
            best_recall = metrics["egyptian_mau_recall"]
            best_metrics = metrics
            torch.save(model.state_dict(), results_dir / f"{args.condition}_best.pt")

    out = {
        "condition": args.condition,
        "egyptian_mau_recall": best_metrics["egyptian_mau_recall"],
        "macro_f1": best_metrics["macro_f1"],
    }
    print("\n=== Best validation results ===")
    print(best_metrics["report"])
    print(f"Egyptian Mau Recall : {best_metrics['egyptian_mau_recall']:.4f}")
    print(f"Macro-F1            : {best_metrics['macro_f1']:.4f}")

    with open(results_dir / f"{args.condition}_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Metrics saved to {results_dir}/{args.condition}_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition",   choices=["imbalanced", "oversampling", "augmentation"],
                        required=True)
    parser.add_argument("--manifest",    default="data/manifest.csv")
    parser.add_argument("--images_dir",  default="data/oxford-iiit-pet/images")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    main(args)
