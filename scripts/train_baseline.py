"""
Train one baseline condition and report validation metrics.

Conditions:
  imbalanced   -- raw training split, no adjustment
  oversampling -- repeat minority rows until budget is met
  augmentation -- add augmented minority copies until budget is met

Backbones are ImageNet-pretrained weights downloaded from the Hugging Face Hub via timm,
kept frozen (and in eval mode, so BatchNorm statistics stay fixed); only a linear head is trained.
Training length is fixed (no epoch selection on validation), and every condition is run
for several training seeds.

Usage:
    py scripts/train_baseline.py --condition imbalanced
    py scripts/train_baseline.py --condition oversampling --model efficientnet_b0
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (AUG_TRANSFORM, BREEDS, CLASS_TO_IDX, EVAL_TRANSFORM,
                     MINORITY, TRAIN_TRANSFORM, PetDataset, load_split)

ADDED_BUDGET = 55  # = 70 - 15
TRAIN_SEEDS = [0, 1, 2]
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

# Hugging Face Hub checkpoints (timm)
HF_MODELS = {
    "resnet18":        "hf-hub:timm/resnet18.tv_in1k",
    "efficientnet_b0": "hf-hub:timm/efficientnet_b0.ra_in1k",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_rows(base_rows: pd.DataFrame, condition: str, seed: int) -> pd.DataFrame:
    """Return the base rows followed by exactly ADDED_BUDGET extra minority rows (if any)."""
    if condition == "imbalanced":
        return base_rows.copy()

    minority_rows = base_rows[base_rows["breed"] == MINORITY]
    # Cycle through the minority images as evenly as possible: 55 = 3 x 15 + 10
    reps = -(-ADDED_BUDGET // len(minority_rows))
    extra = pd.concat([minority_rows.sample(frac=1, random_state=seed + k) for k in range(reps)])
    extra = extra.head(ADDED_BUDGET)

    if condition in ("oversampling", "augmentation"):
        return pd.concat([base_rows, extra], ignore_index=True)
    raise ValueError(f"Unknown condition: {condition}")


class MixedTransformDataset(PetDataset):
    """Applies the stronger augmentation transform only to the extra minority rows."""
    def __init__(self, rows, images_dir, base_len, aug_transform, base_transform):
        super().__init__(rows, images_dir, transform=base_transform)
        self.base_len = base_len
        self.aug_transform = aug_transform

    def __getitem__(self, idx):
        img, label = self.load(idx)
        t = self.aug_transform if idx >= self.base_len else self.transform
        return t(img), label


class FrozenBackboneClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 8):
        super().__init__()
        self.backbone = timm.create_model(HF_MODELS[model_name], pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen features: never update BatchNorm running stats
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        return self.head(feats)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    minority_idx = CLASS_TO_IDX[MINORITY]
    mau_recall = recall_score(all_labels, all_preds, labels=[minority_idx], average="micro", zero_division=0)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report     = classification_report(all_labels, all_preds, labels=list(range(len(BREEDS))),
                                       target_names=BREEDS, zero_division=0)
    return {"egyptian_mau_recall": float(mau_recall), "macro_f1": float(macro_f1), "report": report}


def run_seed(args, seed, train_rows, val_rows, images_dir, device):
    set_seed(seed)
    combined_rows = build_train_rows(train_rows, args.condition, seed)

    if args.condition == "augmentation":
        train_ds = MixedTransformDataset(combined_rows, images_dir, base_len=len(train_rows),
                                         aug_transform=AUG_TRANSFORM, base_transform=TRAIN_TRANSFORM)
    else:
        train_ds = PetDataset(combined_rows, images_dir, transform=TRAIN_TRANSFORM)
    val_ds = PetDataset(val_rows, images_dir, transform=EVAL_TRANSFORM)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=args.workers, persistent_workers=args.workers > 0)

    print(f"\n[{args.model} | {args.condition} | seed {seed}] "
          f"train {len(train_ds)} ({(combined_rows['breed'] == MINORITY).sum()} Mau rows), val {len(val_ds)}")

    model = FrozenBackboneClassifier(args.model).to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

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
        print(f"  Epoch {epoch:2d} | loss {total_loss / len(train_loader):.3f}")

    metrics = evaluate(model, val_loader, device)
    print(metrics["report"])
    print(f"  Mau recall {metrics['egyptian_mau_recall']:.3f} | Macro-F1 {metrics['macro_f1']:.3f}")
    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    images_dir  = Path(args.images_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_split(Path(args.manifest), "train")
    val_rows   = load_split(Path(args.manifest), "val")

    per_seed = []
    for seed in args.seeds:
        m = run_seed(args, seed, train_rows, val_rows, images_dir, device)
        per_seed.append({"seed": seed, "egyptian_mau_recall": m["egyptian_mau_recall"],
                         "macro_f1": m["macro_f1"]})

    recalls = [r["egyptian_mau_recall"] for r in per_seed]
    f1s     = [r["macro_f1"] for r in per_seed]
    out = {
        "model": args.model,
        "hf_checkpoint": HF_MODELS[args.model],
        "condition": args.condition,
        "split": "val",
        "epochs": EPOCHS,
        "seeds": args.seeds,
        "egyptian_mau_recall_mean": float(np.mean(recalls)),
        "egyptian_mau_recall_std":  float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std":  float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "per_seed": per_seed,
    }
    out_path = results_dir / f"{args.model}_{args.condition}_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== {args.model} / {args.condition}: Mau recall {out['egyptian_mau_recall_mean']:.3f} "
          f"± {out['egyptian_mau_recall_std']:.3f} | Macro-F1 {out['macro_f1_mean']:.3f} "
          f"± {out['macro_f1_std']:.3f} ===")
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition",   choices=["imbalanced", "oversampling", "augmentation"],
                        required=True)
    parser.add_argument("--model",       choices=list(HF_MODELS), default="resnet18")
    parser.add_argument("--seeds",       type=int, nargs="+", default=TRAIN_SEEDS)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--manifest",    default="data/manifest.csv")
    parser.add_argument("--images_dir",  default="data/oxford-iiit-pet/images")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    main(args)
