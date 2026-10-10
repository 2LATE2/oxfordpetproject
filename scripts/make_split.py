"""
Split Oxford-IIIT Pet images into train/val/test and save a manifest CSV.
Run once. Never re-run after seeing test results.

Usage:
    py scripts/make_split.py --data_root data/oxford-iiit-pet
"""

import argparse
import csv
import random
from pathlib import Path

BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "Egyptian_Mau", "Ragdoll", "Russian_Blue", "Siamese",
]
MINORITY = "Egyptian_Mau"

TRAIN_MAJORITY = 70
TRAIN_MINORITY = 15
VAL_PER_CLASS  = 20
TEST_PER_CLASS = 30

SEED = 42


def collect_images(images_dir: Path, breed: str) -> list[Path]:
    prefix = breed + "_"
    imgs = sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                  and p.stem.startswith(prefix))
    return imgs


def main(data_root: Path, out_csv: Path) -> None:
    images_dir = data_root / "images"
    assert images_dir.exists(), f"Images folder not found: {images_dir}"

    rng = random.Random(SEED)
    rows: list[dict] = []

    for breed in BREEDS:
        imgs = collect_images(images_dir, breed)
        need = (TRAIN_MINORITY if breed == MINORITY else TRAIN_MAJORITY) + VAL_PER_CLASS + TEST_PER_CLASS
        assert len(imgs) >= need, (
            f"{breed}: only {len(imgs)} images, need {need}. "
            "Lower the split counts or check the dataset."
        )
        shuffled = imgs[:]
        rng.shuffle(shuffled)

        train_n = TRAIN_MINORITY if breed == MINORITY else TRAIN_MAJORITY
        train   = shuffled[:train_n]
        val     = shuffled[train_n : train_n + VAL_PER_CLASS]
        test    = shuffled[train_n + VAL_PER_CLASS : train_n + VAL_PER_CLASS + TEST_PER_CLASS]

        for split, subset in [("train", train), ("val", val), ("test", test)]:
            for p in subset:
                rows.append({"image_id": p.name, "breed": breed, "split": split})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "breed", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest saved to {out_csv}  ({len(rows)} rows)")
    print_counts(rows)


def print_counts(rows: list[dict]) -> None:
    from collections import Counter
    counts: dict[tuple, int] = Counter((r["breed"], r["split"]) for r in rows)
    print(f"\n{'Breed':<20} {'train':>6} {'val':>6} {'test':>6}")
    print("-" * 42)
    for breed in BREEDS:
        tr = counts.get((breed, "train"), 0)
        va = counts.get((breed, "val"),   0)
        te = counts.get((breed, "test"),  0)
        flag = " <-- minority" if breed == MINORITY else ""
        print(f"{breed:<20} {tr:>6} {va:>6} {te:>6}{flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--out_csv",   type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()
    main(args.data_root, args.out_csv)
