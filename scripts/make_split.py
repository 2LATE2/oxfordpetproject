"""
Split Oxford-IIIT Pet images into train/val/test and save a manifest CSV.
Run once. Never re-run after seeing test results.

Before splitting, unreadable files are dropped and near-duplicate images
(perceptual hash distance <= HASH_THRESHOLD, which also catches flipped copies)
are collapsed to one representative, so no image or its copy can end up in two splits.

Usage:
    py scripts/make_split.py --data_root data/oxford-iiit-pet
"""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from hashing import HASH_THRESHOLD, image_hashes

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


def is_readable(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def dedupe(paths: list[Path], hashes: dict[str, list]) -> tuple[list[Path], list[tuple[str, str]]]:
    """Keep the first image of every near-duplicate group (input is sorted, so deterministic)."""
    kept: list[Path] = []
    dropped: list[tuple[str, str]] = []
    for p in paths:
        hp = hashes[p.name]
        match = next((k for k in kept
                      if min(a - b for a in hp for b in hashes[k.name]) <= HASH_THRESHOLD), None)
        if match is None:
            kept.append(p)
        else:
            dropped.append((p.name, match.name))
    return kept, dropped


def main(data_root: Path, out_csv: Path, counts_csv: Path) -> None:
    images_dir = data_root / "images"
    assert images_dir.exists(), f"Images folder not found: {images_dir}"

    rng = random.Random(SEED)
    rows: list[dict] = []

    # Hash every candidate image of every breed so duplicates across breeds are also caught.
    candidates = {b: [p for p in collect_images(images_dir, b) if is_readable(p)] for b in BREEDS}
    all_paths = sorted((p for ps in candidates.values() for p in ps), key=lambda p: p.name)
    hashes = {p.name: image_hashes(p) for p in all_paths}
    kept_all, dropped = dedupe(all_paths, hashes)
    kept_names = {p.name for p in kept_all}
    for name, match in dropped:
        print(f"Near-duplicate dropped: {name} (matches {match})")

    dropped_names = {name for name, _ in dropped}

    for breed in BREEDS:
        all_files = collect_images(images_dir, breed)
        n_files = len(all_files)
        readable = {p.name for p in candidates[breed]}
        imgs = [p for p in candidates[breed] if p.name in kept_names]
        print(f"{breed}: {n_files} files, {len(candidates[breed])} readable, {len(imgs)} after dedup")
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

        # Record every other image of the breed too, so the manifest covers all files.
        assigned = {r["image_id"] for r in rows}
        for p in all_files:
            if p.name in assigned:
                continue
            if p.name not in readable:
                role = "dropped_unreadable"
            elif p.name in dropped_names:
                role = "dropped_duplicate"
            else:
                role = "unused"
            rows.append({"image_id": p.name, "breed": breed, "split": role})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "breed", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest saved to {out_csv}  ({len(rows)} rows)")
    write_counts(rows, counts_csv)


def write_counts(rows: list[dict], counts_csv: Path) -> None:
    counts: dict[tuple, int] = Counter((r["breed"], r["split"]) for r in rows)
    print(f"\n{'Breed':<20} {'train':>6} {'val':>6} {'test':>6} {'unused':>7} {'dropped':>8}")
    print("-" * 60)
    table = []
    for breed in BREEDS:
        tr = counts.get((breed, "train"), 0)
        va = counts.get((breed, "val"),   0)
        te = counts.get((breed, "test"),  0)
        un = counts.get((breed, "unused"), 0)
        dr = counts.get((breed, "dropped_duplicate"), 0) + counts.get((breed, "dropped_unreadable"), 0)
        flag = " <-- minority" if breed == MINORITY else ""
        print(f"{breed:<20} {tr:>6} {va:>6} {te:>6} {un:>7} {dr:>8}{flag}")
        table.append({"breed": breed, "train": tr, "val": va, "test": te, "unused": un, "dropped": dr})

    counts_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(counts_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["breed", "train", "val", "test", "unused", "dropped"])
        writer.writeheader()
        writer.writerows(table)
    print(f"Class counts saved to {counts_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=Path, default=Path("data/oxford-iiit-pet"))
    parser.add_argument("--out_csv",    type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--counts_csv", type=Path, default=Path("data/class_counts.csv"))
    args = parser.parse_args()
    main(args.data_root, args.out_csv, args.counts_csv)
