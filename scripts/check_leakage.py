"""
Automated leakage check:
  1. no image_id appears in more than one split (exact overlap);
  2. no image in one split is a near-duplicate / transformed copy (resize, re-encode,
     horizontal flip) of an image in another split, using perceptual hashing;
  3. val and test are class-balanced.

Exits with code 1 on any failure.

Usage:
    py scripts/check_leakage.py --manifest data/manifest.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from hashing import HASH_THRESHOLD, image_hashes


SPLITS = ("train", "val", "test")
OTHER_ROLES = ("unused", "dropped_duplicate", "dropped_unreadable")


def main(manifest: Path, images_dir: Path) -> None:
    full = pd.read_csv(manifest)
    assert {"image_id", "breed", "split"}.issubset(full.columns), "Unexpected manifest columns"
    failed = False

    bad_roles = set(full["split"]) - set(SPLITS) - set(OTHER_ROLES)
    if bad_roles:
        failed = True
        print(f"Unknown roles in manifest: {bad_roles}")
    if full["image_id"].nunique() != len(full):
        failed = True
        print("Some image_id is listed more than once in the manifest")

    # Unused / dropped images never reach any experiment; check only the experimental splits.
    df = full[full["split"].isin(SPLITS)]

    # 1. Exact overlap: each image_id must appear in exactly one split
    split_map: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        split_map[row["image_id"]].add(row["split"])

    leaks = {img: splits for img, splits in split_map.items() if len(splits) > 1}
    if leaks:
        failed = True
        print(f"LEAKAGE DETECTED — {len(leaks)} image(s) appear in multiple splits:")
        for img, splits in list(leaks.items())[:20]:
            print(f"  {img}: {splits}")

    total = len(df)
    unique_ids = df["image_id"].nunique()
    if total != unique_ids:
        failed = True
        print(f"Duplicate rows detected ({total} rows, {unique_ids} unique ids)")

    # 2. Near-duplicate overlap across splits
    hashes = {img: image_hashes(images_dir / img) for img in df["image_id"]}
    rows = list(df[["image_id", "split"]].itertuples(index=False))
    near = []
    for i, (a, sa) in enumerate(rows):
        for b, sb in rows[i + 1:]:
            if sa == sb:
                continue
            d = min(x - y for x in hashes[a] for y in hashes[b])
            if d <= HASH_THRESHOLD:
                near.append((a, sa, b, sb, d))
    if near:
        failed = True
        print(f"NEAR-DUPLICATE LEAKAGE — {len(near)} cross-split pair(s):")
        for a, sa, b, sb, d in near[:20]:
            print(f"  {a} ({sa}) ~ {b} ({sb})  distance={d}")

    # 3. Val / test must stay balanced
    counts = df.groupby(["breed", "split"]).size().unstack(fill_value=0)
    for split in ("val", "test"):
        if split in counts and counts[split].nunique() != 1:
            failed = True
            print(f"Split '{split}' is not class-balanced: {counts[split].to_dict()}")

    print("\nClass counts per role:")
    print(full.groupby(["breed", "split"]).size().unstack(fill_value=0).to_string())

    if failed:
        sys.exit(1)
    print(f"\nLeakage check PASSED — {unique_ids} unique images, no exact or near-duplicate "
          f"overlap between splits (pHash threshold {HASH_THRESHOLD}, flips included).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",   type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--images_dir", type=Path, default=Path("data/oxford-iiit-pet/images"))
    args = parser.parse_args()
    main(args.manifest, args.images_dir)
