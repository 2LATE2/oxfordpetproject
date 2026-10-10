"""
Automated leakage check: verify no image_id appears in more than one split.

Usage:
    py scripts/check_leakage.py --manifest data/manifest.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main(manifest: Path) -> None:
    df = pd.read_csv(manifest)
    assert {"image_id", "breed", "split"}.issubset(df.columns), "Unexpected manifest columns"

    # Each image_id must appear in exactly one split
    split_map: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        split_map[row["image_id"]].add(row["split"])

    leaks = {img: splits for img, splits in split_map.items() if len(splits) > 1}

    if leaks:
        print(f"LEAKAGE DETECTED — {len(leaks)} image(s) appear in multiple splits:")
        for img, splits in list(leaks.items())[:20]:
            print(f"  {img}: {splits}")
        sys.exit(1)

    total = len(df)
    unique_ids = df["image_id"].nunique()
    assert total == unique_ids, f"Duplicate rows detected ({total} rows, {unique_ids} unique ids)"

    print(f"Leakage check PASSED — {unique_ids} unique images, no overlap between splits.")

    counts = df.groupby(["breed", "split"]).size().unstack(fill_value=0)
    print("\nClass counts per split:")
    print(counts.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()
    main(args.manifest)
