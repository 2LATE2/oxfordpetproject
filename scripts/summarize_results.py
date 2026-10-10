"""Print a comparison table of all three baseline conditions.

Usage:
    py scripts/summarize_results.py
"""

import json
from pathlib import Path

CONDITIONS = ["imbalanced", "oversampling", "augmentation"]
RESULTS_DIR = Path("results")


def main():
    print(f"\n{'Condition':<15} {'Mau Recall':>12} {'Macro-F1':>10}")
    print("-" * 40)
    for cond in CONDITIONS:
        p = RESULTS_DIR / f"{cond}_metrics.json"
        if not p.exists():
            print(f"{cond:<15} {'N/A':>12} {'N/A':>10}")
            continue
        m = json.loads(p.read_text())
        print(f"{cond:<15} {m['egyptian_mau_recall']:>12.4f} {m['macro_f1']:>10.4f}")


if __name__ == "__main__":
    main()
