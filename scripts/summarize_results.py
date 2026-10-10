"""Print (and save as Markdown) a comparison table of all baseline conditions.

Usage:
    py scripts/summarize_results.py
"""

import json
from pathlib import Path

MODELS = ["resnet18", "efficientnet_b0"]
CONDITIONS = ["imbalanced", "oversampling", "augmentation"]
RESULTS_DIR = Path("results")


def main():
    lines = [
        "| Model | Condition | Mau recall (mean ± sd) | Macro-F1 (mean ± sd) | Seeds |",
        "|---|---|---|---|---|",
    ]
    for model in MODELS:
        for cond in CONDITIONS:
            p = RESULTS_DIR / f"{model}_{cond}_metrics.json"
            if not p.exists():
                lines.append(f"| {model} | {cond} | N/A | N/A | – |")
                continue
            m = json.loads(p.read_text())
            lines.append(
                f"| {model} | {cond} "
                f"| {m['egyptian_mau_recall_mean']:.3f} ± {m['egyptian_mau_recall_std']:.3f} "
                f"| {m['macro_f1_mean']:.3f} ± {m['macro_f1_std']:.3f} "
                f"| {len(m['seeds'])} |"
            )
    table = "\n".join(lines)
    print(table)
    (RESULTS_DIR / "baselines.md").write_text(
        "# Baseline results (validation split)\n\n" + table + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_DIR / 'baselines.md'}")


if __name__ == "__main__":
    main()
