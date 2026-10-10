"""Plot baseline results: mean ± sd over training seeds, with individual seeds as dots.

Reads results/<model>_<condition>_metrics.json and writes results/baselines.png.

Usage:
    py scripts/plot_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {"resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0"}
CONDITIONS = {"imbalanced": "Imbalanced", "oversampling": "Oversampling", "augmentation": "Augmentation"}
METRICS = {"egyptian_mau_recall": "Egyptian Mau recall (primary)",
           "macro_f1": "Macro-F1, 8 classes (guardrail)"}
COLORS = {"resnet18": "#2a78d6", "efficientnet_b0": "#eb6834"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RESULTS_DIR = Path("results")


def load():
    out = {}
    for model in MODELS:
        for cond in CONDITIONS:
            p = RESULTS_DIR / f"{model}_{cond}_metrics.json"
            if p.exists():
                out[(model, cond)] = json.loads(p.read_text())
    return out


def main():
    res = load()
    models = [m for m in MODELS if any((m, c) in res for c in CONDITIONS)]
    assert models, "No metrics files found in results/"
    n_seeds = max(len(r["seeds"]) for r in res.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    x = np.arange(len(CONDITIONS))
    width = 0.8 / len(models)

    for ax, (metric, title) in zip(axes, METRICS.items()):
        ax.set_facecolor(SURFACE)
        for i, model in enumerate(models):
            xs = x - 0.4 + width * (i + 0.5)
            for xi, cond in zip(xs, CONDITIONS):
                r = res.get((model, cond))
                if r is None:
                    continue
                vals = np.array([s[metric] for s in r["per_seed"]])
                mean = vals.mean()
                sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
                ax.bar(xi, mean, width * 0.9, color=COLORS[model], alpha=0.85, zorder=2,
                       label=MODELS[model] if cond == "imbalanced" else None)
                ax.errorbar(xi, mean, yerr=sd, color=INK, capsize=4, lw=1.2, zorder=3)
                jitter = np.linspace(-width * 0.18, width * 0.18, len(vals))
                ax.scatter(xi + jitter, vals, s=18, color=SURFACE, edgecolor=INK,
                           linewidth=0.9, zorder=4)
                ax.text(xi, min(mean + sd, 1.0) + 0.02, f"{mean:.2f}±{sd:.2f}",
                        ha="center", va="bottom", fontsize=8, color=INK_2)

        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xticks(x, list(CONDITIONS.values()), color=INK)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.tick_params(colors=INK_2, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle(f"Real-data baselines on validation split — bar = mean, whisker = ±1 SD, "
                 f"dots = {n_seeds} training seeds", fontsize=10, color=INK_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS_DIR / "baselines.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
