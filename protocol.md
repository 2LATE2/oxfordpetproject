# Protocol: Dataset Split and Real-Data Baselines

## Research Question
Can diffusion-generated images improve recognition of an underrepresented class beyond
oversampling and ordinary augmentation, without harming overall classification quality?

## Dataset
Oxford-IIIT Pet (downloaded via `torchvision.datasets.OxfordIIITPet`) — 8 visually similar cat breeds:
Abyssinian, Bengal, Birman, Bombay, Egyptian Mau, Ragdoll, Russian Blue, Siamese.
**Egyptian Mau is the minority class.** Each breed has 200 images in the raw dataset.

### Cleaning before the split
- Unreadable files are dropped (none found).
- Near-duplicates are collapsed to one image *before* splitting: two images are copies if the
  Hamming distance between their 64-bit perceptual hashes (pHash, original or horizontally
  mirrored) is ≤ 6. 41 images were dropped (most in Bombay — 24 — where dark, low-texture
  photos also produce some false matches; dropping them is conservative and harmless).

## Split (seed 42, `scripts/make_split.py`, done once)

| Role                         | Count        |
|------------------------------|--------------|
| Training — majority classes  | 70 per class |
| Training — Egyptian Mau      | 15           |
| Validation — each class      | 20 per class |
| Final test — each class      | 30 per class |

Manifest: `data/manifest.csv` — all 1600 images of the 8 breeds with their role
(`train` / `val` / `test` = 905 used images; `unused`, `dropped_duplicate`, `dropped_unreadable`).
Class-count table: `data/class_counts.csv`.
Imbalance exists **only** in training; validation and test are balanced.

**The test split is locked** until Meeting 5. All model, prompt, subset and training-setting
decisions are made on validation only.

## Added-Data Budget
**55 observations** (= 70 − 15), identical for every condition that adds data:
oversampling, augmentation, and the later synthetic-data conditions. The 55 extra rows cycle
through the 15 Mau images as evenly as possible (3 full passes + 10 more).

## Baseline conditions

| Condition       | What changes?                                                       |
|-----------------|---------------------------------------------------------------------|
| Imbalanced real | Nothing — this is the problem to improve upon                       |
| Oversampling    | +55 repeated Mau rows (standard train transform)                    |
| Augmentation    | +55 Mau rows with stronger transforms (crop 0.6–1, flip, colour jitter, ±15° rotation) |

Everything else is fixed: split, representation, classifier, training procedure, metrics.

## Classifier and training (fixed for all conditions)
- Backbones: **ResNet-18** (`timm/resnet18.tv_in1k`) and **EfficientNet-B0**
  (`timm/efficientnet_b0.ra_in1k`), ImageNet-pretrained, downloaded from the Hugging Face Hub.
  EfficientNet-B0 checks whether conclusions transfer across architectures.
- Backbone frozen and kept in eval mode (BatchNorm statistics fixed); only a linear head is trained.
- Adam, lr 1e-3, batch 32, cross-entropy, **10 epochs, final epoch reported** — no epoch
  selection on validation, so the validation metrics are not optimistically biased.
- Train transform: RandomResizedCrop 224 + horizontal flip. Eval: resize 256, center crop 224.
- Hardware: NVIDIA RTX 4060 Laptop GPU (CUDA).

## Metrics
- **Primary**: Egyptian Mau recall
- **Guardrail**: Macro-F1 across all 8 classes

With 20 Mau validation images, recall moves in steps of 0.05, so each condition is reported as
mean ± sd over training seeds.

## Random Seeds
- Data split seed: **42**
- Training seeds: **0, 1, 2** (head initialisation, batch order, augmentation, extra-row order)

## Leakage Policy
No image — and no transformed copy of it — may appear in more than one split.
Enforced by `scripts/check_leakage.py`, which fails if:
1. any `image_id` is in more than one split;
2. any cross-split pair is a near-duplicate (pHash distance ≤ 6, incl. mirrored copies);
3. validation or test is not class-balanced.

Augmented/oversampled/synthetic rows are created on the fly from training images only.

## Reproduce
```
py scripts/make_split.py
py scripts/check_leakage.py
py scripts/train_baseline.py --model resnet18 --condition imbalanced    # etc. for each condition/model
py scripts/summarize_results.py   # table  -> results/baselines.md
py scripts/plot_results.py        # figure -> results/baselines.png (mean ± SD, per-seed dots)
```

## Baseline results (validation)
See `results/baselines.md` and `results/baselines.png`.
