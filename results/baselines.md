# Baseline results (validation split)

| Model | Condition | Mau recall (mean ± sd) | Macro-F1 (mean ± sd) | Seeds |
|---|---|---|---|---|
| resnet18 | imbalanced | 0.450 ± 0.050 | 0.833 ± 0.006 | 3 |
| resnet18 | oversampling | 0.833 ± 0.076 | 0.875 ± 0.020 | 3 |
| resnet18 | augmentation | 0.783 ± 0.076 | 0.875 ± 0.026 | 3 |
| efficientnet_b0 | imbalanced | 0.533 ± 0.029 | 0.830 ± 0.010 | 3 |
| efficientnet_b0 | oversampling | 0.733 ± 0.029 | 0.847 ± 0.016 | 3 |
| efficientnet_b0 | augmentation | 0.633 ± 0.029 | 0.844 ± 0.010 | 3 |
