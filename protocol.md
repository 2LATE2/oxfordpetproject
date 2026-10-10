# Protocol: Dataset Split and Real-Data Baselines

## Research Question
Can diffusion-generated images improve recognition of an underrepresented class beyond
oversampling and ordinary augmentation, without harming overall classification quality?

## Dataset
Oxford-IIIT Pet — 8 visually similar cat breeds:
Abyssinian, Bengal, Birman, Bombay, Egyptian Mau, Ragdoll, Russian Blue, Siamese.
**Egyptian Mau is the minority class.**

## Split Counts

| Role                         | Count     |
|------------------------------|-----------|
| Training — majority classes  | 70 per class |
| Training — Egyptian Mau      | 15        |
| Validation — each class      | 20 per class |
| Final test — each class      | 30 per class |

## Added-Data Budget
55 images (= 70 − 15). Same budget used for oversampling, augmentation, and later synthetic data.

## Metrics
- **Primary**: Egyptian Mau recall
- **Guardrail**: Macro-F1 across all 8 classes

## Three Baselines

| Condition       | What changes?                                        |
|-----------------|------------------------------------------------------|
| Imbalanced real | Nothing — this is the problem to improve upon        |
| Oversampling    | Repeat minority training images up to the budget     |
| Augmentation   | Add transformed minority images up to the budget     |

Everything else is fixed: split, classifier, training procedure, evaluation metrics.

## Random Seeds
- Data split seed: **42**
- Training seed: **0**

## Leakage Policy
No image — and no transformed copy of it — may appear in more than one split.
Enforced by `scripts/check_leakage.py`.
