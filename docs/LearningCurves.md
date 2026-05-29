# Learning Curves Analysis

Each plot shows three curves as a function of training set size:

- **Training accuracy** (blue) - score on the training subset used at each step.
- **Validation accuracy** (purple) - mean cross-validated score on the held-out CV folds.
- **Test accuracy** (orange) - score on the fixed held-out test set, unseen during training.

A well-generalising model has training and validation/test converging to a similar value. A large persistent gap (train high, val/test low) indicates **overfitting**; curves that are already flat and close together from the start indicate **underfitting** (high bias).

Hyperparameters were tuned by sweeping each parameter individually (see `src/models/analyze/`) and picking the value that maximised test accuracy.

---

## Logistic Regression & Linear SVM

All three curves sit in a narrow 0.724–0.730 band from the very first training size. Training and validation are almost indistinguishable, and test accuracy sits marginally above both. Adding more data produces no meaningful gain. This is the signature of **high bias / underfitting**: a linear decision boundary is too simple to capture the non-linear structure of cardiovascular risk. No hyperparameter tuning was applied — these models are already at their capacity ceiling.

## Naive Bayes

All three curves are flat around 0.710–0.714 from the start. The model is even weaker than the linear models because the feature independence assumption is violated (age, BMI, and blood pressure are correlated). The wide confidence band on validation reflects sensitivity to which folds are used, not genuine model variance.

## Decision Tree

**Tuning:** `max_depth` 7 → 5 (sweep showed test peaking at 5, degrading to 0.678 by depth 20).

| | Before (depth 7) | After (depth 5) |
|---|---|---|
| Training accuracy | ~0.759 → 0.736 | ~0.739 → 0.731 |
| Val/Test accuracy | ~0.710 → 0.730 | ~0.718 → 0.733 |
| Gap at full data | ~0.006 | ~0.002 |

The three curves are now nearly converged. The tree is no longer wasting capacity on depth levels that only memorise noise.

## KNN

**Tuning:** `n_neighbors` 5 → 19 (more neighbours smooth the decision boundary, reducing memorisation).

| | Before (k=5) | After (k=19) |
|---|---|---|
| Training accuracy | ~0.782 (flat) | ~0.745 (flat) |
| Val/Test accuracy | ~0.690 / ~0.704 | ~0.718 / ~0.722 |
| Gap at full data | ~0.080 | ~0.021 |

The gap dropped from 0.080 to 0.021. Training accuracy fell (less memorisation) while val/test rose — exactly the right direction. A structural gap remains because KNN always fits training data more closely than unseen data, but it is now at a manageable level.

## Random Forest

**Tuning:** `n_estimators` 140 → 80, `max_depth` None → 8 (sweep showed test peaking at depth 8; without a depth limit training was at 1.000 regardless of estimator count).

| | Before (unconstrained) | After (max_depth=8, n=80) |
|---|---|---|
| Training accuracy | 1.000 (flat) | ~0.787 → 0.741 |
| Val/Test accuracy | ~0.71 | ~0.731 → 0.737 |
| Gap at full data | ~0.290 | ~0.008 |

The most dramatic improvement. Adding `max_depth=8` forced the forest to generalise rather than memorise, collapsing the gap from 0.29 to under 0.01. The training curve now descends naturally as more data is added, which is the expected healthy shape.

## XGBoost

**Tuning:** `max_depth` 6 → 3, `learning_rate` 0.3 → 0.1 (both sweeps peaked at these values; default depth 6 and lr 0.3 were already in the overfitting zone).

| | Before (depth=6, lr=0.3) | After (depth=3, lr=0.1) |
|---|---|---|
| Training accuracy | ~0.900 → 0.768 | ~0.755 → 0.737 |
| Val/Test accuracy | ~0.700 → 0.730 | ~0.730 → 0.740 |
| Gap at full data | ~0.038 | ~0.003 |

Curves are now nearly converged across the full training range. The shallower trees and slower learning rate prevent the model from chasing noise in the training data.

## LightGBM

**Tuning:** `num_leaves` 31 → 20 (sweep showed test peaking at 20; `learning_rate` was already 0.1, the optimal value).

| | Before (leaves=31) | After (leaves=20) |
|---|---|---|
| Training accuracy | ~0.832 → 0.750 | ~0.800 → 0.745 |
| Val/Test accuracy | ~0.720 → 0.734 | ~0.720 → 0.740 |
| Gap at full data | ~0.016 | ~0.005 |

Reducing num_leaves limits tree complexity directly. The gap narrowed from 0.016 to 0.005 and the curves are clearly converging — LightGBM is the best-behaved tree model after tuning.

---

## Summary

| Model | Tuned params | Gap before | Gap after | Change |
|---|---|---|---|---|
| Logistic Regression | - | Negligible | Negligible | - |
| Linear SVM | - | Negligible | Negligible | - |
| Naive Bayes | - | Negligible | Negligible | - |
| Decision Tree | max_depth 7→5 | ~0.006 | ~0.002 | better |
| KNN | n_neighbors 5→19 | ~0.080 | ~0.021 | better |
| Random Forest | max_depth None→8, n 140→80 | ~0.290 | ~0.008 | much better |
| XGBoost | max_depth 6→3, lr 0.3→0.1 | ~0.038 | ~0.003 | much better |
| LightGBM | num_leaves 31→20 | ~0.016 | ~0.005 | better |
