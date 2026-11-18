# Equal Accuracy, Different Mistakes

Per-image predictions and analysis code for the paper *Equal Accuracy,
Different Mistakes: Classifier Heads and Error Diversity in Fine-Grained Waste
Classification*.

The point of releasing this is that none of the results need the grid to be
retrained. Every number, table and figure in the paper is recomputed from the
stored predictions in a few minutes on a CPU.

## What is here

```
analysis/        scripts that produce every number in the paper
predictions/     per-image predictions, 18 configurations x 3 seeds x 2 splits
results/         the outputs those scripts produce
figures/         the figures as they appear in the paper
paper/           LaTeX source
```

`predictions/` holds 108 `.npz` files. Each contains `y_pred` (N,), `y_prob`
(N, 36) and `y_true` (N,) for one configuration, in a fixed image order shared
by every file. The test split has 3,552 images and the validation split 3,511.

Configurations are `<backbone>_<head>` over six backbones (EfficientNetV2B0,
MobileNetV3Small, ResNet50, DenseNet201, InceptionV3, Xception) and three
classifier heads (`SIMPLE_FC`, `GAP_BN_FC_Drop`, `DUAL_POOLING_FC`).

## Reproducing the paper

```bash
pip install numpy pandas scipy scikit-learn matplotlib
cd analysis

# main analysis: members selected on validation, scored on test
python analysis_03_honest.py --seeds seed36 seed42 seed123

# learned (NLL-fitted) member weights vs. equal weights
python analysis_04_weighted.py --seeds seed36 seed42 seed123

# stability of each selection rule over 500 validation resamples
python analysis_05_bootstrap.py --seeds seed36 seed42 seed123

# tables
python paper_tables.py

# figures
python make_figures.py
```

`analysis_02_diversity.py` is the earlier version that measured diversity on
the test split. It is kept because the difference between the two is a result
in the paper, not because it should be used.

## Two things worth knowing before reusing this

**Image order.** Every pairwise comparison here assumes all files share one
image ordering. `check_consistency.py` verifies this by hashing `y_true` across
files; if it reports more than one group, the comparisons are meaningless.
Models were evaluated with `shuffle=False`.

**Selection protocol.** Estimating diversity on the test split and then
selecting ensemble members on it lets the selection rule fit the test set. In
our grid that was worth about two points of apparent gain. Anything derived
from `analysis_02_diversity.py` carries that bias; `analysis_03_honest.py` does
not.

## How the models were trained

Three Kaggle notebooks, chained only because a single run exceeds Kaggle's
12-hour limit. Their code cells are byte-identical apart from which
configurations each one skips, and all three used the same container image and
the same dataset, so the two seeds differ only in the seed.

Seed 123 is a held-out check on the ensemble-selection results and is not used
for the per-configuration tables or figures. Ten of its configurations come from
the same P100 runs. The other eight (all ResNet50 and DenseNet201 models, and
Xception with `GAP_BN_FC_Drop` and `DUAL_POOLING_FC`) were trained in November
2026 on an NVIDIA Tesla T4, because the P100 was no longer offered, with the
same code, data and seed. Retraining `xception_SIMPLE_FC` on the T4 as a control
changed 237 of 3,552 test predictions, comparable to a change of seed (217-306);
the released file for that configuration is the P100 one.

Training used TensorFlow 2.19 on one NVIDIA Tesla P100 (seeds 36 and 42): three warm-up epochs
with the backbone frozen at lr 1e-3, then up to 30 fine-tuning epochs at lr
1e-5, with focal loss (gamma 3.0), balanced class weights, `ReduceLROnPlateau`
(factor 0.3, patience 3) and early stopping on validation accuracy (patience 5).
Augmentation is applied online inside the model.

Data is the 36-class Stage 3 subset of the TriCascade WasteImage dataset,
split 80/10/10 into 28,201 / 3,511 / 3,552 images.

## Citation

Le, M.H., Tran, V.: *Equal Accuracy, Different Mistakes: Classifier Heads and
Error Diversity in Fine-Grained Waste Classification*. Under review.
