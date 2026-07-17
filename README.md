# SAS Viya for Learners Challenge 2026

**Kaggle Competition — Workbench Track Submission**

Predict hospital length of stay (`ADMIT_LOS`, in days) from clinical and administrative encounter records.

## Competition

- **Platform:** Kaggle / SAS Viya for Learners Challenge 2026
- **Task:** Regression — predict `ADMIT_LOS` (days, range 0–51)
- **Metric:** RMSE (lower is better)
- **Track:** Workbench (code submission)

## Results

| Stage | Model | CV RMSE | Kaggle LB |
|-------|-------|---------|-----------|
| Baseline | Naive group median | 4.987 | — |
| Stage 3 | CatBoost full features | 1.990 | — |
| Stage 5 | CB + LGB + XGB + HGBM → Ridge stack, Optuna-tuned | 1.982 | 1.960 |
| Stage 7 | + Raw-float export + 3-seed bagging | 1.978 | 1.947 |
| Stage 10C | + OOF target encoding + 5-seed bagging + pseudo-labelling | 1.973 | 1.94486 |
| + Isotonic | Isotonic calibration on stack OOF | 1.9711 | 1.94478 |
| **Final** | **3 calibrated stacks (15 seeds) + embedding MLP blend** | **1.9694** | **1.94154** |

**Final leaderboard position: #1**

## Final model architecture

The leaderboard submission is a blend of four components:

**Three independent stacked ensembles** (`train_stack.py`), identical
architecture, disjoint seed sets:

- Stack A: seeds [42, 7, 123, 999, 2025]
- Stack B: seeds [100, 200, 300, 400, 500]
- Stack C: seeds [555, 1717, 8080, 3141, 9999]

Each stack: per seed, 5-fold CV training of **CatBoost** (Optuna-tuned,
depth 6), **LightGBM** (65 leaves), **XGBoost** (depth 7) and
**HistGradientBoosting** — all on a `log1p` target with fold-safe smoothed
target encoding (DOCTOR, DX_CODE, DIAGNOSIS_SUBCAT_CODE, DEPARTMENT) and
pseudo-labelled test rows appended to each fold's training data. Seed-averaged
OOF predictions feed a **Ridge meta-learner**.

**One embedding MLP** (`train_mlp.py`): learned embeddings for all categorical
columns, standard-scaled numerics, 128→64→32 dense layers with BatchNorm and
dropout, trained on the same folds. Solo RMSE 2.135 — weaker than the trees,
but its errors correlate only ~0.925 with the stacks (the stacks correlate
0.9998 with each other), so it contributes genuine diversity.

**Final blend** (`build_submission.py`): each stack is isotonic-calibrated on
its own OOF, the three calibrated stacks are averaged, and the calibrated MLP
is mixed in at weight 0.05 (tuned on cross-validated OOF only):

```
submission = 0.95 * mean(iso(stack_a), iso(stack_b), iso(stack_c)) + 0.05 * iso(mlp)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon Macs, CatBoost/LightGBM need the ARM64 OpenMP library:

```bash
cp $(brew --prefix libomp)/lib/libomp.dylib venv/lib/libomp.dylib
export DYLD_LIBRARY_PATH=venv/lib:$DYLD_LIBRARY_PATH
```

## Data

Download the competition data from Kaggle and place it under `data/`:

```
data/
├── train.csv
└── test.csv
```

## Running the pipeline

The full pipeline, in order:

```bash
# 1. Bootstrap: train stack A without pseudo-labels, predict test
python train_stack.py --seeds 42 7 123 999 2025 --tag boot

# 2. Self-training: two rounds of pseudo-labelling with stack A
python train_stack.py --seeds 42 7 123 999 2025 --tag round1 --pseudo artifacts/preds_boot.csv
python train_stack.py --seeds 42 7 123 999 2025 --tag stack_a --pseudo artifacts/preds_round1.csv

# 3. Stacks B and C reuse the round-1 pseudo-labels
python train_stack.py --seeds 100 200 300 400 500 --tag stack_b --pseudo artifacts/preds_round1.csv
python train_stack.py --seeds 555 1717 8080 3141 9999 --tag stack_c --pseudo artifacts/preds_round1.csv

# 4. Embedding MLP
python train_mlp.py

# 5. Calibrate, blend, write submission.csv
python build_submission.py
```

Each stack run takes roughly 45 minutes on an 8-core machine; the MLP takes
about 6 minutes.

## Key findings

- **DEPARTMENT** is the #1 feature by importance — surgical vs medical
  departments have dramatically different LOS profiles
- **Post-admission features** (`ICU_DAYS`, `ORDER_TOTAL_CHARGES`,
  `DISCHARGED_TO`) drive the majority of signal
- **Integer rounding costs ~0.020 RMSE** — always export raw floats
- **OOF target encoding** for DOCTOR/DX_CODE/DIAGNOSIS_SUBCAT_CODE/DEPARTMENT
  provided the largest categorical signal gain
- **Pseudo-labelling** (self-training on test predictions) gave +0.005 LB across
  two rounds; a third round showed zero gain — the technique saturates fast
- **Isotonic calibration** on stack OOF corrects systematic over-prediction in
  the 47–51 day tail and under-prediction around 28–33 days
- **Model diversity beats model count**: adding a 4th correlated tree draw was
  worth +0.0002; adding one decorrelated MLP at 5% weight was worth ~10x that
  on the leaderboard
- `DIAGNOSIS_ICD_CODE` dropped (r=1.000 with `DIAGNOSIS_SUBCAT_CODE`);
  `DISCH_NURSE_ID` dropped (signal collapsed in the updated dataset);
  `HOSPITAL` frequency-encoded (near-zero direct LOS signal across 39 hospitals)
