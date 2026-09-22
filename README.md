# SAS Viya for Learners Challenge 2026 / First Place

**Kaggle Competition · Workbench Track**

Predict hospital length of stay (`ADMIT_LOS`, in days) from clinical and administrative encounter records.

## Competition

- **Platform:** Kaggle / SAS Viya for Learners Challenge 2026
- **Task:** Regression — predict `ADMIT_LOS` (days, range 0–51)
- **Metric:** RMSE (lower is better)
- **Track:** Workbench (code submission)

## Results

| Stage | Model | CV RMSE | Kaggle LB |
|-------|-------|---------|----------|
| Baseline | Naive group median | 4.987 | — |
| Stage 3 | CatBoost full features | 1.990 | — |
| Stage 5 | CB + LGB + XGB + HGBM → Ridge stack, Optuna-tuned | 1.982 | 1.960 |
| Stage 7 | + Raw-float export + 3-seed bagging | 1.978 | 1.947 |
| Stage 10C | + OOF target encoding + 5-seed bagging + pseudo-labelling | 1.973 | 1.94486 |
| + Isotonic | Isotonic calibration on stack OOF | 1.9711 | 1.94478 |
| **Final** | **3 calibrated stacks (15 seeds) + embedding MLP blend** | **1.9694** | **1.94154** |

**Final leaderboard position: #1**

## Architecture

The winning submission blends four components.

**Three stacked ensembles** (`train_stack.py`), same architecture, disjoint seed sets:

- Stack A: [42, 7, 123, 999, 2025]
- Stack B: [100, 200, 300, 400, 500]
- Stack C: [555, 1717, 8080, 3141, 9999]

Each stack trains CatBoost (Optuna-tuned, depth 6), LightGBM (65 leaves), XGBoost (depth 7), and HistGradientBoosting across 5 folds on a `log1p` target. Target encoding for DOCTOR, DX_CODE, DIAGNOSIS_SUBCAT_CODE, and DEPARTMENT is computed fold-safely to prevent leakage. Pseudo-labelled test rows are appended to each fold's training data. OOF predictions from all four base models then feed a Ridge meta-learner.

**An embedding MLP** (`train_mlp.py`) trains on the same 5-fold split with learned categorical embeddings, standard-scaled numerics, and a 128→64→32 network with BatchNorm and dropout. Solo RMSE is 2.135 — weaker than the trees, but its errors correlate only ~0.925 with the stacks (vs. 0.9998 between stacks), so it contributes genuine diversity rather than redundancy.

**The blend** (`build_submission.py`): each stack gets isotonic calibration fitted on its own OOF, the three calibrated stacks are averaged, and the calibrated MLP is mixed in at 5% (tuned on cross-validated OOF only):

```
submission = 0.95 × mean(iso(stack_a), iso(stack_b), iso(stack_c)) + 0.05 × iso(mlp)
```

## Files

| File | What it does |
|------|--------------|
| `preprocess.py` | Shared data loading, cleaning, and feature engineering. Every training script imports from here. |
| `train_stack.py` | Trains one tree-model stack (4 base models × N seeds × 5 folds) with optional pseudo-labelling. Writes OOF and test predictions to `artifacts/`. |
| `train_mlp.py` | Trains the embedding MLP across the same 5-fold split. Writes OOF and test predictions to `artifacts/`. |
| `build_submission.py` | Loads all artifacts, applies isotonic calibration per stack, blends the four components, and writes `submission.csv`. |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, CatBoost and LightGBM need the ARM64 OpenMP library:

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

```bash
# Step 1 — bootstrap: train stack A without pseudo-labels
python train_stack.py --seeds 42 7 123 999 2025 --tag boot

# Step 2 — two rounds of self-training on stack A's test predictions
python train_stack.py --seeds 42 7 123 999 2025 --tag round1 --pseudo artifacts/preds_boot.csv
python train_stack.py --seeds 42 7 123 999 2025 --tag stack_a --pseudo artifacts/preds_round1.csv

# Step 3 — stacks B and C reuse the round-1 pseudo-labels
python train_stack.py --seeds 100 200 300 400 500 --tag stack_b --pseudo artifacts/preds_round1.csv
python train_stack.py --seeds 555 1717 8080 3141 9999 --tag stack_c --pseudo artifacts/preds_round1.csv

# Step 4 — embedding MLP
python train_mlp.py

# Step 5 — calibrate, blend, and write submission.csv
python build_submission.py
```

Each stack run takes roughly 45 minutes on an 8-core machine. The MLP takes about 6 minutes.

## What actually mattered

- **DEPARTMENT** was the single most important feature — surgical vs. medical departments have dramatically different LOS distributions.
- **Post-admission signals** (`ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`) carried most of the predictive weight. These are known at discharge, so they're fair game for the task.
- **Exporting raw floats** instead of rounded integers was worth ~0.020 RMSE. Always export floats for regression.
- **OOF target encoding** for DOCTOR, DX_CODE, DIAGNOSIS_SUBCAT_CODE, and DEPARTMENT gave the largest single categorical gain.
- **Pseudo-labelling saturates at round 2.** Two self-training rounds gave +0.005 LB; a third round did nothing — the model stops learning from its own confident predictions.
- **Isotonic calibration** fixed systematic over-prediction in the 47–51 day tail and under-prediction around 28–33 days, where the tree models were consistently off.
- **Diversity beats quantity.** Adding a fourth correlated tree draw was worth +0.0002 on LB; swapping it for one decorrelated MLP at 5% weight was worth roughly 10× that.
- Dropped columns: `DIAGNOSIS_ICD_CODE` (r=1.000 with `DIAGNOSIS_SUBCAT_CODE`), `DISCH_NURSE_ID` (signal collapsed in the updated dataset), and `HOSPITAL` as a direct feature (near-zero LOS signal across 39 hospitals — kept only as frequency encoding).
