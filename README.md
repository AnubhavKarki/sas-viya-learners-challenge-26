# SAS Viya for Learners Challenge 2026 - First Place

**Kaggle Competition · Workbench Track**

Predict hospital length of stay (`ADMIT_LOS`, in days) from clinical and administrative encounter records.

## Competition

- **Platform:** Kaggle / SAS Viya for Learners Challenge 2026
- **Task:** Regression - predict `ADMIT_LOS` (days, range 0-51)
- **Metric:** RMSE (lower is better)
- **Track:** Workbench (code submission)

## Results

| Stage | Model | CV RMSE | Kaggle LB |
|-------|-------|---------|----------|
| Baseline | Naive group median | 4.987 | - |
| Stage 3 | CatBoost full features | 1.990 | - |
| Stage 5 | CB + LGB + XGB + HGBM → Ridge stack, Optuna-tuned | 1.982 | 1.960 |
| Stage 7 | + Raw-float export + 3-seed bagging | 1.978 | 1.947 |
| Stage 10C | + OOF target encoding + 5-seed bagging + pseudo-labelling | 1.973 | 1.94486 |
| + Isotonic | Isotonic calibration on stack OOF | 1.9711 | 1.94478 |
| **Final** | **3 calibrated stacks (15 seeds) + embedding MLP blend** | **1.9694** | **1.94154** |

**Final leaderboard position: #1**

## Architecture

The final submission blends four components: three independent tree stacks and one embedding MLP.

---

### 1. Tree Stacks (`train_stack.py`)

Three stacks trained with the same architecture but disjoint seed sets to maximise ensemble diversity:

| Stack | Seeds |
|-------|-------|
| A | 42, 7, 123, 999, 2025 |
| B | 100, 200, 300, 400, 500 |
| C | 555, 1717, 8080, 3141, 9999 |

Each stack runs 5-fold CV per seed with four base models:

| Model | Key settings |
|-------|--------------|
| CatBoost | Optuna-tuned, depth 6, 5000 iterations |
| LightGBM | 65 leaves, learning rate 0.037 |
| XGBoost | depth 7, learning rate 0.03, hist method |
| HistGradientBoosting | 31 leaf nodes, learning rate 0.010 |

**Training assumptions and design choices:**

- Target is `log1p`-transformed before training and back-transformed after; this compresses the long tail and stabilises tree splits.
- Target encoding for `DOCTOR`, `DX_CODE`, `DIAGNOSIS_SUBCAT_CODE`, and `DEPARTMENT` is computed inside each fold to prevent any leakage from validation rows.
- Pseudo-labelled test rows (from the previous bootstrap run) are appended to each fold's training data during self-training rounds.
- OOF predictions from all four base models are stacked as features for a Ridge meta-learner, which outputs the final stack prediction.
- Seed-averaged OOF and test predictions are used before stacking to reduce variance across the 5 seeds.

---

### 2. Embedding MLP (`train_mlp.py`)

A neural network trained on the same 5-fold split as the tree stacks.

**Architecture:**
```
Categorical columns  ->  Learned embeddings (dim = min(50, (n+1)//2))
Numeric columns      ->  StandardScaler per fold
                              |
                        Concatenate
                              |
                     Linear(128) -> BatchNorm -> ReLU -> Dropout(0.25)
                     Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.25)
                     Linear(32)  -> BatchNorm -> ReLU
                     Linear(1)
```

**Why include it if the RMSE is worse (2.135 vs. ~1.97)?**
The MLP's prediction errors correlate only ~0.925 with the tree stacks, whereas the three stacks correlate 0.9998 with each other. That low correlation means the MLP is making different mistakes, which is exactly what you want from a blend component.

---

### 3. Final Blend (`build_submission.py`)

Each stack is isotonic-calibrated on its own OOF before blending. Isotonic regression corrects the systematic shape errors the Ridge meta-learner can't fix (over-prediction in the 47-51 day tail, under-prediction around 28-33 days).

The MLP blend weight (5%) was tuned on cross-validated OOF only, never on leaderboard feedback.

```
submission = 0.95 * mean(iso(stack_a), iso(stack_b), iso(stack_c)) + 0.05 * iso(mlp)
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
# Step 1 - bootstrap: train stack A without pseudo-labels
python train_stack.py --seeds 42 7 123 999 2025 --tag boot

# Step 2 - two rounds of self-training on stack A's test predictions
python train_stack.py --seeds 42 7 123 999 2025 --tag round1 --pseudo artifacts/preds_boot.csv
python train_stack.py --seeds 42 7 123 999 2025 --tag stack_a --pseudo artifacts/preds_round1.csv

# Step 3 - stacks B and C reuse the round-1 pseudo-labels
python train_stack.py --seeds 100 200 300 400 500 --tag stack_b --pseudo artifacts/preds_round1.csv
python train_stack.py --seeds 555 1717 8080 3141 9999 --tag stack_c --pseudo artifacts/preds_round1.csv

# Step 4 - embedding MLP
python train_mlp.py

# Step 5 - calibrate, blend, and write submission.csv
python build_submission.py
```

Each stack run takes roughly 45 minutes on an 8-core machine. The MLP takes about 6 minutes.

## What actually mattered

### Features

**DEPARTMENT** was the single most important feature by a wide margin. Surgical and medical departments have dramatically different LOS distributions, and the model leans on it heavily.

**Post-admission signals** (`ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`) carried the majority of predictive weight. These are known at discharge, which makes them fair game for the task.

**OOF target encoding** for `DOCTOR`, `DX_CODE`, `DIAGNOSIS_SUBCAT_CODE`, and `DEPARTMENT` gave the largest single categorical gain over one-hot or label encoding.

**Dropped columns and why:**

| Column | Reason dropped |
|--------|----------------|
| `DIAGNOSIS_ICD_CODE` | r = 1.000 with `DIAGNOSIS_SUBCAT_CODE` - pure duplicate |
| `DISCH_NURSE_ID` | Signal collapsed completely in the updated dataset |
| `HOSPITAL` (direct) | Near-zero LOS signal across 39 hospitals; kept as frequency encoding instead |

---

### Modelling

**Exporting raw floats** instead of rounded integers was worth ~0.020 RMSE on its own. Integer-rounding a continuous regression output is a silent, easy-to-miss mistake.

**Pseudo-labelling saturates fast.** Two self-training rounds gave +0.005 LB improvement. A third round showed zero gain - once the model is confident enough in its own test predictions, there's nothing new to learn from them.

**Isotonic calibration** fixed the two systematic shape errors the meta-learner couldn't correct:
- Over-prediction in the 47-51 day range
- Under-prediction around 28-33 days

**Diversity beats quantity.** This was the most surprising finding:

| Addition | LB gain |
|----------|---------|
| 4th correlated tree stack | +0.0002 |
| 1 decorrelated MLP at 5% weight | ~+0.002 |

Once the three tree stacks are correlating at 0.9998 with each other, adding a fourth is almost pure noise. One diverse signal is worth far more.
