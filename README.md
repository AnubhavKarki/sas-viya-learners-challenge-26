# SAS Viya for Learners Challenge 2026

**Kaggle Competition — Workbench Track Submission**

Predict hospital length of stay (`ADMIT_LOS`, in days) from clinical and administrative encounter records.

---

## Competition

- **Platform:** Kaggle / SAS Viya for Learners Challenge 2026
- **Task:** Regression — predict `ADMIT_LOS` (integer days, range 0–51)
- **Metric:** RMSE (lower is better)
- **Track:** Workbench (code submission)

---

## Results

| Stage | Model | CV RMSE | Kaggle LB |
|-------|-------|---------|-----------|
| Baseline | Naive group median | 4.987 | — |
| Stage 3 | CatBoost admission-only | 2.492 | — |
| Stage 3 | CatBoost full features | 1.990 | — |
| Stage 4 | CB + LGB + XGB → Ridge stack | 1.993 | — |
| Stage 5 | + Optuna tuning + HGBM + new features | 1.982 | 1.96 |
| Stage 7 | + Rounding fix + 3-seed bagging | 1.978 | 1.947 |
| **Stage 9A** | **+ OOF TE for DX_CODE, DIAG_SUBCAT, DEPT** | **1.977** | **1.947** |

**Current leaderboard position: #1**

---

## Approach

### Model architecture

Four-model stacked ensemble with Ridge meta-learner:

- **CatBoost** (Optuna-tuned, depth=6, lr=0.0405) — native categorical handling
- **LightGBM** (Optuna-tuned, num_leaves=65, lr=0.0372)
- **XGBoost** (lr=0.03, max_depth=7)
- **HistGradientBoosting** (Optuna-tuned, lr=0.0105)

All base models train on `log1p(ADMIT_LOS)` and predictions are recovered via `expm1`. Raw float predictions are used (no integer rounding — rounding costs ~0.020 RMSE on LB).

### Key design choices

- **3-seed bagging** [42, 7, 123]: each seed produces independent 5-fold CV predictions; OOF and test predictions are averaged across seeds before Ridge stacking. Reduces variance.
- **OOF target encoding** (fold-safe, smoothing=20):
  - `DOCTOR` — high-cardinality ID, OOF TE replaces raw value
  - `DX_CODE_TE` — 55 categories, group-mean std=2.76 (highest signal)
  - `DIAGNOSIS_SUBCAT_CODE_TE` — 21 categories, group-mean std=2.68
  - `DEPARTMENT_TE` — 10 categories, group-mean std=2.35 (#1 feature by importance)
- **Hospital encoding:** frequency encoding (hospital mean LOS has near-zero variance across 39 hospitals; frequency is the useful proxy)
- **Ridge meta-learner:** fits on OOF predictions with CV RMSE estimate; final coefs CB:0.729, LGB:0.276, XGB:0.105, HGBM:−0.069

### Feature engineering (17 interaction features)

| Feature | Formula |
|---------|---------|
| `AGE_x_SEVERITY` | `PATIENT_AGE × DRG_APR_SEVERITY` |
| `CHRONIC_x_SEVERITY` | `NUM_CHRONIC_COND × DRG_APR_SEVERITY` |
| `ICU_x_CHRONIC` | `ICU_DAYS × NUM_CHRONIC_COND` |
| `ICU_x_OPERATION` | `ICU_DAYS × OPERATION_COUNT` |
| `LOG_CHARGES` | `log1p(ORDER_TOTAL_CHARGES)` |
| `CHARGE_PER_ICU` | `ORDER_TOTAL_CHARGES / (ICU_DAYS + 1)` |
| `ICU_DAYS_SQRT` | `sqrt(ICU_DAYS)` |
| `PATIENT_AGE_SQ` | `PATIENT_AGE²` |
| `ADMIT_QUARTER` | Quarter from `ADMIT_MTH` |
| `IS_SUMMER` | 1 if `ADMIT_MTH` ∈ {7,8,9} |
| `MONITOR_x_ICU` | `MONITORING_HOURS × ICU_DAYS` |
| `COMORBID_x_SEV` | `COMORBIDITY_INDEX × DRG_APR_SEVERITY` |
| `TEAM_x_COMORBID` | `CARE_TEAM_SIZE × COMORBIDITY_INDEX` |
| `LOG_MONITORING` | `log1p(MONITORING_HOURS)` |
| `MONITOR_PER_COMORBID` | `MONITORING_HOURS / (COMORBIDITY_INDEX + 1)` |
| `COMORBID_SQ` | `COMORBIDITY_INDEX²` |
| `TEAM_x_ICU` | `CARE_TEAM_SIZE × ICU_DAYS` |

### Critical finding: rounding penalty

All prior submissions exported `ADMIT_LOS` as rounded integers. Cross-validation was always computed on raw floats. The delta:

- CV RMSE (raw floats): **1.9812**
- CV RMSE (rounded integers): **2.0017**
- Rounding penalty: **+0.0205 RMSE** on every prior submission

Switching to raw float export was the single largest confirmed improvement.

---

## Repository structure

```
sas-viya-2026-submission/
├── train_and_predict.py     # Full self-contained pipeline
├── requirements.txt         # Pinned dependencies
├── submission.csv           # Current best predictions (Stage 9A)
├── reports/
│   ├── baseline_report.md   # Full stage-by-stage results and decisions
│   └── train_eda_report.md  # EDA findings
└── data/                    # NOT included — download from Kaggle
    ├── train.csv
    └── test.csv
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

On Apple Silicon (M-series) Macs, CatBoost/LightGBM need the ARM64 OpenMP library:

```bash
# If you get "Library not loaded: @rpath/libomp.dylib":
cp $(brew --prefix libomp)/lib/libomp.dylib venv/lib/libomp.dylib
export DYLD_LIBRARY_PATH=venv/lib:$DYLD_LIBRARY_PATH
```

---

## Data

Download the competition data from Kaggle and place it under `data/`:

```
data/
├── train.csv
└── test.csv
```

---

## Running the pipeline

```bash
python train_and_predict.py
```

This will:
1. Load and clean `data/train.csv` and `data/test.csv`
2. Apply feature engineering (17 interaction features)
3. Apply fold-safe OOF target encoding per fold
4. Train CB + LGB + XGB + HGBM across 3 seeds × 5 folds
5. Fit Ridge meta-learner on seed-averaged OOF predictions
6. Write raw float predictions (clipped to [0, 51]) to `submission.csv`

**Custom paths:**

```bash
python train_and_predict.py --train path/to/train.csv --test path/to/test.csv --out my_submission.csv
```

---

## Key findings

- **DEPARTMENT** is the #1 feature by importance (22%) — patients in surgical vs medical departments have dramatically different LOS profiles
- **Post-admission features** (`ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`) drive the majority of signal; removing them degrades RMSE by ~1.1
- **Integer rounding** costs ~0.020 RMSE on LB — always export raw floats for regression targets
- `DIAGNOSIS_ICD_CODE` dropped (r=1.000 with `DIAGNOSIS_SUBCAT_CODE`)
- `DISCH_NURSE_ID` dropped (r=−0.001 in updated dataset — signal collapsed)
- `HOSPITAL` mean LOS varies only 5.58–5.94 across 39 hospitals — near-zero LOS signal; frequency encoding used instead
- OOF target encoding for `DX_CODE` (55 categories, group std=2.76) provided the largest untapped categorical signal
