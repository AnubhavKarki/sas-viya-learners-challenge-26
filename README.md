# SAS Viya for Learners Challenge 2026

**Kaggle Competition — Workbench Track Submission**

Predict hospital length of stay (`ADMIT_LOS`, in days) from clinical and administrative encounter records.

---

## Competition

- **Platform:** Kaggle / SAS Viya for Learners Challenge 2026
- **Task:** Regression — predict `ADMIT_LOS` (integer days)
- **Metric:** RMSE (lower is better)
- **Track:** Workbench (code submission)

---

## Approach

### Baseline progression

| Model | Features | CV RMSE |
|-------|----------|---------|
| Naive (group median) | DEPARTMENT × DRG_APR_SEVERITY | 4.987 ± 0.026 |
| CatBoost (admission-only) | 28 admission-time features | 1.492 ± 0.024 |
| CatBoost (full-feature) | + ICU_DAYS, ORDER_TOTAL_CHARGES, DISCHARGED_TO, DISCH_NURSE_ID | 0.879 ± 0.016 |

### Final model (Stage 4 stacked ensemble)

A three-model stack (CatBoost + LightGBM + XGBoost) with a Ridge meta-learner, using:

- **Feature engineering:** `CHARGE_PER_ICU = ORDER_TOTAL_CHARGES / (ICU_DAYS + 1)`, log charges, ICU sqrt, age × severity interaction, and six additional derived features
- **OOF target encoding:** `DOCTOR` and `DISCH_NURSE_ID` encoded with smoothed mean encoding computed inside each CV fold (no leakage)
- **5-fold KFold** (shuffle=True, seed=42) for all CV estimates and OOF meta-features
- **Integer rounding** of final predictions (LOS is always a whole number of days)

| Model | OOF RMSE |
|-------|----------|
| CatBoost | 0.0944 |
| LightGBM | 0.0227 |
| XGBoost | 0.0202 |
| **Ridge Stack** | **0.0197** |

---

## Repository structure

```
sas-viya-2026-submission/
├── train_and_predict.py     # Full pipeline: clean → engineer → encode → train → predict
├── requirements.txt         # Pinned dependencies
├── reports/
│   ├── train_eda_report.md  # EDA findings and open-item resolutions
│   └── baseline_report.md   # Baseline model comparison and feature importances
└── data/                    # NOT included — download from Kaggle (see below)
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

On Apple Silicon (M-series) Macs, LightGBM/XGBoost need the ARM64 OpenMP library:

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
2. Apply feature engineering and OOF target encoding
3. Train CatBoost, LightGBM, and XGBoost with 5-fold CV
4. Fit a Ridge meta-learner on OOF predictions
5. Write integer-rounded predictions to `submission.csv`

**Custom paths:**

```bash
python train_and_predict.py --train path/to/train.csv --test path/to/test.csv --out my_submission.csv
```

---

## Key findings

- `CHARGE_PER_ICU` (charges divided by ICU days + 1) is the single most important engineered feature — it exposes the near-linear billing formula underlying LOS
- The four post-admission columns (`ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`, `DISCH_NURSE_ID`) drive the majority of the predictive signal; the Model 2 → Model 3 gap is −0.613 RMSE
- `DIAGNOSIS_ICD_CODE` was dropped (perfectly collinear with `DIAGNOSIS_SUBCAT_CODE`, r=1.000)
- 43 rows with `ORDER_TOTAL_CHARGES = −2104` treated as missing (SAS billing sentinel)
- Files read with `encoding="utf-8-sig"` to strip the BOM that SAS adds to CSV exports
