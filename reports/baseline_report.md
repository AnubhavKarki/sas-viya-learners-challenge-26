# SAS Viya 2026 — Baseline Modeling Report
*Generated: 2026-07-14*

---

## 0. Setup & Decisions Applied (EDA Open Items)

Before any modeling, the 15 EDA open questions (Section 11 of the EDA report) were resolved
with baseline-safe defaults. Decisions are documented here rather than implied by code.

| # | Column / Issue | Baseline Decision | Justification |
|---|----------------|-------------------|---------------|
| 1 | Leakage columns | `ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`, `DISCH_NURSE_ID` excluded from Model 2; included in Model 3 since they exist in test.csv | Establishes Model 2 as clinical baseline, Model 3 as competition ceiling |
| 2 | `ADMIT_DATE`/`DISCHARGE_DATE` features | Not used. Only `ADMIT_MTH` used for seasonal signal | Dates absent from test.csv |
| 3 | `NUM_CHRONIC_COND` coercion | `pd.to_numeric(..., errors='coerce')` — coercion failures → NaN | EDA confirmed 99.9%+ coerce cleanly |
| 4 | `ENCOUNTER_KEY` | Used as submission row ID only, never as feature | Unique per row; no predictive signal |
| 5 | Procedure missingness | `HAS_PROCEDURE = (OPERATION_COUNT > 0)` binary flag; all procedure text/code columns dropped | Missingness is MNAR (OPERATION_COUNT=0 explains all 40,876 NaN rows) |
| 6 | `DRG_APR_SEVERITY` ordering | Encoded as ordinal integer: 1=Minor, 2=Moderate, 3=Major, 4=Extreme | Clinical ordering is meaningful; preserving it captures monotonic severity→LOS relationship |
| 7 | `NUM_VISITS` interpretation | Used as-is (historical/lifetime visit count, not this file's row count) | EDA confirmed 81.4% mismatch with row count — it is a pre-existing feature |
| 8 | Patient uniqueness / CV strategy | Standard 5-fold KFold, no group stratification | `PATIENT_NUMBER` is 1:1 with `ENCOUNTER_KEY`; no repeat patients |
| 9 | `HOSPITAL` encoding | Frequency-encoded (count of occurrences, globally on train set) | Target-encoding deferred; requires careful OOF setup |
| 10 | `DOCTOR`/`DISCH_NURSE_ID` | `DOCTOR` frequency-encoded; `DISCH_NURSE_ID` used as raw numeric in Model 3 | Target-encoding deferred |
| 11 | `DEPARTMENT` contamination | "Hosp 46" and "Hosp 39" kept as opaque category labels | 13,711 rows affected; ETL error but semantically safe for tree models |
| 12 | `DRG_APR_SEVERITY` whitespace | `.str.strip()` applied; "." → NaN (SAS sentinel) | All 127,802 values had 11 leading spaces; "." = 173 rows of missing severity |
| 13 | `ORDER_TOTAL_CHARGES` sentinel | 43 rows with value exactly -2104 → NaN | Single magic number = billing credit code, not a real charge |
| 14 | `ADMIT_LOS = 0` (518 rows) | Kept | Valid same-day discharges; clinically real |
| 15 | `PATIENT_AGE` floor at 27 | No action; flagged as OOD risk | Dataset is restricted to adults ≥27; flag if test ever contains younger patients |

**Additional cleaning applied:**
- `DIAGNOSIS_ICD_CODE` dropped (perfectly collinear with `DIAGNOSIS_SUBCAT_CODE`, r=1.000)
- `DRG_APR_CODE` coerced to numeric (object → float64)
- All object columns stripped of leading/trailing whitespace

---

## 1. Cross-Validation Setup

```
KFold(n_splits=5, shuffle=True, random_state=42)
```

All three models use **identical fold assignments** — the same `random_state=42` on the same
`train.csv` row order ensures fold indices are deterministic and comparable across models.

Standard KFold was chosen over StratifiedKFold following the 2025 solution finding:
> "Reverted to KFold instead of StratifiedKFold for better natural variation."
For regression, StratifiedKFold requires binning the continuous target, which can create
artificially homogeneous folds. Standard KFold with shuffle is the correct choice.
*(Reference: `reports/2025_solution_review.md` — Section 4)*

---

## 2. Per-Model Results

### Model 1 — Naive Group-Median Floor

Predicts `ADMIT_LOS` as the median LOS within each `(DEPARTMENT, DRG_APR_SEVERITY)` group
seen in the training fold. Falls back to overall training median for unseen combinations.
No machine learning.

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 5.0027 | 2.9716 |
| 2 | 5.0075 | 2.9648 |
| 3 | 4.9361 | 2.9191 |
| 4 | 4.9949 | 2.9325 |
| 5 | 4.9945 | 2.9423 |
| **CV Mean ± Std** | **4.987 ± 0.026** | **2.946 ± 0.020** |

**Interpretation:** The naive model barely outperforms predicting the overall mean (overall std
of `ADMIT_LOS` = 4.95). The group-median by `DEPARTMENT × DRG_APR_SEVERITY` adds only marginal
signal in isolation — ~RMSE 4.99 vs ~RMSE 4.95 for the global mean predictor. This is the
absolute floor: any ML model worth deploying must beat RMSE 4.99.

---

### Model 2 — Admission-Time CatBoost

**Feature set (28 features):** Excludes `ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`,
`DISCH_NURSE_ID` (all post-admission). Includes: patient demographics, diagnosis codes
and groups, DRG codes and severity, geographic features, admit month, order sets used,
chronic condition count, procedure flag, historical visit count, frequency-encoded hospital/doctor.

**Hyperparameters (baseline, lightly tuned):**
```
iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, random_seed=42
```

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 1.4857 | 1.0496 |
| 2 | 1.5242 | 1.0597 |
| 3 | 1.4749 | 1.0352 |
| 4 | 1.4616 | 1.0267 |
| 5 | 1.5158 | 1.0526 |
| **CV Mean ± Std** | **1.492 ± 0.024** | **1.045 ± 0.012** |

**Implied R² ≈ 0.909** — CatBoost with DRG codes and severity alone achieves 91% explained
variance at admission time. This is a clinically remarkable result, largely explained by the
fact that MS-DRG (Medicare Severity DRG) codes are explicitly designed by CMS to cluster
expected hospital resource utilization, making them a near-direct proxy for LOS.

---

### Model 3 — Full-Feature CatBoost

**Identical hyperparameters and folds as Model 2.** Adds the four post-admission features:
`ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`, `DISCH_NURSE_ID`.

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 0.8916 | 0.6054 |
| 2 | 0.8853 | 0.5946 |
| 3 | 0.8784 | 0.5914 |
| 4 | 0.8494 | 0.5812 |
| 5 | 0.8914 | 0.6015 |
| **CV Mean ± Std** | **0.879 ± 0.016** | **0.595 ± 0.008** |

**Implied R² ≈ 0.968**

---

## 3. Model 2 vs Model 3 — Delta Analysis

| Metric | Model 2 (admission-only) | Model 3 (full features) | Delta |
|--------|--------------------------|-------------------------|-------|
| CV RMSE | 1.492 | 0.879 | **−0.613** |
| CV MAE | 1.045 | 0.595 | **−0.450** |
| Implied R² | 0.909 | 0.968 | **+0.059** |

**The four post-admission features (ICU_DAYS, ORDER_TOTAL_CHARGES, DISCHARGED_TO,
DISCH_NURSE_ID) reduce RMSE by 0.613 — a 41.1% relative improvement.**

`ICU_DAYS` alone accounts for 25.3% of Model 3's feature importance, which is consistent
with its Pearson r = 0.611 found in the EDA (the single highest numeric correlation with
ADMIT_LOS). The message is clear: a patient's ICU utilization during the stay is a powerful
explainer of total LOS, but it is a *consequence* of severity rather than a cause, and is
unavailable at the time of admission in a real clinical setting.

For **competition purposes**, Model 3's feature set is legitimate (all four columns exist
in `test.csv`) and is the correct choice for maximizing leaderboard RMSE. For **clinical
deployment**, Model 2 is the appropriate architecture.

---

## 4. Feature Importance

### Model 2 — Admission-Time Features (top 20 by mean importance across 5 folds)

| Rank | Feature | Importance (%) | Notes |
|------|---------|---------------|-------|
| 1 | `ADMIT_MTH` | 12.06 | Strongest seasonal signal; Aug/Sep show +1.5 days mean LOS vs baseline |
| 2 | `DOCTOR_freq` | 10.87 | Doctor volume → specialty proxy; high-volume doctors see different case mixes |
| 3 | `DRG_APR_SEVERITY` | 8.49 | Ordinal severity 1–4; monotonic LOS increase confirmed in EDA |
| 4 | `X` | 6.78 | Longitude (geographic); likely capturing hospital cluster effects |
| 5 | `NUM_CHRONIC_COND` | 6.09 | More chronic conditions → longer stays (Pearson r=0.074 but non-linear) |
| 6 | `DEPARTMENT` | 5.90 | HEART dominates (54% of rows); TRANSPLANT/ONCOLOGY have high LOS |
| 7 | `ZIP` | 5.79 | Geographic granularity below city/county level |
| 8 | `CITY` | 5.72 | Geographic; CatBoost encoding captures city-level utilization patterns |
| 9 | `MS_DRG_CODE` | 5.23 | CMS billing code designed to cluster by expected resource utilization |
| 10 | `Y` | 5.12 | Latitude — geographic complement to longitude |
| 11 | `DX_CODE` | 5.01 | Diagnosis code |
| 12 | `DRG_APR_CODE` | 4.42 | APR-DRG code (complements MS_DRG) |
| 13 | `PATIENT_AGE` | 4.11 | Older patients → longer stays (weakly linear, strongly non-linear) |
| 14 | `DIAGNOSIS_SUBCAT_CODE` | 3.48 | ICD subcategory numeric code |
| 15 | `DX_GROUP` | 3.15 | Coarse diagnosis group (CHF, AMI, etc.) |

**Notable finding:** `ADMIT_MTH` being the top feature is initially surprising given its
Pearson r = 0.027 in the EDA, but the EDA seasonal table shows months 8–9 have mean LOS
of 7.29–7.32 vs April's 4.89 — a range of 2.4 days. CatBoost captures non-linear month
interactions (e.g., severe cardiac patients in August may have longer stays than the same
diagnosis in spring). This warrants a quarterly/seasonal bin feature in Stage 4.

**`DOCTOR_freq` at rank 2** confirms that a physician's patient volume encodes specialty
and practice-style signal. Upgrading to OOF target-encoding in Stage 4 is expected to
unlock additional signal without leakage.

### Model 3 — Full-Feature (top 20)

| Rank | Feature | Importance (%) | Notes |
|------|---------|---------------|-------|
| 1 | `ICU_DAYS` | 25.28 | Dominant post-admission signal; EDA r=0.611 |
| 2 | `ORDER_TOTAL_CHARGES` | 10.76 | Total charges accumulate over stay; proxy for intensity |
| 3 | `NUM_CHRONIC_COND` | 10.07 | Rises from rank 5 in Model 2 — clinical complexity matters more given known severity |
| 4 | `DOCTOR_freq` | 7.27 | Consistent across both models |
| 5 | `ADMIT_MTH` | 5.31 | Drops from rank 1 when post-admission features absorb variance |
| 6 | `DX_GROUP` | 5.27 | |
| 7 | `DRG_APR_CODE` | 4.10 | |
| 8 | `DISCHARGED_TO` | 3.52 | Discharge disposition encodes complexity (e.g., skilled nursing = long stay) |
| 9 | `DEPARTMENT` | 2.97 | |
| 10 | `CITY` | 2.86 | |

**Key shift from Model 2 → Model 3:** `ICU_DAYS` dominates and displaces `ADMIT_MTH`,
`DRG_APR_SEVERITY`, and the geographic features. Once we know actual ICU utilization,
the pre-admission severity proxies matter less. `DRG_APR_SEVERITY` drops from rank 3
(8.49%) to rank 14 (2.24%), confirming it was acting as a severity proxy in Model 2
that ICU_DAYS now directly measures.

---

## 5. Summary Table

| Model | Description | CV RMSE | CV MAE | Implied R² |
|-------|-------------|---------|--------|-----------|
| Model 1 | Naive group-median (DEPT × SEV) | 4.987 ± 0.026 | 2.946 ± 0.020 | ~0.0 (floor) |
| Model 2 | CatBoost, admission-time only | 1.492 ± 0.024 | 1.045 ± 0.012 | ~0.909 |
| Model 3 | CatBoost, full features | **0.879 ± 0.016** | **0.595 ± 0.008** | **~0.968** |
| M2→M3 delta | Post-admission 4-column lift | −0.613 | −0.450 | +0.059 |

---

## 6. Progression Path — What to Do Next

### Decision point: Admission-only vs Full-feature

Given the 0.613 RMSE delta between Models 2 and 3, and that all four post-admission columns
ARE present in `test.csv`, **the primary competition path should be Model 3 (full features)**.
The admission-only model (Model 2) is valuable for clinical interpretation but is not the
competition-optimal direction.

### Recommended Stage 4 priorities (in order):

**1. Target-encode `DOCTOR` and `HOSPITAL` (highest-expected gain, ~0.05–0.15 RMSE)**
- Replace frequency encoding with OOF target encoding using the same 5-fold split
- `DOCTOR_freq` ranks 2nd in Model 2 and 4th in Model 3 with only count signal
- With OOF target means, doctor-level LOS patterns will be directly encoded
- Use `category_encoders.TargetEncoder` with `smoothing` to avoid overfitting small doctors
- Implement strictly within CV folds to prevent leakage
- *(2025 lesson: target-encoding requires OOF logic; the 2025 solution avoided this by using
  only features that didn't need it. This dataset requires it.)*

**2. Feature interactions for clinical hypotheses**
- `PATIENT_AGE × DRG_APR_SEVERITY`: old + severe patients have disproportionately long stays
- `NUM_CHRONIC_COND × DRG_APR_SEVERITY`: comorbidity-severity interaction
- `ICU_DAYS × OPERATION_COUNT`: surgery + ICU is a distinct LOS phenotype
- `ADMIT_MTH` seasonal features: quarterly bins (Q1/Q2/Q3/Q4) or sin/cos encoding

**3. Upgrade geographic features**
- `X` (longitude) and `Y` (latitude) combined show high importance
- Consider: hospital cluster assignment via K-means on (X, Y), median LOS per cluster
- `ZIP` is high-cardinality numeric — consider grouping to 3-digit ZIP

**4. Hyperparameter tuning on Model 3 architecture**
- The baseline used `iterations=1000, depth=6, lr=0.05` — not tuned
- Expected gains from tuning: 0.03–0.08 RMSE
- Grid: depth ∈ {4, 6, 8}, l2_leaf_reg ∈ {1, 3, 10}, iterations with early stopping
- Use Optuna or a coarse grid, evaluated on the same 5 folds

**5. Stacking/ensembling (following 2025 methodology)**
- 2025 result showed RF + GBR + LGB + XGB + Ridge meta gave the best score
- Here: CatBoost (Model 3) + LightGBM + XGBoost as base models, Ridge as meta
- Stacking requires clean OOF predictions — these are already saved to `reports/baseline/`

**6. Log1p target transform experiment**
- EDA noted `ADMIT_LOS` skewness = 2.70; log1p reduces to 0.37
- Train with `log1p(ADMIT_LOS)`, predict, then `expm1()` the predictions
- Tree-based models are less sensitive to this but may still benefit on extreme tails
- Evaluate both approaches on same folds; pick lower raw-scale RMSE

### What NOT to do next
- Do not increase CatBoost `iterations` without early stopping — will overfit
- Do not one-hot-encode `CITY` or `COUNTY_NAME` — already handled natively by CatBoost
- Do not use `ADMIT_DATE` or `DISCHARGE_DATE` — absent from test.csv
- Do not submit Model 2 — it is ~0.61 RMSE worse than Model 3

---

## 7. Submission Files

| File | Model | CV RMSE |
|------|-------|---------|
| `reports/baseline/submission_model_1.csv` | Naive group-median | 4.987 |
| `reports/baseline/submission_model_2.csv` | CatBoost, admission-only | 1.492 |
| `reports/baseline/submission_model_3.csv` | CatBoost, full features | 0.879 |

OOF prediction files saved for stacking:
- `reports/baseline/oof_model_1.csv`
- `reports/baseline/oof_model_2.csv`
- `reports/baseline/oof_model_3.csv`

**No submissions have been made to Kaggle.** These files are ready for your review.
Recommend Model 3 as the first submission to establish a leaderboard baseline.

---

## 8. Connections to 2025 Solution

| 2025 Finding | Applied Here |
|-------------|-------------|
| KFold > StratifiedKFold for regression | Used standard KFold(5, shuffle=True, seed=42) |
| RF+GBR workhorse, Ridge meta | Deferred to Stage 5 (stacking); OOF predictions saved |
| Feature filtering by correlation caused information loss | Did NOT apply correlation-based filtering; trusted CatBoost importance instead |
| Scaling essential for Ridge meta-learner | Not needed for CatBoost alone; will apply inside stacking meta-layer |
| Interaction terms boosted 2025 score | Deferred to Stage 4; clinical hypotheses identified above |
| OOF target encoding needed careful setup | Correctly deferred all target encoding to Stage 4 |

*(Reference: `reports/2025_solution_review.md`)*

---

*End of baseline report.*
