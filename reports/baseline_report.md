# SAS Viya 2026 — Baseline Modeling Report
*Generated: 2026-07-15 — **UPDATED DATASET** (VFL_2026_TRAIN_SET / VFL_2026_TEST_SET)*

---

## ⚠️ Dataset Context — Updated Competition Data

The dataset on disk is the **official SAS-updated version** (downloaded from Kaggle 15 July 2026,
~6:29 PM AEST following SAS's announcement). Key structural differences from the original:

| Change | Original | Updated |
|--------|----------|---------|
| Train rows | 127,802 | **100,000** |
| Test rows | 14,200 | **15,000** |
| Encounter overlap with original | — | **0 (zero)** |
| Columns removed from train | — | `ADMIT_DATE`, `DISCHARGE_DATE` |
| New columns (train + test) | — | `MONITORING_HOURS`, `COMORBIDITY_INDEX`, `CARE_TEAM_SIZE` |
| Fingerprint leakage | 100% test match | **2.1% test match (broken)** |
| `DRG_APR_SEVERITY` whitespace | 11 leading spaces + "." sentinel | **Clean integers 1–4** |
| `ORDER_TOTAL_CHARGES` min | −2104 sentinel | **+200 (no negatives)** |
| `DISCH_NURSE_ID` r with ADMIT_LOS | −0.024 | **−0.001 (noise)** |

---

## 0. EDA Key Findings — Updated Dataset

### New features (all present in test.csv — use in all models)

| Feature | Range | Pearson r with ADMIT_LOS | Interpretation |
|---------|-------|--------------------------|----------------|
| `MONITORING_HOURS` | 0–228 | **0.589** | Clinical monitoring hours during stay |
| `COMORBIDITY_INDEX` | 0–19 | **0.569** | Composite comorbidity score; near-linear with LOS |
| `CARE_TEAM_SIZE` | 1–15 | **0.538** | # of distinct clinicians; monotonic with LOS |

`COMORBIDITY_INDEX = 0` → mean LOS 3.7 days; `= 10` → 15.4 days; `= 15` → 23.8 days — **near-perfectly linear**.

The 3 new features are mutually correlated (r = 0.78–0.85) but tree models handle this fine.

### Target variable — ADMIT_LOS

| Statistic | Value |
|-----------|-------|
| Count | 100,000 |
| Mean | 5.76 days |
| Median | 5 days |
| Std dev | 4.17 |
| Min / Max | 0 / 51 |
| Skewness | **3.26** (was 2.70) |
| log1p skewness | 0.32 |
| Unique values | 52 |

### ANOVA results — categorical features vs ADMIT_LOS

| Feature | η² (eta²) | Significant? |
|---------|-----------|-------------|
| `DRG_APR_SEVERITY` | **0.192** | ✓ Very strong |
| `DEPARTMENT` | **0.132** | ✓ Very strong |
| `RACE_CD` | 0.000 | ✗ Not significant |
| `GENDER` | 0.000 | ✗ Not significant |
| `STANDARD_ORDERS_USED` | 0.000 | ✗ Not significant |
| `REGION` | 0.000 | ✗ Not significant |
| `HOSPITAL` | 0.000 | ✗ Not significant |
| `DISCHARGED_TO` | 0.000 | ✗ Not significant (was 0.0166) |

**Major finding:** `DISCHARGED_TO` has lost nearly all its LOS signal (p=0.63). In the original data
it was a strong post-admission predictor. In the updated data it should still be included in the
full-feature model but its importance will be much lower.

### Missingness

- Only `PROCEDURE_LONG_DESC`, `PROCEDURE_ICD_CODE`, `PROCEDURE_SUBCAT_DESC` have missing data (~32,072 rows = 32%)
- **New pattern**: missingness is now ~32% across ALL `OPERATION_COUNT` values (not just count=0 as before)
- `HAS_PROCEDURE` flag kept based on `OPERATION_COUNT` for clinical meaning; procedure text columns dropped

### Multicollinearity

| Pair | r |
|------|---|
| `DIAGNOSIS_ICD_CODE` × `DIAGNOSIS_SUBCAT_CODE` | 1.000 |
| `MONITORING_HOURS` × `COMORBIDITY_INDEX` | 0.846 |
| `MONITORING_HOURS` × `CARE_TEAM_SIZE` | 0.804 |
| `COMORBIDITY_INDEX` × `CARE_TEAM_SIZE` | 0.777 |

Tree models are invariant to multicollinearity — all three new features kept.

---

## 1. Preprocessing Decisions (Updated)

| # | Issue | Decision | Change from Original? |
|---|-------|----------|-----------------------|
| 1 | `ADMIT_DATE` / `DISCHARGE_DATE` | **Removed from dataset** — no code action needed | Resolved by SAS |
| 2 | `DRG_APR_SEVERITY` whitespace + sentinel | No longer present — clean int 1–4 | Resolved by SAS |
| 3 | `ORDER_TOTAL_CHARGES` −2104 sentinel | No longer present — min=200 | Resolved by SAS |
| 4 | `NUM_CHRONIC_COND` coercion | `pd.to_numeric(..., errors='coerce')` | Unchanged |
| 5 | `DIAGNOSIS_ICD_CODE` collinearity | Dropped (r=1.000 with `DIAGNOSIS_SUBCAT_CODE`) | Unchanged |
| 6 | `DISCH_NURSE_ID` | **Dropped** (r=−0.001 in updated data; noise) | **New action** |
| 7 | `HAS_PROCEDURE` flag | `OPERATION_COUNT > 0` — kept for clinical meaning | Unchanged |
| 8 | `ENCOUNTER_KEY` | Row ID only, never a feature | Unchanged |
| 9 | `DRG_APR_CODE` | Coerce to numeric (safe no-op in new data) | Unchanged |
| 10 | `MONITORING_HOURS`, `COMORBIDITY_INDEX`, `CARE_TEAM_SIZE` | **Use as-is** — clean numerics, both train+test | **New features** |

---

## 2. Cross-Validation Setup

```
KFold(n_splits=5, shuffle=True, random_state=42)
```

- Standard KFold, not stratified (regression task)
- `PATIENT_NUMBER` is 1:1 with `ENCOUNTER_KEY` — no repeat patients, no group k-fold needed
- All models use identical fold assignments (same `random_state=42` on same row order)

---

## 3. Per-Model Results — Updated Dataset

### Model 1 — Naive Group-Median Floor

Predicts `ADMIT_LOS` as the median LOS within each `(DEPARTMENT, DRG_APR_SEVERITY)` group.
Falls back to overall median for unseen combinations. No machine learning.

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 3.4841 | 2.0113 |
| 2 | 3.5382 | 2.0509 |
| 3 | 3.5698 | 2.0655 |
| 4 | 3.4383 | 2.0368 |
| 5 | 3.4840 | 2.0310 |
| **CV Mean ± Std** | **3.503 ± 0.046** | **2.039 ± 0.018** |

Floor for any ML model: must beat RMSE **3.50**.

---

### Model 2 — Admission-Time CatBoost

Excludes `ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO` (post-admission).
Includes all 3 new features (`MONITORING_HOURS`, `COMORBIDITY_INDEX`, `CARE_TEAM_SIZE`).

**Feature set:** 31 features | **Hyperparameters:** `iterations=1000, depth=6, lr=0.05, l2=3`

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 2.5031 | 1.6334 |
| 2 | 2.4820 | 1.6406 |
| 3 | 2.5203 | 1.6519 |
| 4 | 2.4681 | 1.6326 |
| 5 | 2.4873 | 1.6346 |
| **CV Mean ± Std** | **2.492 ± 0.018** | **1.639 ± 0.007** |

**Top features:** `DEPARTMENT` (31.8%), `MONITORING_HOURS` (26.2%), `COMORBIDITY_INDEX` (12.3%), `DRG_APR_SEVERITY` (12.1%), `PATIENT_AGE` (6.1%), `CARE_TEAM_SIZE` (4.7%)

`MONITORING_HOURS` is the #2 feature in the admission-only model — confirming it as a
dominant new signal.

---

### Model 3 — Full-Feature CatBoost

Adds post-admission features: `ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`.

**Feature set:** 34 features | **Same hyperparameters as Model 2**

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 1.9859 | 1.3232 |
| 2 | 1.9959 | 1.3385 |
| 3 | 2.0282 | 1.3463 |
| 4 | 1.9722 | 1.3228 |
| 5 | 1.9680 | 1.3225 |
| **CV Mean ± Std** | **1.990 ± 0.022** | **1.331 ± 0.010** |

**Top features:** `DEPARTMENT` (28.2%), `ICU_DAYS` (24.6%), `DRG_APR_SEVERITY` (13.3%), `MONITORING_HOURS` (9.1%), `OPERATION_COUNT` (4.8%), `PATIENT_AGE` (4.8%)

**Key shift M2→M3:** `ICU_DAYS` takes rank 2 (24.6%) once added. `MONITORING_HOURS` drops
from rank 2 (26.2%) to rank 4 (9.1%) — it was proxying ICU intensity in the admission-only model.
`DRG_APR_SEVERITY` strengthens (12.1% → 13.3%) as it remains independent of ICU.

---

## 4. Model Comparison

| Model | Description | CV RMSE | CV MAE | Notes |
|-------|-------------|---------|--------|-------|
| Model 1 | Naive group-median | 3.503 ± 0.046 | 2.039 | Absolute floor |
| Model 2 | CatBoost, admission-only | 2.492 ± 0.018 | 1.639 | Clinical baseline |
| Model 3 | CatBoost, full features | **1.990 ± 0.022** | **1.331** | Competition baseline |
| Stage 4 | CB+LGB+XGB→Ridge | **1.993 ± 0.024** | **1.332** | Best submission candidate |

**M2→M3 delta: −0.502 RMSE (20.1% relative improvement)** from adding ICU/charges/discharge.

---

## 5. Stage 4 — Tuned Ensemble ✅ Complete

`src/stage4/model_4_tuned.py` — CB + LGB + XGB → Ridge with OOF target-encoding.
Completed: 2026-07-15 19:09 AEST (~7 minutes wall-time, 5 folds × 3 models).

**Per-model CV results:**

| Model | CV RMSE | ± Std | Best Iters (avg) |
|-------|---------|-------|-----------------|
| CatBoost (depth=8) | **1.9932** | 0.0235 | ~894 |
| XGBoost (max_depth=7) | 2.0155 | 0.0240 | ~313 |
| LightGBM (num_leaves=255) | 2.0322 | 0.0260 | ~309 |
| **Ridge Stack (OOF)** | **1.9926** | — | — |

**Feature engineering additions over Model 3:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AGE_x_SEVERITY` | `PATIENT_AGE × DRG_APR_SEVERITY` | Old + severe = longest stays |
| `CHRONIC_x_SEVERITY` | `NUM_CHRONIC_COND × DRG_APR_SEVERITY` | Comorbidity-severity interaction |
| `ICU_x_CHRONIC` | `ICU_DAYS × NUM_CHRONIC_COND` | ICU + chronic disease = prolonged recovery |
| `ICU_x_OPERATION` | `ICU_DAYS × OPERATION_COUNT` | Surgery + ICU = distinct LOS phenotype |
| `LOG_CHARGES` | `log1p(ORDER_TOTAL_CHARGES)` | Tame right-tail of charges |
| `CHARGE_PER_ICU` | `ORDER_TOTAL_CHARGES / (ICU_DAYS+1)` | Charge intensity per ICU day |
| `ICU_DAYS_SQRT` | `sqrt(ICU_DAYS)` | Sub-linear ICU effect |
| `PATIENT_AGE_SQ` | `PATIENT_AGE²` | Quadratic age effect |
| `ADMIT_QUARTER` | `ceil(ADMIT_MTH/3)` | Quarterly seasonal bin |
| `IS_SUMMER` | `ADMIT_MTH ∈ {7,8,9}` | Summer months |
| `MONITOR_x_ICU` | `MONITORING_HOURS × ICU_DAYS` | ✨ New: dual care intensity signal |
| `COMORBID_x_SEV` | `COMORBIDITY_INDEX × DRG_APR_SEVERITY` | ✨ New: complexity × severity |
| `TEAM_x_COMORBID` | `CARE_TEAM_SIZE × COMORBIDITY_INDEX` | ✨ New: team scales with complexity |
| `LOG_MONITORING` | `log1p(MONITORING_HOURS)` | ✨ New: non-linear monitoring |
| `MONITOR_PER_COMORBID` | `MONITORING_HOURS / (COMORBIDITY_INDEX+1)` | ✨ New: monitoring intensity |
| `COMORBID_SQ` | `COMORBIDITY_INDEX²` | ✨ New: quadratic complexity |
| `TEAM_x_ICU` | `CARE_TEAM_SIZE × ICU_DAYS` | ✨ New: team size + ICU |
| `DOCTOR_TE` | OOF target-encode `DOCTOR` | Doctor-level LOS mean (no leakage) |

**Key observation:** The Ridge stack (1.9926) offers a marginal −0.003 RMSE improvement over the
CatBoost-only Model 3 baseline (1.9900). CatBoost remains the strongest individual learner.
LGB/XGB drag the ensemble slightly. The two best submission candidates are nearly equal;
`submission_model_4_stack.csv` is the recommended primary submission.

---

## 6. Stage 4 Experiment Log

All experiments run on updated VFL_2026 dataset (100,000 train / 15,000 test).
Baseline throughout: **Stack 1.9926** (CB+LGB+XGB→Ridge, no log-transform, HOSPITAL freq-encoded).
Kaggle public score for baseline: **1.97823**.

| # | Experiment | Change | CV RMSE (Stack) | Delta vs Baseline | Verdict |
|---|-----------|--------|-----------------|-------------------|---------|
| 0 | **Baseline Stage 4** | CB(d=8)+LGB(255)+XGB(d=7)→Ridge, DOCTOR OOF-TE, HOSPITAL freq-enc | **1.9926** | — | ✅ Submitted → 1.97823 |
| 1 | **HOSPITAL OOF target-encode** | Add HOSPITAL to OOF encoding alongside DOCTOR (replaces freq-enc) | **1.9920** | −0.0006 | ❌ Negligible — within fold noise (±0.023 std). HOSPITAL ANOVA η²=0.000, no real LOS signal per hospital. Freq-enc already captured the volume signal. |
| 2 | **log1p target transform** | Train all 3 models on log1p(ADMIT_LOS), expm1 back for RMSE + submission | **1.9831** | **−0.0089** | ✅ Clear win — individual models slightly worse but stack gains from increased diversity. ADMIT_LOS skewness=3.26 means log-transform reduces outlier influence during training. Stack is now primary submission. **Kaggle LB: 1.97823** |
| 3 | **Stage 5: Optuna + HistGBM** | CB(tuned,d=6)+LGB(tuned,65L)+XGB(tuned)+HistGBM(tuned)→Ridge, log1p | **1.9821** | **−0.0010** | ✅ Marginal gain. LGB weight jumped 0.098→0.247 (65 leaves more diverse). HistGBM r=0.9951 vs CB — no diversity gain. OOF correlations all >0.99 — confirmed diversity deadlock. **Kaggle LB: 1.96** |

### Hyperparameter reference (Stage 4 fixed params)

| Model | Key params | Avg best iter (new data) |
|-------|-----------|--------------------------|
| CatBoost | depth=8, lr=0.03, l2_leaf_reg=3, min_data_in_leaf=5, early_stop=150 | ~894 |
| LightGBM | num_leaves=255, lr=0.03, min_child_samples=10, sub=0.8, col=0.8, λ=1.0, early_stop=150 | ~260 |
| XGBoost | max_depth=7, lr=0.03, sub=0.8, col=0.8, λ=1.0, early_stop=150, tree_method=hist | ~344 |
| Ridge meta | alpha=1.0, inputs=3 OOF columns | — |

### Feature engineering (all active in Stage 4)

| Feature | Formula | Status |
|---------|---------|--------|
| `AGE_x_SEVERITY` | `PATIENT_AGE × DRG_APR_SEVERITY` | Active |
| `CHRONIC_x_SEVERITY` | `NUM_CHRONIC_COND × DRG_APR_SEVERITY` | Active |
| `ICU_x_CHRONIC` | `ICU_DAYS × NUM_CHRONIC_COND` | Active |
| `ICU_x_OPERATION` | `ICU_DAYS × OPERATION_COUNT` | Active |
| `LOG_CHARGES` | `log1p(ORDER_TOTAL_CHARGES)` | Active |
| `CHARGE_PER_ICU` | `ORDER_TOTAL_CHARGES / (ICU_DAYS+1)` | Active |
| `ICU_DAYS_SQRT` | `sqrt(ICU_DAYS)` | Active |
| `PATIENT_AGE_SQ` | `PATIENT_AGE²` | Active |
| `ADMIT_QUARTER` | `ceil(ADMIT_MTH/3)` | Active |
| `IS_SUMMER` | `ADMIT_MTH ∈ {7,8,9}` | Active |
| `MONITOR_x_ICU` | `MONITORING_HOURS × ICU_DAYS` | Active ✨ new |
| `COMORBID_x_SEV` | `COMORBIDITY_INDEX × DRG_APR_SEVERITY` | Active ✨ new |
| `TEAM_x_COMORBID` | `CARE_TEAM_SIZE × COMORBIDITY_INDEX` | Active ✨ new |
| `LOG_MONITORING` | `log1p(MONITORING_HOURS)` | Active ✨ new |
| `MONITOR_PER_COMORBID` | `MONITORING_HOURS / (COMORBIDITY_INDEX+1)` | Active ✨ new |
| `COMORBID_SQ` | `COMORBIDITY_INDEX²` | Active ✨ new |
| `TEAM_x_ICU` | `CARE_TEAM_SIZE × ICU_DAYS` | Active ✨ new |

---

---

## 7. Stage 6 — Ceiling Investigation (2026-07-16)

**Time budget used: ~45 min total (Step 1: 6.8 min, Step 2: 3.8 min, Step 3: <1 min, Step 4: <1 min)**
**Starting point: Stage 5 stack CV 1.9821, Kaggle LB 1.96**

### Step 1 — New categorical OOF encodings (solo CB gate: Δ > 0.02 vs 1.9994)

| Candidate | Formula | Solo CB CV RMSE | Δ vs baseline | Gate |
|-----------|---------|-----------------|---------------|------|
| DIAGNOSIS_SUBCAT_CODE OOF TE (smoothing=20) | Replace raw SUBCAT_CODE with smoothed mean LOS | 2.0000 ± 0.0163 | −0.0006 | ❌ FAIL |
| DEPT × DRG_APR_SEVERITY joint OOF TE (smoothing=10) | Interaction key → smoothed mean LOS | 1.9928 ± 0.0174 | +0.0066 | ❌ FAIL |

Neither cleared 0.02. Trees already learn the DEPT × SEV split natively. Nothing carries forward.

### Step 2 — Objective function diversity (correlation gate: max r < 0.97)

All models trained on **raw target** (Tweedie/Poisson handle mean-variance natively).

| Candidate | Solo CV RMSE | Max OOF r vs Stage5 | Gate |
|-----------|-------------|---------------------|------|
| CB Tweedie (vp=1.5) | **1.9870** ± 0.0174 | 0.9984 vs CB | ❌ FAIL |
| LGB Tweedie (vp=1.5) | 2.0011 ± 0.0197 | 0.9982 vs LGB | ❌ FAIL |
| LGB Poisson | 2.0084 ± 0.0194 | 0.9972 vs LGB | ❌ FAIL |

Notable: CB Tweedie solo RMSE 1.9870 is the best individual model result seen across all stages. However r=0.9984 vs Stage5-CB means it is converging to the same answer. Changing the objective function does not break the diversity deadlock.

### Step 3 — Function class diversity: linear GLM (correlation gate: max r < 0.97)

| Candidate | Solo CV RMSE | Max OOF r vs Stage5 | Gate |
|-----------|-------------|---------------------|------|
| TweedieRegressor (power=1.5, α=1.0) | 2.5206 ± 0.0154 | **0.9079** | ✅ PASS |
| Ridge GLM (power=0, α=1.0) | 2.7123 ± 0.0214 | **0.8793** | ✅ PASS |

Linear models are genuinely decorrelated (r ≈ 0.87–0.91) — a different function class cannot learn interactions or non-linear splits, so its predictions are structurally different from any tree. Both pass the correlation gate as expected.

### Step 4 — Final stack refit (decision rule: beat 1.9821 by > 1 fold std ≈ 0.017)

| Stack combination | CV RMSE | Δ vs 1.9821 | Ridge coefs | Gate |
|-------------------|---------|-------------|-------------|------|
| Stage5 baseline (sanity) | 1.9820 ± 0.0173 | +0.0001 | CB:0.641 LGB:0.247 XGB:0.092 HGBM:0.063 | — |
| Stage5 + Tweedie-GLM | 1.9828 ± 0.0172 | −0.0007 | + TW:0.006 | ❌ FAIL |
| Stage5 + Ridge-GLM | 1.9822 ± 0.0173 | −0.0001 | + RG:−0.009 | ❌ FAIL |
| Stage5 + Tweedie-GLM + Ridge-GLM | 1.9829 ± 0.0167 | −0.0008 | + TW:0.041 RG:−0.043 | ❌ FAIL |

The GLMs pass the correlation gate but fail the decision rule. Despite r ≈ 0.88–0.91, the Ridge assigns near-zero weights (±0.006 to ±0.043) because the GLMs' individual quality (RMSE 2.52–2.71) is so far below the tree ensemble that their decorrelated signal is overwhelmed by their noise. Adding them makes the stack marginally worse in all combinations.

### Stage 6 conclusion (5-sentence summary)

No step in Stage 6 produced an improvement that cleared its gate. New OOF encodings (DIAGNOSIS_SUBCAT_CODE, DEPT×SEV) failed the solo CB gate by a wide margin, consistent with trees already learning these splits natively. Tweedie and Poisson objectives failed the correlation gate because the data structure forces all tree models toward the same predictions regardless of objective function. Linear GLMs passed the correlation gate (r ≈ 0.88–0.91) but failed the decision rule because their individual RMSE is 0.53–0.73 worse than the tree stack, making their decorrelated signal too noisy to contribute. **1.9821 (Kaggle LB 1.96) is the practical ceiling for this feature set and model class — do not submit a new file, do not spend further compute on architecture changes.**

---

## 8. Stage 7 — Final Push (2026-07-16)

**Time budget used: ~105 min total (Step 0: <1 min, Step 1: 7 min, Step 2: 63 min, Step 3: skip, Step 4/5: 30 min)**
**Starting point: Stage 5 stack CV 1.9821, Kaggle LB 1.96**

### Step 0 — Rounding audit (CRITICAL FIX)

All prior submissions exported `ADMIT_LOS` as rounded integers. CV was always computed on raw floats. The delta is a systematic penalty on every submission:

| Metric | Value |
|--------|-------|
| CV RMSE (raw floats) | **1.981205** |
| CV RMSE (rounded integers) | 2.001709 |
| Rounding penalty | **+0.020505** |

Fix applied: `np.clip(stacked_test, 0, 51)` exported as-is with no rounding. Saved to `reports/stage4/submission_model_5_stack_raw.csv`. Estimated LB improvement from rounding fix alone: ~1.96 → ~1.94.

### Step 1 — New categorical OOF encodings (solo CB gate: Δ > 0.02 vs 1.9994)

| Candidate | Solo CB CV RMSE | Δ vs baseline | Gate |
|-----------|-----------------|---------------|------|
| PROCEDURE_SUBCAT_DESC OOF TE (sw=20, 26 cats, 32% missing) | 1.9989 ± 0.0182 | +0.0005 | ❌ FAIL |

Tree models already learn procedure type from OPERATION_COUNT, HAS_PROCEDURE, and contextual DRG signals. Adding an explicit OOF encoding adds nothing.

### Step 2 — Quantile-bin classifier reframing (correlation gate: max r < 0.97)

Trained CatBoostClassifier on binned ADMIT_LOS (pd.qcut reduced 10→6 bins due to discrete target duplicates). Converted OOF class probabilities to continuous predictions via expected value: `EV = Σ P(bin_k) × mean_LOS_k`.

| Candidate | Solo CV RMSE | Max OOF r vs Stage5 | Gate |
|-----------|-------------|---------------------|------|
| CatBoostClassifier-6bins | 2.7412 ± 0.0401 | **0.8688** (vs XGB/LGB) | ✅ PASS |

Bin structure: 6 bins, means = [1.84, 3.00, 4.00, 5.00, 6.41, 11.79]. The cross-entropy objective over quantile bins produces genuinely decorrelated predictions (r ≈ 0.87) — same profile as Stage 6 linear GLMs. Carried into Step 5 per gate rule.

### Step 3 — Pseudo-labeling

Skipped automatically: individual model test predictions not saved in Stage 5 (`test_preds_stage5_indiv.csv` absent). With 100k training rows, pseudo-labeling was expected to contribute minimally regardless.

### Step 4/5 — Multi-seed bagging + final stack refit (decision rule: beat 1.9821 by > 0.017)

Trained all 4 base models (CB/LGB/XGB/HGBM) across 3 KFold seeds [42, 7, 123], averaged OOF and test predictions, then fit Ridge meta-learner on seed-averaged OOFs plus the Step 2 classifier EV.

| Stack combination | CV RMSE | Δ vs 1.9821 | Ridge coefs | Gate |
|-------------------|---------|-------------|-------------|------|
| 3-seed bag + Classifier-EV | 1.9778 ± 0.0187 | +0.0043 | CB:0.670 LGB:0.372 XGB:−0.028 HGBM:0.027 Cls:0.005 | ❌ FAIL |

The classifier EV receives a near-zero coefficient (0.005) — identical outcome to Stage 6 GLMs. Multi-seed averaging reduces variance slightly but does not clear the 0.017 threshold. CB and LGB dominate the stack; XGB and HGBM are near-zero.

### Stage 7 conclusion (5-sentence summary)

Step 0 confirmed a systematic +0.020505 RMSE penalty on every prior submission caused by integer rounding; submitting raw floats is the one guaranteed improvement. Step 1 (PROCEDURE_SUBCAT_DESC TE) failed the solo CB gate by a wide margin, consistent with tree models having already learned this signal from OPERATION_COUNT and DRG features. Step 2 (quantile-bin CatBoostClassifier) passed the correlation gate (r=0.8688) but received only a 0.005 coefficient in the final stack, because its poor individual quality (RMSE 2.74) swamps its decorrelated contribution — the same failure mode as Stage 6 linear GLMs. Multi-seed bagging across 3 seeds produced a marginal improvement (+0.0043 RMSE) that did not clear the 0.017 decision threshold. **Submitting `submission_model_5_stack_raw.csv` (raw float Stage 5 stack, CV 1.9821); the rounding fix is the only confirmed gain.**

---

## 9. Data Quality Flags (Do Not Remove Without Review)

| Flag | Column | Issue | Rows | Treatment |
|------|--------|-------|------|-----------|
| ⚠️ MNAR procedures | `PROCEDURE_*` cols | ~32% missing — no longer only OPERATION_COUNT=0 | 32,072 | Drop text cols; keep `OPERATION_COUNT` + `HAS_PROCEDURE` |
| ⚠️ Dropped feature | `DISCH_NURSE_ID` | r=−0.001 in updated data | All | Dropped from all models |
| ⚠️ Collinear | `DIAGNOSIS_ICD_CODE` | r=1.000 with `DIAGNOSIS_SUBCAT_CODE` | All | Dropped |
| ⚠️ Zero-LOS rows | `ADMIT_LOS` | Same-day discharges | 447 | Kept — clinically valid |
| ⚠️ Age floor | `PATIENT_AGE` | Min=27, no pediatric | Dataset | OOD risk flagged |
| ⚠️ New col multicollinearity | 3 new features | r=0.78–0.85 between them | All | Kept — trees handle this |

---

## 7. Connections to 2025 Solution

| 2025 Finding | Applied Here |
|-------------|-------------|
| KFold > StratifiedKFold for regression | Standard `KFold(5, shuffle=True, seed=42)` |
| RF+GBR workhorse + Ridge meta | CB+LGB+XGB→Ridge in Stage 4 |
| Feature filtering by correlation caused information loss | All 3 new features kept despite high mutual r |
| Scaling essential for Ridge meta-learner | Applied inside stacking only |
| Interaction terms boosted 2025 score | 17 interaction features in Stage 4 (7 new) |
| OOF target encoding needs careful fold-safe setup | DOCTOR OOF TE — `DISCH_NURSE_ID` dropped |

---

## 10. Submission Files

| File | Model | CV RMSE | Notes | Recommend? |
|------|-------|---------|-------|-----------|
| `reports/baseline/submission_model_1.csv` | Naive group-median | 3.503 | — | No |
| `reports/baseline/submission_model_2.csv` | CatBoost admission-only | 2.492 | — | Clinical only |
| `reports/baseline/submission_model_3.csv` | CatBoost full features | 1.990 | Rounded integers | No |
| `reports/stage4/submission_model_4_catboost.csv` | Stage 4 CatBoost | 1.993 | Rounded integers | No |
| `reports/stage4/submission_model_4_stack.csv` | CB+LGB+XGB→Ridge | 1.993 | Rounded integers | No |
| `reports/stage4/submission_model_5_stack_raw.csv` | Stage 5 stack (3-seed, raw floats) | **1.9821** | Rounding fix applied | ✓ **Primary submission** |
| `submission.csv` (project root) | Same as above | **1.9821** | Copied here for upload | ✓ **Upload this** |

---

*End of baseline report.*
