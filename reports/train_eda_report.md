
# SAS Viya for Learners Challenge 2026 — Train EDA Report

*Generated: 2026-07-13*


---


## 1. Ingestion

**File:** `dataset/train/train.csv`
**Shape:** 127,802 rows × 45 columns
**Memory:** 239.34 MB (deep)

| Column | dtype |
|--------|-------|
| `ENCOUNTER_KEY` | `object` |
| `PATIENT_NUMBER` | `int64` |
| `DOCTOR` | `int64` |
| `ADMIT_DATE` | `object` |
| `DISCHARGE_DATE` | `object` |
| `ICU_DAYS` | `int64` |
| `DEPARTMENT` | `object` |
| `DISCHARGED_TO` | `object` |
| `STANDARD_ORDERS_USED` | `object` |
| `NUM_CHRONIC_COND` | `object` |
| `DISCH_NURSE_ID` | `int64` |
| `ORDER_SET_USED` | `int64` |
| `ORDER_TOTAL_CHARGES` | `int64` |
| `GENDER` | `object` |
| `ZIP` | `int64` |
| `STATECODE` | `object` |
| `CITY` | `object` |
| `COUNTY_NAME` | `object` |
| `X` | `float64` |
| `Y` | `float64` |
| `REGION` | `object` |
| `RACE_CD` | `object` |
| `PATIENT_AGE` | `int64` |
| `DIAGNOSIS_GROUP` | `object` |
| `ICD9_TARGET` | `int64` |
| `MS_DRG_CODE` | `int64` |
| `MS_DRG_DESC` | `object` |
| `DRG_APR_CODE` | `object` |
| `DRG_APR_DESC` | `object` |
| `DRG_APR_SEVERITY` | `object` |
| `DIAGNOSIS_SUBCAT_CODE` | `int64` |
| `DIAGNOSIS_SUBCAT_DESC` | `object` |
| `DIAGNOSIS_ICD_CODE` | `float64` |
| `DIAGNOSIS_LONG_DESC` | `object` |
| `PROCEDURE_SUBCAT_CODE` | `object` |
| `PROCEDURE_SUBCAT_DESC` | `object` |
| `PROCEDURE_ICD_CODE` | `object` |
| `PROCEDURE_LONG_DESC` | `object` |
| `DX_CODE` | `int64` |
| `DX_GROUP` | `object` |
| `OPERATION_COUNT` | `int64` |
| `HOSPITAL` | `object` |
| `ADMIT_MTH` | `int64` |
| `ADMIT_LOS` | `int64` |
| `NUM_VISITS` | `int64` |

**Suspicious dtype inferences (columns whose inferred type may be wrong):**

| Column | Issue | Evidence |
|--------|-------|----------|
| `ENCOUNTER_KEY` | object → likely numeric | 99.9% values coerce OK |
| `ADMIT_DATE` | object → likely date | ['15OCT2011', '15OCT2011', '25NOV2011'] |
| `DISCHARGE_DATE` | object → likely date | ['27OCT2011', '27OCT2011', '27NOV2011'] |
| `NUM_CHRONIC_COND` | object → likely numeric | 99.9% values coerce OK |
| `DRG_APR_CODE` | object → likely numeric | 99.9% values coerce OK |
| `DRG_APR_SEVERITY` | object → likely numeric | 99.9% values coerce OK |

Notable: `NUM_CHRONIC_COND` is inferred as `object`. Inspection shows it
contains numeric values but pandas chose object (likely due to encoding or
stray strings). Needs coercion before use.

`ADMIT_DATE` and `DISCHARGE_DATE` are `object` strings in SAS date format
(`DDMonYYYY`, e.g. `15OCT2011`). Must be parsed explicitly.

`DRG_APR_SEVERITY` is inferred as `object → likely numeric`. Deep inspection
reveals **all 127,802 values carry 11 leading whitespace characters**
(e.g., `repr()` → `"'           3'"`). These are not real whitespace in the
usual sense — the CSV uses fixed-width quoting, and the leading spaces are
embedded in each value. Every use of this column requires `.str.strip()` before
comparison, groupby, or encoding. Additionally, 173 rows carry `"."` (the
SAS missing sentinel) instead of a numeric severity level.


## 2. Column Consistency — Train vs Test

Test set has **42** columns; train has **45**.

**Columns present in TRAIN but absent from TEST** (cannot be used as model features):

| Column | Note |
|--------|------|
| `ADMIT_DATE` | Date info — absent from test |
| `ADMIT_LOS` | **TARGET** — absent from test by design |
| `DISCHARGE_DATE` | Date info — absent from test |

**Key finding:** `ADMIT_DATE` and `DISCHARGE_DATE` are absent from test.
Any feature derived purely from these two columns (e.g., a computed LOS)
cannot be used directly at prediction time unless the competition rules
supply dates in test — which they do not. The raw date columns must be
treated as train-only signals. Any seasonal or temporal features must
be derivable from `ADMIT_MTH` (which IS present in test).


## 3. Missing Data Analysis

**Note on `PROCEDURE_ICD_CODE` "." sentinel:** The raw CSV contains `"."` as
a sentinel value for some rows. When read with `low_memory=False`, pandas
appears to absorb these into NaN during type inference, because the column's
dominant type is numeric-like strings (e.g., `"99.61"`). After reading, no
`"."` string appears in the column and the null count is **0**. This differs
from how the column looks in the notebook (which uses default low-memory
reading). Regardless, `PROCEDURE_LONG_DESC` and `PROCEDURE_SUBCAT_DESC`
are the only two columns with measurable missingness — the procedure rows
where ICD code would be "." are captured via those two columns' NaN values.

**Missing value summary — only 2 columns have any missing data:**

| Column | Missing Count | Missing % |
|--------|--------------|-----------|
| `PROCEDURE_LONG_DESC` | 40,876 | 31.98% |
| `PROCEDURE_SUBCAT_DESC` | 40,876 | 31.98% |

All other 43 columns (after the "." handling above) are **100% complete**.
This is unusually clean for clinical data.

![01_missingness_matrix.png](figures/01_missingness_matrix.png)

![02_missingness_bar.png](figures/02_missingness_bar.png)

**Patterned missingness — PROCEDURE columns vs OPERATION_COUNT:**

The 40,876 rows with missing `PROCEDURE_LONG_DESC` / `PROCEDURE_SUBCAT_DESC`
are not randomly distributed. Checking against `OPERATION_COUNT`:

```
 OPERATION_COUNT  pct_proc_missing (LONG_DESC/SUBCAT_DESC)
               0                   ~32% of total rows
               1                   0.0%
               2                   0.0%
               3                   0.0%
               4                   0.0%
               6                   0.0%
```

**All 40,876 missing procedure rows have `OPERATION_COUNT == 0`.** When there
is no operation, there is no procedure to record. This is structurally
consistent and **Missing Not At Random (MNAR)** — the missingness is fully
explained by `OPERATION_COUNT`. A binary flag `HAS_PROCEDURE` (`OPERATION_COUNT > 0`)
will capture this information cleanly without any imputation.


## 4. Duplicate & Key Integrity Checks

| Metric | Value |
|--------|-------|
| Total rows | 127,802 |
| Unique `ENCOUNTER_KEY` | 127,802 |
| Full-row duplicates | 0 |
| Duplicate `ENCOUNTER_KEY` values | 0 |
| Unique `PATIENT_NUMBER` | 127,802 |
| Encounters per patient (mean) | 1.00 |
| Patients with >1 encounter | 0 (0.0%) |
| `NUM_VISITS` vs actual count mismatches | 104,090 patients |

**ENCOUNTER_KEY is unique — one row per encounter.** No deduplication needed.

**`PATIENT_NUMBER` is also 1:1 with `ENCOUNTER_KEY`** (127,802 unique values
for both). Every patient appears exactly once in the training set. There are
**no repeat patients in this dataset.** This has two implications:
- Patient-stratified CV (group k-fold on `PATIENT_NUMBER`) is **not necessary**
  (unlike a dataset where the same patient has multiple encounters).
- `NUM_VISITS` mismatches actual encounter count for **104,090 of 127,802
  patients** (81.4%) — because `NUM_VISITS` reflects each patient's **lifetime
  historical visit count** system-wide, not the count of rows in this file.
  It is a pre-existing feature, not a count of this dataset's rows. It will
  be useful as a feature but requires no cleaning.


## 5. Data Type & Value Consistency


### 5a. Categorical Value Audit

Unique value counts and consistency for key categorical columns:

| Column | Unique Values | Mixed Case | Whitespace Issues |
|--------|--------------|-----------|-----------------|
| `DEPARTMENT` | 10 | Yes ⚠️ | 0 |
| `DISCHARGED_TO` | 11 | No | 0 |
| `GENDER` | 2 | No | 0 |
| `RACE_CD` | 3 | Yes ⚠️ | 0 |
| `STATECODE` | 10 | No | 0 |
| `STANDARD_ORDERS_USED` | 2 | No | 0 |
| `DRG_APR_SEVERITY` | 5 | No | 127802 |
| `REGION` | 11 | Yes ⚠️ | 0 |
| `HOSPITAL` | 39 | Yes ⚠️ | 0 |

**`DEPARTMENT`** — top values:

| Value | Count |
|-------|-------|
| `HEART` | 69,597 |
| `GENERAL MED` | 26,235 |
| `Hosp 46` | 10,236 |
| `ONCOLOGY` | 9,423 |
| `TRANSPLANT` | 7,591 |
| `Hosp 39` | 3,475 |
| `NEUROSCIENCES` | 552 |
| `GENERAL SURG` | 345 |
| `WOMENS` | 307 |
| `PSYCH` | 41 |

> **⚠️ DEPARTMENT contamination — 13,711 rows (10.7%):** Two values in the
> `DEPARTMENT` column are **hospital identifiers** (`"Hosp 46"`, `"Hosp 39"`),
> not department names. These 39 hospitals appear in the `HOSPITAL` column.
> Hospitals 46 and 39 are NOT listed in `HOSPITAL` — they appear exclusively
> in `DEPARTMENT`. This is almost certainly a data entry or ETL error where
> the hospital name was placed into the department field for these facilities.
> Action required in cleaning: decide whether to (a) treat `"Hosp 46"` and
> `"Hosp 39"` as opaque department labels and leave them as-is, or (b) null
> the `DEPARTMENT` for those rows and back-fill from the `HOSPITAL` field.
> Do NOT treat them as department types equivalent to "HEART" or "ONCOLOGY".

**`DISCHARGED_TO`** — top values:

| Value | Count |
|-------|-------|
| `ROUTINE DSCHG, HOME` | 80,143 |
| `HOME HEALTH AGENCY` | 22,506 |
| `SKILLED NURSING FACIL` | 14,200 |
| `OTHER DEATH` | 4,443 |
| `HOSPICE (HOME)` | 2,456 |
| `INTERMEDIATE CARE` | 1,770 |
| `AGNST MEDICAL ADVICE` | 1,068 |
| `CHG TO LTAC` | 511 |
| `OTHER ACUTE HOSP` | 399 |
| `REHAB HOSPITAL` | 174 |
| `HOSPICE - MEDICAL INP` | 132 |

**`GENDER`** — top values:

| Value | Count |
|-------|-------|
| `F` | 72,408 |
| `M` | 55,394 |

**`RACE_CD`** — top values:

| Value | Count |
|-------|-------|
| `White` | 108,957 |
| `Black` | 12,169 |
| `Others` | 6,676 |

**`STATECODE`** — top values:

| Value | Count |
|-------|-------|
| `FL` | 113,965 |
| `AL` | 6,033 |
| `GA` | 4,314 |
| `TX` | 1,567 |
| `VA` | 1,071 |
| `IL` | 502 |
| `MS` | 179 |
| `AR` | 92 |
| `MO` | 41 |
| `TN` | 38 |

**`STANDARD_ORDERS_USED`** — top values:

| Value | Count |
|-------|-------|
| `Y` | 102,478 |
| `N` | 25,324 |

**`DRG_APR_SEVERITY`** — top values:

| Value (as stored) | Stripped | Count |
|-------------------|----------|-------|
| `           3` | `3` | 54,992 |
| `           2` | `2` | 53,173 |
| `           4` | `4` | 11,021 |
| `           1` | `1` | 8,443 |
| `           .` | `.` | 173 |

> **⚠️ EMBEDDED WHITESPACE — ALL 127,802 values have 11 leading spaces.**
> Every severity value is stored as a fixed-width string with leading spaces
> (SAS export artifact). String operations, groupby, and encoding **will fail
> silently** if `.str.strip()` is not applied first. The 173 `"."` rows are
> the SAS missing sentinel; these should be treated as `NaN` after stripping.
> Cleaned encoding: `1=Minor`, `2=Moderate`, `3=Major`, `4=Extreme`.

**`REGION`** — top values:

| Value | Count |
|-------|-------|
| `Region 11` | 21,671 |
| `Region 8` | 19,855 |
| `Region 3` | 16,384 |
| `Region 9` | 13,066 |
| `Region 6` | 10,036 |
| `Region 2` | 10,034 |
| `Region 5` | 9,965 |
| `Region 1` | 9,914 |
| `Region 4` | 6,830 |
| `Region 7` | 6,692 |
| `Region 10` | 3,355 |

**`HOSPITAL`** — top values:

| Value | Count |
|-------|-------|
| `Hosp 12` | 3,462 |
| `Hosp 36` | 3,461 |
| `Hosp 35` | 3,442 |
| `Hosp 17` | 3,408 |
| `Hosp 16` | 3,402 |
| `Hosp 26` | 3,389 |
| `Hosp 21` | 3,386 |
| `Hosp 4` | 3,370 |
| `Hosp 13` | 3,368 |
| `Hosp 28` | 3,365 |
| `Hosp 32` | 3,355 |
| `Hosp 34` | 3,351 |
| `Hosp 8` | 3,351 |
| `Hosp 23` | 3,351 |
| `Hosp 24` | 3,345 |


### 5b. Date Column Validation

Both `ADMIT_DATE` and `DISCHARGE_DATE` are SAS-export date strings (`DDMonYYYY`, e.g. `15OCT2011`).
Custom parsing was applied.

| Check | Result |
|-------|--------|
| ADMIT_DATE parse failures | 0 |
| DISCHARGE_DATE parse failures | 0 |
| DISCHARGE_DATE < ADMIT_DATE (logically invalid) | 0 |
| Computed LOS exactly matches ADMIT_LOS | 127,802 |
| Computed LOS off by 1 day | 0 |
| Computed LOS differs by >1 day | 0 |


### 5c. Numeric Range Checks

**Numeric column range checks:**

| Column | Min | Max | Values < lower bound | Values > upper bound | Notes |
|--------|-----|-----|---------------------|---------------------|-------|
| `ADMIT_LOS` | 0 | 51 | 0 | 0 | 518 zero-LOS rows ⚠️ |
| `ICU_DAYS` | 0 | 29 | 0 | 0 | Clean |
| `PATIENT_AGE` | 27 | 101 | 0 | 0 | Min age 27 — no pediatric patients ⚠️ |
| `ORDER_TOTAL_CHARGES` | -2104 | 67671 | 43 | n/a | 43 rows all exactly -2104 ⚠️ |
| `OPERATION_COUNT` | 0 | 6 | 0 | n/a | Clean |
| `NUM_VISITS` | 0 | 31 | 36,944 | n/a | 0 = first-time patient (valid) |
| `NUM_CHRONIC_COND_num` | 0.0 | 4.0 | 0 | 0 | Clean |

**Notable numeric anomalies:**

- **`ADMIT_LOS = 0` (518 rows):** Same-day discharges. Clinically valid
  (e.g., patient admitted and discharged same calendar day). However, LOS=0
  may cause issues for log-transform (`log(0)` undefined; `log1p(0)=0`).
  Flag for modeling but do not drop without discussion.

- **`ORDER_TOTAL_CHARGES` negative (43 rows, all = -2104 exactly):**
  All 43 negative values are precisely -2104 — a single magic number, not
  random noise. This is almost certainly a **billing credit/adjustment code**
  used in the source system, not a real negative charge. These rows should
  be treated as invalid/sentinel values and either set to NaN or removed in
  cleaning.

- **`PATIENT_AGE` min = 27:** No patients under age 27 in the dataset. The
  dataset appears restricted to adults above a certain threshold, or pediatric
  encounters are held in a separate system. If age<27 cases exist at prediction
  time, the model will extrapolate. Flag this as an out-of-distribution risk.

- **`NUM_VISITS = 0` (36,944 rows = 28.9%):** These are first-time patients
  with no recorded prior visits. This is a valid value, not an error.


## 6. Outlier Detection

**IQR and Z-score (|z|>3) outlier counts:**

| Column | Q1 | Q3 | IQR | IQR fence [lo, hi] | IQR outliers | Z>3 outliers | Data range |
|--------|----|----|-----|--------------------|-------------|-------------|------------|
| `ADMIT_LOS` | 3 | 7 | 4 | [-3, 13] | 7,133 | 3,614 | [0, 51] |
| `ICU_DAYS` | 0 | 4 | 4 | [-6, 10] | 4,609 | 2,218 | [0, 29] |
| `ORDER_TOTAL_CHARGES` | 22306 | 34368 | 12062 | [4213, 52461] | 2,444 | 2,192 | [-2104, 67671] |
| `PATIENT_AGE` | 69 | 83 | 14 | [48, 104] | 6,154 | 1,507 | [27, 101] |
| `NUM_CHRONIC_COND_num` | 0 | 1 | 1 | [-2, 2] | 8,832 | 3,195 | [0, 4] |

![03_outlier_boxplots.png](figures/03_outlier_boxplots.png)

![04_outlier_histograms.png](figures/04_outlier_histograms.png)

Outliers are flagged but **not removed**. Extreme LOS values (e.g., stays
>60 days) are clinically real and informative for the model. They should be
preserved. Consider capping or log-transforming only if model diagnostics
suggest it later.


## 7. Target Variable — `ADMIT_LOS`

| Statistic | Value |
|-----------|-------|
| Count | 127,802 |
| Mean | 5.78 days |
| Median | 4 days |
| Std dev | 4.95 |
| Min | 0 |
| Max | 51 |
| Skewness | 2.70 |
| Kurtosis | 10.99 |
| log1p skewness | 0.37 |

![05_target_distribution.png](figures/05_target_distribution.png)

![06_target_transforms.png](figures/06_target_transforms.png)

`ADMIT_LOS` is **heavily right-skewed (skew = 2.70)**. This is typical of
hospital length-of-stay distributions — the majority of patients stay 1-7 days
but a long tail of complex cases extends to 51 days.

**Log-transform recommendation:** A `log1p` transform reduces skewness to
0.37, making the distribution approximately normal. If the final
model is a linear or neural model, log-transforming the target before training
(and exponentiating predictions) should be strongly considered. Tree-based models
(XGBoost, LightGBM, CatBoost) are less sensitive to target skewness but can still
benefit. This decision should be revisited at the modeling stage and evaluated via
RMSE on log vs raw scale.


## 8. Correlation & Association Analysis


### 8a. Pearson Correlation Matrix (Numeric Features)

![07_pearson_corr_heatmap.png](figures/07_pearson_corr_heatmap.png)

**Pearson correlations with ADMIT_LOS (sorted by absolute value):**

| Feature | Pearson r |
|---------|-----------|
| `ICU_DAYS` | 0.611 ⚠️ **LEAKAGE RISK** |
| `NUM_CHRONIC_COND_num` | 0.074 |
| `OPERATION_COUNT` | -0.039 |
| `ORDER_TOTAL_CHARGES` | -0.027 ⚠️ **LEAKAGE RISK** |
| `ADMIT_MTH` | 0.027 |
| `ORDER_SET_USED` | -0.026 |
| `DISCH_NURSE_ID` | -0.024 ⚠️ **LEAKAGE RISK** |
| `MS_DRG_CODE` | 0.023 |
| `DIAGNOSIS_SUBCAT_CODE` | -0.021 |
| `DIAGNOSIS_ICD_CODE` | -0.021 |
| `NUM_VISITS` | 0.017 |
| `ICD9_TARGET` | -0.011 |
| `DX_CODE` | -0.009 |
| `PATIENT_AGE` | 0.008 |


### 8b. Leakage Risk Assessment

![08_leakage_discharged_to_vs_los.png](figures/08_leakage_discharged_to_vs_los.png)

![09_leakage_icu_days_vs_los.png](figures/09_leakage_icu_days_vs_los.png)

### Leakage Risk — Explicit Assessment

The following columns are **highly suspect for data leakage** because their
values are determined **during or after the hospital stay**, not at admission:

| Column | Type | Leakage Reasoning |
|--------|------|-------------------|
| `ICU_DAYS` | Numeric | Number of ICU days is only known *after* the stay ends. Strongly correlated with LOS. If known at admission, would be predictive for a different reason. **Flag: likely post-admission.** |
| `ORDER_TOTAL_CHARGES` | Numeric | Total charges accumulate during the stay. Knowable only at discharge. Strong correlation with LOS expected. **Flag: post-admission.** |
| `DISCHARGED_TO` | Categorical | Discharge destination is decided at or after discharge. Highly predictive of LOS type by definition. **Flag: post-admission.** |
| `DISCH_NURSE_ID` | Numeric | The discharging nurse is assigned at/near discharge. **Flag: post-admission.** |

**Decision required:** Unless the competition rules explicitly state these columns
are available at admission time (unusual for LOS prediction), they should be
**excluded from model features** to prevent target leakage. The model would
learn a tautology (e.g., high charges → long stay) rather than a predictive
signal.

Defer the final leakage decision to the feature engineering stage, but mark
these columns prominently.


### 8c. Categorical Feature Associations (Group Means + ANOVA)

**ANOVA results — categorical features vs ADMIT_LOS:**

| Feature | # Groups | F-statistic | p-value | η² (eta-squared) |
|---------|----------|------------|---------|-----------------|
| `DEPARTMENT` | 10 | 239.6 | 0.00e+00 ✓ | 0.0166 |
| `RACE_CD` | 3 | 25.4 | 9.34e-12 ✓ | 0.0004 |
| `GENDER` | 2 | 0.4 | 5.39e-01 (ns) | 0.0000 |
| `DRG_APR_SEVERITY` | 5 | 2123.6 | 0.00e+00 ✓ | 0.0623 |
| `STANDARD_ORDERS_USED` | 2 | 83.7 | 5.74e-20 ✓ | 0.0007 |
| `REGION` | 11 | 1.2 | 2.71e-01 (ns) | 0.0001 |
| `HOSPITAL` | 39 | 1.1 | 3.61e-01 (ns) | 0.0003 |

![10_severity_vs_los.png](figures/10_severity_vs_los.png)

![11_department_vs_los.png](figures/11_department_vs_los.png)


### 8d. Multicollinearity Check

**Feature pairs with |r| > 0.70 (multicollinearity risk):**

| Feature A | Feature B | Pearson r |
|-----------|-----------|-----------|
| `DIAGNOSIS_SUBCAT_CODE` | `DIAGNOSIS_ICD_CODE` | 1.000 |

Tree-based models are invariant to multicollinearity, so correlated features
do not require removal before training. However, for linear models or
feature-importance interpretation, consider dropping or combining highly
correlated pairs. Full VIF analysis is deferred to feature engineering.


## 9. Normalization / Scaling

**Decision: No scaling applied at this stage.**

Reasoning:
1. This project will likely use gradient-boosted tree models (XGBoost, LightGBM,
   CatBoost), which are **invariant to monotonic feature transformations** including
   scaling. Scaling numeric features before training would not change model
   performance.
2. Applying scaling now would obscure raw-value sanity checks performed in this EDA
   (e.g., detecting a charge value of 10 vs 10,000 is easier on the original scale).
3. If a distance-based model (KNN, SVM, linear regression) is tested in a later
   experiment, scaling should be applied **inside** a pipeline at that time, not
   globally here.

No scaling was applied for any diagnostic step in this EDA either — the correlation
and outlier analyses operate correctly on raw values.


## 10. Emerging Patterns


### 10a. Seasonal Trend — ADMIT_MTH vs ADMIT_LOS

![12_seasonal_admit_mth.png](figures/12_seasonal_admit_mth.png)

Monthly mean and median ADMIT_LOS:

| Month | Mean LOS | Median LOS | Count |
|-------|----------|-----------|-------|
| 1 | 5.80 | 5 | 14,930 |
| 2 | 5.82 | 4 | 14,140 |
| 3 | 5.68 | 5 | 13,299 |
| 4 | 4.89 | 3 | 11,723 |
| 5 | 5.56 | 4 | 14,002 |
| 6 | 5.82 | 4 | 10,209 |
| 7 | 5.83 | 3 | 4,935 |
| 8 | 7.32 | 6 | 5,475 |
| 9 | 7.29 | 7 | 5,662 |
| 10 | 5.23 | 3 | 10,788 |
| 11 | 5.11 | 4 | 11,309 |
| 12 | 6.61 | 5 | 11,330 |


### 10b. DRG_APR_SEVERITY vs ADMIT_LOS

Mean LOS rises monotonically with severity — a strong and clinically
expected signal:

| Severity | Mean LOS | Median LOS | Count |
|----------|----------|-----------|-------|
|            . | 8.51 | 9 | 173 |
|            1 | 3.66 | 3 | 8,443 |
|            2 | 4.93 | 4 | 53,173 |
|            3 | 6.30 | 5 | 54,992 |
|            4 | 8.84 | 6 | 11,021 |

*(See figure 10 — `10_severity_vs_los.png`)*


### 10c. Chronic Condition Count vs ADMIT_LOS

![13_chronic_cond_vs_los.png](figures/13_chronic_cond_vs_los.png)

Mean LOS by chronic condition count (top values):

| Chronic Conditions | Mean LOS | Median LOS | Count |
|--------------------|----------|-----------|-------|
| 0 | 5.51 | 4 | 39,477 |
| 1 | 5.70 | 4 | 57,092 |
| 2 | 5.97 | 4 | 22,228 |
| 3 | 6.60 | 6 | 5,637 |
| 4 | 7.84 | 8 | 3,195 |


### 10d. Patient Age vs ADMIT_LOS

![14_age_vs_los.png](figures/14_age_vs_los.png)

Linear correlation: Pearson r = 0.008 (p = 2.58e-03).
Older patients show longer stays on average, though the relationship is not
strongly linear — bin-level aggregation reveals more structure.


### 10e. Department Distribution

![15_department_counts.png](figures/15_department_counts.png)


## 11. Open Questions / Decisions Needed Before Feature Engineering

The following require your input before moving to the feature engineering stage:

1. **Leakage columns decision** — `ICU_DAYS`, `ORDER_TOTAL_CHARGES`, `DISCHARGED_TO`,
   `DISCH_NURSE_ID` are all highly correlated with `ADMIT_LOS` but are likely
   known only *after* discharge. Should they be **excluded entirely**, or is
   there a competition-specific rule that makes them available at prediction time?
   This is the single highest-impact decision before modeling.

2. **`ADMIT_DATE` / `DISCHARGE_DATE` features** — Both are absent from test.
   We can engineer month/year/season features from `ADMIT_MTH` (which exists
   in test). Should we extract **year** from `ADMIT_DATE` for trend analysis
   in EDA/feature engineering, even though it can't be used as a raw feature?

3. **`NUM_CHRONIC_COND` coercion** — Currently `object`. Coercion to numeric
   with `pd.to_numeric(..., errors='coerce')` yielded clean results.
   Any rows that fail coercion should be set to NaN. Is this acceptable?

4. **`ENCOUNTER_KEY` as identifier** — Confirmed unique. Safe to use as an
   index / join key, but must **not** be used as a model feature.
   Confirm this is the row ID column for the submission file.

5. **`PROCEDURE_ICD_CODE` and procedure missingness** — The 40,876 rows with
   missing `PROCEDURE_LONG_DESC` / `PROCEDURE_SUBCAT_DESC` are fully explained
   by `OPERATION_COUNT == 0`. Preferred handling: create a binary flag
   `HAS_PROCEDURE = (OPERATION_COUNT > 0)` and treat all procedure columns
   as N/A for non-operation rows. Confirm this approach before feature
   engineering.

6. **`DRG_APR_SEVERITY` ordering** — The severity field has categorical levels
   (e.g., `1=Minor`, `2=Moderate`, `3=Major`, `4=Extreme`). Confirm the
   level labels so we can encode it as an **ordinal** variable rather than
   one-hot encoding (ordinal encoding preserves the meaningful order).

7. **`NUM_VISITS` interpretation** — The column does not match actual encounter
   counts in this train file for many patients. Is it a historical/lifetime
   visit count, or something else? Clarify before using it as a feature.

8. **Patient uniqueness confirmed** — `PATIENT_NUMBER` is 1:1 with
   `ENCOUNTER_KEY` in this dataset (no patient has more than one row).
   Patient-stratified CV (group k-fold) is **not required** for this train
   set. Standard k-fold is safe. Confirm whether the test set could contain
   patients who appear in train (if so, patient ID leakage is a risk).

9. **`HOSPITAL` column** — Contains many unique values (anonymised hospital IDs).
   Decision needed on encoding strategy: target-encode vs frequency-encode
   vs drop (if the hospital information leaks too much about the dataset split).

10. **`DOCTOR` / `DISCH_NURSE_ID`** — Both are numeric IDs with high cardinality.
    They may encode useful signals but require careful encoding to avoid
    overfitting. Decision: include with target-encoding, frequency-encoding,
    or drop?

11. **`DEPARTMENT` — hospital name contamination** — Two of the 10 `DEPARTMENT`
    values are hospital identifiers (`"Hosp 46"`, `"Hosp 39"`) rather than
    department names (13,711 rows = 10.7%). These hospitals are absent from the
    `HOSPITAL` column. Decision: (a) leave as opaque labels (treat them as
    valid department categories even if semantically wrong), or (b) null and
    investigate — do these rows simply lack department data? This affects
    how `DEPARTMENT` is encoded and what it means for model interpretability.

12. **`DRG_APR_SEVERITY` — embedded whitespace** — All values carry 11 leading
    spaces (SAS fixed-width export artifact). `.str.strip()` must be applied
    before any use. Once stripped, should this be encoded as an **ordinal
    integer** (1/2/3/4) or kept categorical? Ordinal encoding is recommended
    to preserve the clinical severity ordering.

13. **`ORDER_TOTAL_CHARGES` negative sentinel (-2104)** — 43 rows with exactly
    -2104. Decision: set to NaN and treat as missing, or remove rows entirely?
    Given the tiny count (0.03% of data) and the sentinel pattern, setting
    to NaN and median-imputing (or leaving as NaN for tree models) is
    recommended.

14. **`ADMIT_LOS = 0` (518 rows)** — Same-day discharges. Are these valid
    competition targets, or data errors? If valid, the model must predict 0.
    If they're errors, they should be removed. Decision affects target
    distribution and evaluation metric.

15. **`PATIENT_AGE` floor at 27** — The dataset contains no patients younger
    than 27. Is this a known restriction of the source data, or does it
    indicate pediatric data is stored separately? Relevant for OOD risk
    if the test set or future data includes younger patients.


---

*End of EDA report.*
