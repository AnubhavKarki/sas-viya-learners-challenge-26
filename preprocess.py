"""Shared data loading, cleaning and feature engineering."""

import numpy as np
import pandas as pd

TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"

DROP_ALWAYS = [
    "ENCOUNTER_KEY",
    "PATIENT_NUMBER",
    "DIAGNOSIS_ICD_CODE",
    "DISCH_NURSE_ID",
]

PROCEDURE_COLS = [
    "PROCEDURE_SUBCAT_CODE",
    "PROCEDURE_SUBCAT_DESC",
    "PROCEDURE_ICD_CODE",
    "PROCEDURE_LONG_DESC",
]

DESC_COLS = [
    "MS_DRG_DESC",
    "DRG_APR_DESC",
    "DIAGNOSIS_SUBCAT_DESC",
    "DIAGNOSIS_LONG_DESC",
]

CAT_COLS = [
    "GENDER",
    "RACE_CD",
    "STATECODE",
    "CITY",
    "COUNTY_NAME",
    "REGION",
    "DEPARTMENT",
    "DIAGNOSIS_GROUP",
    "DX_GROUP",
    "STANDARD_ORDERS_USED",
    "DISCHARGED_TO",
]


def clean(df):
    # severity arrives as a string with stray dots in some exports
    if df["DRG_APR_SEVERITY"].dtype == object:
        df["DRG_APR_SEVERITY"] = (
            df["DRG_APR_SEVERITY"].str.strip().replace(".", np.nan)
            .map({str(i): i for i in range(1, 5)}).astype("float64")
        )
    else:
        df["DRG_APR_SEVERITY"] = df["DRG_APR_SEVERITY"].astype("float64")
    df["NUM_CHRONIC_COND"] = pd.to_numeric(df["NUM_CHRONIC_COND"], errors="coerce")
    df["ORDER_TOTAL_CHARGES"] = df["ORDER_TOTAL_CHARGES"].replace(-2104, np.nan)  # -2104 is a sentinel for missing charges in the raw export
    df["HAS_PROCEDURE"] = (df["OPERATION_COUNT"] > 0).astype(int)
    df["DRG_APR_CODE"] = pd.to_numeric(df["DRG_APR_CODE"], errors="coerce")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def engineer_features(df):
    df = df.copy()
    sev = df["DRG_APR_SEVERITY"].fillna(2)
    cc = df["NUM_CHRONIC_COND"].fillna(0)
    icu = df["ICU_DAYS"]
    # df.get() gracefully handles columns that appear only in one of train/test
    mh = df.get("MONITORING_HOURS", pd.Series(0, index=df.index)).fillna(0)
    ci = df.get("COMORBIDITY_INDEX", pd.Series(0, index=df.index)).fillna(0)
    cts = df.get("CARE_TEAM_SIZE", pd.Series(1, index=df.index)).fillna(1)
    charges = df["ORDER_TOTAL_CHARGES"].fillna(df["ORDER_TOTAL_CHARGES"].median()).clip(lower=1)
    df["AGE_x_SEVERITY"] = df["PATIENT_AGE"] * sev
    df["CHRONIC_x_SEVERITY"] = cc * sev
    df["ICU_x_CHRONIC"] = icu * cc
    df["ICU_x_OPERATION"] = icu * df["OPERATION_COUNT"]
    df["LOG_CHARGES"] = np.log1p(charges)
    df["CHARGE_PER_ICU"] = charges / (icu + 1)
    df["ICU_DAYS_SQRT"] = np.sqrt(icu)
    df["PATIENT_AGE_SQ"] = df["PATIENT_AGE"] ** 2
    df["ADMIT_QUARTER"] = ((df["ADMIT_MTH"] - 1) // 3) + 1
    df["IS_SUMMER"] = df["ADMIT_MTH"].isin([7, 8, 9]).astype(int)
    df["MONITOR_x_ICU"] = mh * icu
    df["COMORBID_x_SEV"] = ci * sev
    df["TEAM_x_COMORBID"] = cts * ci
    df["LOG_MONITORING"] = np.log1p(mh)
    df["MONITOR_PER_COMORBID"] = mh / (ci + 1)
    df["COMORBID_SQ"] = ci ** 2
    df["TEAM_x_ICU"] = cts * icu
    return df


def load_data(pseudo_pred_path=None):
    """Returns X, y, X_test, cat_cols, raw_train, raw_test, y_pseudo, test_keys."""
    train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    test_raw = pd.read_csv(TEST_PATH, low_memory=False)
    y = train_raw["ADMIT_LOS"].values
    test_keys = test_raw["ENCOUNTER_KEY"]

    y_pseudo = None
    if pseudo_pred_path:
        y_pseudo = pd.read_csv(pseudo_pred_path)["ADMIT_LOS"].values.clip(0, 51)

    train = clean(train_raw.copy())
    test = clean(test_raw.copy())
    train = engineer_features(train)
    test = engineer_features(test)

    # HOSPITAL has near-zero direct LOS signal; frequency counts act as a soft size proxy
    hosp_freq = train["HOSPITAL"].value_counts().to_dict()
    med_hosp = float(np.median(list(hosp_freq.values())))  # fallback for hospitals unseen at test time
    for df in [train, test]:
        df["HOSPITAL_freq"] = df["HOSPITAL"].map(hosp_freq).fillna(med_hosp)

    drop_cols = DROP_ALWAYS + PROCEDURE_COLS + DESC_COLS + ["ADMIT_LOS", "HOSPITAL"]
    feature_cols = [c for c in train.columns if c not in drop_cols]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]
    for df in [train, test]:
        for col in cat_cols:
            df[col] = df[col].fillna("__MISSING__").astype(str)

    return (train[feature_cols].copy(), y, test[feature_cols].copy(),
            cat_cols, train_raw, test_raw, y_pseudo, test_keys)
