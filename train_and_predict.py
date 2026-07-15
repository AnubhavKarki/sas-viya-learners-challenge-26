"""
SAS Viya for Learners Challenge 2026
Predict ADMIT_LOS (hospital length of stay in days)

Approach: CatBoost + LightGBM + XGBoost ensemble with Ridge meta-learner.
          OOF target-encoding for high-cardinality ID columns (DOCTOR, DISCH_NURSE_ID).
          5-fold KFold CV throughout for consistent, leak-free evaluation.

Usage:
    python train_and_predict.py
    python train_and_predict.py --train data/train.csv --test data/test.csv --out submission.csv
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

# ─── Configuration ────────────────────────────────────────────────────────────

SEED    = 42
N_FOLDS = 5

# Early stopping patience (rounds without improvement before halting)
EARLY_STOP = 150

# CatBoost: depth=8 over baseline depth=6 to capture complex interactions
CB_PARAMS = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    min_data_in_leaf=5,
    random_seed=SEED,
    eval_metric="RMSE",
    verbose=0,
    allow_writing_files=False,
)

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    num_leaves=255,
    learning_rate=0.03,
    n_estimators=5000,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    tree_method="hist",
    verbosity=0,
    early_stopping_rounds=EARLY_STOP,
    eval_metric="rmse",
)

# Columns to drop: identifiers, train-only date columns, perfectly collinear column,
# procedure text columns (captured by OPERATION_COUNT / HAS_PROCEDURE),
# and description columns whose codes are already included
DROP_COLS = [
    # Identifiers
    "ENCOUNTER_KEY", "PATIENT_NUMBER",
    # Absent from test.csv
    "ADMIT_DATE", "DISCHARGE_DATE",
    # Perfectly collinear with DIAGNOSIS_SUBCAT_CODE (r=1.000)
    "DIAGNOSIS_ICD_CODE",
    # Procedure text — captured by OPERATION_COUNT and HAS_PROCEDURE flag
    "PROCEDURE_SUBCAT_CODE", "PROCEDURE_SUBCAT_DESC",
    "PROCEDURE_ICD_CODE", "PROCEDURE_LONG_DESC",
    # Description columns — code equivalents already included
    "MS_DRG_DESC", "DRG_APR_DESC",
    "DIAGNOSIS_SUBCAT_DESC", "DIAGNOSIS_LONG_DESC",
    # Frequency-encoded separately below
    "HOSPITAL",
    # Target
    "ADMIT_LOS",
]

# Categorical columns passed to CatBoost's native encoder
CAT_COLS = [
    "GENDER", "RACE_CD", "STATECODE", "CITY", "COUNTY_NAME",
    "REGION", "DEPARTMENT", "DIAGNOSIS_GROUP", "DX_GROUP",
    "STANDARD_ORDERS_USED", "DISCHARGED_TO",
]

# Columns that receive OOF target-encoding (high-cardinality ID columns)
OOF_ENCODE_COLS = ["DOCTOR", "DISCH_NURSE_ID"]


# ─── Preprocessing ───────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all data-quality fixes identified during EDA."""

    # DRG_APR_SEVERITY: SAS export embeds 11 leading spaces in every value;
    # "." is the SAS missing sentinel. Encode as ordinal int 1–4.
    severity_map = {"1": 1, "2": 2, "3": 3, "4": 4}
    df["DRG_APR_SEVERITY"] = (
        df["DRG_APR_SEVERITY"].str.strip()
        .replace(".", np.nan)
        .map(severity_map)
        .astype("float64")
    )

    # NUM_CHRONIC_COND stored as object due to SAS whitespace — coerce to numeric
    df["NUM_CHRONIC_COND"] = pd.to_numeric(df["NUM_CHRONIC_COND"], errors="coerce")

    # ORDER_TOTAL_CHARGES: 43 rows equal exactly -2104 (billing credit sentinel, not real)
    df["ORDER_TOTAL_CHARGES"] = df["ORDER_TOTAL_CHARGES"].replace(-2104, np.nan)

    # DRG_APR_CODE inferred as object but is numeric
    df["DRG_APR_CODE"] = pd.to_numeric(df["DRG_APR_CODE"], errors="coerce")

    # Binary procedure flag — missingness in procedure columns is fully explained
    # by OPERATION_COUNT == 0 (Missing Not At Random)
    df["HAS_PROCEDURE"] = (df["OPERATION_COUNT"] > 0).astype(int)

    # Strip residual whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinical interaction features and derived signals."""
    sev = df["DRG_APR_SEVERITY"].fillna(2)
    cc  = df["NUM_CHRONIC_COND"].fillna(0)

    # Severity × age: older + sicker patients have disproportionately long stays
    df["AGE_x_SEVERITY"]     = df["PATIENT_AGE"] * sev
    # Comorbidity × severity: combined clinical complexity
    df["CHRONIC_x_SEVERITY"] = cc * sev
    # ICU days × chronic conditions
    df["ICU_x_CHRONIC"]      = df["ICU_DAYS"] * cc
    # Surgery + ICU: high-complexity phenotype
    df["ICU_x_OPERATION"]    = df["ICU_DAYS"] * df["OPERATION_COUNT"]

    # Charge-based features
    charges = df["ORDER_TOTAL_CHARGES"].fillna(
        df["ORDER_TOTAL_CHARGES"].median()
    ).clip(lower=1)
    df["LOG_CHARGES"]        = np.log1p(charges)
    # Charge intensity relative to ICU component
    df["CHARGE_PER_ICU"]     = charges / (df["ICU_DAYS"] + 1)

    # Non-linear transforms
    df["ICU_DAYS_SQRT"]      = np.sqrt(df["ICU_DAYS"])
    df["PATIENT_AGE_SQ"]     = df["PATIENT_AGE"] ** 2

    # Seasonal features (EDA: months 7–9 average ~1.5 days longer than baseline)
    df["ADMIT_QUARTER"]      = ((df["ADMIT_MTH"] - 1) // 3) + 1
    df["IS_SUMMER"]          = df["ADMIT_MTH"].isin([7, 8, 9]).astype(int)

    return df


# ─── OOF target encoding ─────────────────────────────────────────────────────

def oof_target_encode(
    s_tr: pd.Series, y_tr: np.ndarray,
    s_vl: pd.Series,
    global_mean: float,
    smoothing: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smoothed mean target encoding fitted on the training fold only.
    Unseen values in val/test fall back to the global mean.
    smoothing=20 prevents small groups from dominating.
    """
    idx    = pd.Series(y_tr, index=s_tr.index)
    stats  = idx.groupby(s_tr).agg(["mean", "count"])
    smooth = (
        (stats["mean"] * stats["count"] + global_mean * smoothing) /
        (stats["count"] + smoothing)
    )
    enc_tr = s_tr.map(smooth).fillna(global_mean).values
    enc_vl = s_vl.map(smooth).fillna(global_mean).values
    return enc_tr, enc_vl


def apply_oof_encoding(X_tr, X_vl, X_te, y_tr, cols=OOF_ENCODE_COLS):
    """Apply OOF target encoding to all specified columns in one fold."""
    X_tr = X_tr.copy()
    X_vl = X_vl.copy()
    X_te = X_te.copy()
    gm   = float(np.mean(y_tr))

    for col in cols:
        if col not in X_tr.columns:
            continue
        enc_tr, enc_vl = oof_target_encode(X_tr[col], y_tr, X_vl[col], gm)

        # For test: use full fold-train statistics
        stats  = pd.Series(y_tr, index=X_tr.index).groupby(X_tr[col]).agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + gm * 20) / (stats["count"] + 20)
        enc_te = X_te[col].map(smooth).fillna(gm).values

        X_tr[col] = enc_tr
        X_vl[col] = enc_vl
        X_te[col] = enc_te

    return X_tr, X_vl, X_te


# ─── Data loading ────────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str):
    # utf-8-sig strips the BOM present in the SAS-exported CSV header
    train = pd.read_csv(train_path, low_memory=False, encoding="utf-8-sig")
    test  = pd.read_csv(test_path,  low_memory=False, encoding="utf-8-sig")

    # ENCOUNTER_KEY may carry embedded BOM characters in some rows
    encounter_keys = (
        test["ENCOUNTER_KEY"].astype(str)
        .str.replace("﻿", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
        .astype("int32")
    )

    y = train["ADMIT_LOS"].values

    train = clean(train)
    test  = clean(test)
    train = engineer(train)
    test  = engineer(test)

    # Frequency-encode HOSPITAL: count of occurrences is a proxy for hospital
    # volume/specialisation. Count-based (not target-based), no leakage risk.
    hosp_freq = train["HOSPITAL"].value_counts().to_dict()
    med_freq  = float(np.median(list(hosp_freq.values())))
    for df in [train, test]:
        df["HOSPITAL_freq"] = df["HOSPITAL"].map(hosp_freq).fillna(med_freq)

    # Build feature matrices
    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    X_train = train[feature_cols].copy()
    X_test  = test[feature_cols].copy()

    # Fill NaN in CatBoost categorical columns
    active_cat = [c for c in CAT_COLS if c in X_train.columns]
    for df in [X_train, X_test]:
        for col in active_cat:
            df[col] = df[col].fillna("__MISSING__").astype(str)

    return X_train, y, X_test, active_cat, encounter_keys


# ─── Base model: CatBoost ────────────────────────────────────────────────────

def run_catboost(X, y, X_test, cat_cols):
    print("\n[CatBoost] 5-fold CV...")
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof        = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fi_accum   = np.zeros(X.shape[1])
    fold_rmse  = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X), 1):
        X_tr, X_vl, X_te = apply_oof_encoding(
            X.iloc[tr_idx], X.iloc[vl_idx], X_test, y[tr_idx]
        )
        pool_tr = Pool(X_tr, y[tr_idx], cat_features=cat_cols)
        pool_vl = Pool(X_vl, y[vl_idx], cat_features=cat_cols)

        m = CatBoostRegressor(**CB_PARAMS)
        m.fit(pool_tr, eval_set=pool_vl, early_stopping_rounds=EARLY_STOP)

        oof[vl_idx]  = m.predict(pool_vl)
        test_preds  += m.predict(Pool(X_te, cat_features=cat_cols)) / N_FOLDS
        fi_accum    += m.get_feature_importance()

        rmse = np.sqrt(mean_squared_error(y[vl_idx], oof[vl_idx]))
        fold_rmse.append(rmse)
        print(f"  Fold {fold}: RMSE={rmse:.4f}  best_iter={m.get_best_iteration()}")

    print(f"CatBoost CV RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")

    fi_df = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "importance": fi_accum / N_FOLDS,
    }).sort_values("importance", ascending=False)

    return oof, test_preds, fi_df


# ─── Shared helper: encode categoricals as integers for LGB / XGB ────────────

def int_encode_cats(X, X_test, cat_cols):
    X_out    = X.copy()
    X_te_out = X_test.copy()
    for col in cat_cols:
        if col not in X_out.columns:
            continue
        codes = pd.Categorical(
            pd.concat([X_out[col], X_te_out[col]], ignore_index=True)
        ).codes
        n = len(X_out)
        X_out[col]    = codes[:n].astype(int)
        X_te_out[col] = codes[n:].astype(int)
    return X_out, X_te_out


# ─── Base model: LightGBM ────────────────────────────────────────────────────

def run_lgbm(X, y, X_test, cat_cols):
    print("\n[LightGBM] 5-fold CV...")
    X_enc, X_te_enc = int_encode_cats(X, X_test, cat_cols)
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof        = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_rmse  = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_enc), 1):
        X_tr, X_vl, X_te = apply_oof_encoding(
            X_enc.iloc[tr_idx], X_enc.iloc[vl_idx], X_te_enc, y[tr_idx]
        )
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(
            X_tr, y[tr_idx],
            eval_set=[(X_vl, y[vl_idx])],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof[vl_idx]  = m.predict(X_vl)
        test_preds  += m.predict(X_te) / N_FOLDS

        rmse = np.sqrt(mean_squared_error(y[vl_idx], oof[vl_idx]))
        fold_rmse.append(rmse)
        print(f"  Fold {fold}: RMSE={rmse:.4f}  best_iter={m.best_iteration_}")

    print(f"LightGBM CV RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")
    return oof, test_preds


# ─── Base model: XGBoost ─────────────────────────────────────────────────────

def run_xgb(X, y, X_test, cat_cols):
    print("\n[XGBoost] 5-fold CV...")
    X_enc, X_te_enc = int_encode_cats(X, X_test, cat_cols)
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof        = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_rmse  = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_enc), 1):
        X_tr, X_vl, X_te = apply_oof_encoding(
            X_enc.iloc[tr_idx], X_enc.iloc[vl_idx], X_te_enc, y[tr_idx]
        )
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=False)
        oof[vl_idx]  = m.predict(X_vl)
        test_preds  += m.predict(X_te) / N_FOLDS

        rmse = np.sqrt(mean_squared_error(y[vl_idx], oof[vl_idx]))
        fold_rmse.append(rmse)
        print(f"  Fold {fold}: RMSE={rmse:.4f}  best_iter={m.best_iteration}")

    print(f"XGBoost CV RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")
    return oof, test_preds


# ─── Ridge meta-learner (stacking) ───────────────────────────────────────────

def ridge_stack(cb_oof, lgb_oof, xgb_oof, cb_test, lgb_test, xgb_test, y):
    print("\n[Stack] Ridge meta-learner on 3 OOF base-model predictions...")
    meta_tr   = np.column_stack([cb_oof, lgb_oof, xgb_oof])
    meta_te   = np.column_stack([cb_test, lgb_test, xgb_test])
    kf        = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    stacked_oof  = np.zeros(len(y))
    stacked_test = np.zeros(len(meta_te))

    for tr_idx, vl_idx in kf.split(meta_tr):
        m = Ridge(alpha=1.0)
        m.fit(meta_tr[tr_idx], y[tr_idx])
        stacked_oof[vl_idx]  = m.predict(meta_tr[vl_idx])
        stacked_test        += m.predict(meta_te) / N_FOLDS

    rmse = np.sqrt(mean_squared_error(y, stacked_oof))
    mae  = mean_absolute_error(y, stacked_oof)
    print(f"Stack CV RMSE: {rmse:.4f}  MAE: {mae:.4f}")
    return stacked_test, rmse, mae


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAS Viya 2026 — LOS prediction pipeline")
    p.add_argument("--train", default="data/train.csv",  help="Path to train.csv")
    p.add_argument("--test",  default="data/test.csv",   help="Path to test.csv")
    p.add_argument("--out",   default="submission.csv",  help="Output submission file")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SAS Viya 2026 — Hospital LOS Prediction Pipeline")
    print("=" * 60)
    print(f"Train: {args.train}")
    print(f"Test:  {args.test}")
    print(f"Output: {args.out}")

    print("\nLoading and preprocessing data...")
    X, y, X_test, cat_cols, encounter_keys = load_data(args.train, args.test)
    print(f"Train shape: {X.shape}  |  Test shape: {X_test.shape}")
    print(f"Target — mean: {y.mean():.2f}  std: {y.std():.2f}  range: [{y.min()}, {y.max()}]")

    cb_oof,  cb_test,  fi_df = run_catboost(X, y, X_test, cat_cols)
    lgb_oof, lgb_test        = run_lgbm(X, y, X_test, cat_cols)
    xgb_oof, xgb_test        = run_xgb(X, y, X_test, cat_cols)

    test_preds, stack_rmse, stack_mae = ridge_stack(
        cb_oof, lgb_oof, xgb_oof,
        cb_test, lgb_test, xgb_test,
        y,
    )

    # Clip small negatives (edge-case extrapolation) and round to nearest integer
    # since ADMIT_LOS is always a whole number of days
    test_preds = np.clip(test_preds, 0, None).round(0)

    submission = pd.DataFrame({
        "ENCOUNTER_KEY": encounter_keys.values,
        "ADMIT_LOS":     test_preds,
    })
    submission.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Stack CV RMSE : {stack_rmse:.4f}")
    print(f"  Stack CV MAE  : {stack_mae:.4f}")
    print(f"  Submission    : {args.out}  ({len(submission):,} rows)")

    print("\nTop 15 features by CatBoost importance:")
    print(fi_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
