"""
SAS Viya for Learners Challenge 2026
Predict ADMIT_LOS (hospital length of stay in days)

Approach:
  - CatBoost + LightGBM + XGBoost + HistGradientBoosting ensemble
  - Log1p target transformation (expm1 to recover predictions)
  - OOF target-encoding for DOCTOR + 3 high-signal categoricals
    (DX_CODE, DIAGNOSIS_SUBCAT_CODE, DEPARTMENT)
  - 3-seed bagging [42, 7, 123] for variance reduction
  - Ridge meta-learner stacking
  - Raw float predictions (no integer rounding)
  - All params Optuna-tuned on raw float RMSE (Stage 5)

CV RMSE: 1.9767  |  Kaggle LB: 1.9469 (Stage 9A)

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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS   = [42, 7, 123]
N_FOLDS = 5
EARLY_STOP = 150

# Optuna-tuned CatBoost params (Stage 5, 20 trials → best CV 1.9821)
CB_PARAMS_BASE = dict(
    iterations          = 5000,
    learning_rate       = 0.04050837781329675,
    depth               = 6,
    l2_leaf_reg         = 1.2151617026673374,
    bagging_temperature = 1.42332830588,
    random_strength     = 2.9140800826863984,
    min_data_in_leaf    = 41,
    eval_metric="RMSE", verbose=0, allow_writing_files=False,
)

# Optuna-tuned LightGBM params (Stage 5)
LGB_PARAMS_BASE = dict(
    n_estimators=5000, verbose=-1,
    num_leaves       = 65,
    learning_rate    = 0.037198143773777674,
    min_child_samples= 5,
    subsample        = 0.683417948,
    colsample_bytree = 0.658219841,
    reg_lambda       = 1.0,
    reg_alpha        = 3.65,
)

# XGBoost — Stage 4 params (Optuna found lr=0.0103 hitting iter cap; reverted)
XGB_PARAMS_BASE = dict(
    n_estimators=5000, verbosity=0, eval_metric="rmse",
    early_stopping_rounds=EARLY_STOP,
    max_depth=7, learning_rate=0.03,
    reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist",
)

# Optuna-tuned HistGradientBoosting params (Stage 5)
HGBM_PARAMS_BASE = dict(
    max_iter=5000,
    max_leaf_nodes   = 31,
    learning_rate    = 0.010489291,
    min_samples_leaf = 40,
    l2_regularization= 0.159447,
    early_stopping   = True,
    validation_fraction= 0.1,
    n_iter_no_change = 50,
)

# Columns always dropped
DROP_COLS = [
    "ENCOUNTER_KEY", "PATIENT_NUMBER",
    "ADMIT_DATE", "DISCHARGE_DATE",      # absent from updated dataset
    "DIAGNOSIS_ICD_CODE",                # r=1.000 with DIAGNOSIS_SUBCAT_CODE
    "DISCH_NURSE_ID",                    # r=-0.001 in updated dataset
    "PROCEDURE_SUBCAT_CODE", "PROCEDURE_SUBCAT_DESC",
    "PROCEDURE_ICD_CODE", "PROCEDURE_LONG_DESC",
    "MS_DRG_DESC", "DRG_APR_DESC",
    "DIAGNOSIS_SUBCAT_DESC", "DIAGNOSIS_LONG_DESC",
    "HOSPITAL",    # replaced by HOSPITAL_freq
    "ADMIT_LOS",   # target
]

# Low-cardinality nominals — CatBoost encodes natively; LGB/XGB get integer codes
CAT_COLS = [
    "GENDER", "RACE_CD", "STATECODE", "CITY", "COUNTY_NAME",
    "REGION", "DEPARTMENT", "DIAGNOSIS_GROUP", "DX_GROUP",
    "STANDARD_ORDERS_USED", "DISCHARGED_TO",
]

# High-signal categoricals that receive fold-safe OOF target-encoding
# These get a _TE numeric column added alongside their existing representation
NEW_TE_COLS = [
    ("DX_CODE",               20),   # 55 cats, group-mean std=2.76
    ("DIAGNOSIS_SUBCAT_CODE", 20),   # 21 cats, group-mean std=2.68
    ("DEPARTMENT",            20),   # 10 cats, group-mean std=2.35
]

# OOF TE for high-cardinality ID columns
BASE_ENCODE = [("DOCTOR", 20), ("HOSPITAL", 20)]


# ─── Preprocessing ───────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # DRG_APR_SEVERITY: handle both old (whitespace + "." sentinel) and new (clean int) formats
    if df["DRG_APR_SEVERITY"].dtype == object:
        sev_map = {"1": 1, "2": 2, "3": 3, "4": 4}
        df["DRG_APR_SEVERITY"] = (
            df["DRG_APR_SEVERITY"].str.strip()
            .replace(".", np.nan)
            .map(sev_map)
            .astype("float64")
        )

    # NUM_CHRONIC_COND stored as object in some exports
    df["NUM_CHRONIC_COND"] = pd.to_numeric(df["NUM_CHRONIC_COND"], errors="coerce")

    # ORDER_TOTAL_CHARGES: -2104 is a billing credit sentinel in old dataset
    df["ORDER_TOTAL_CHARGES"] = df["ORDER_TOTAL_CHARGES"].replace(-2104, np.nan)

    # DRG_APR_CODE can be object-typed in SAS exports
    df["DRG_APR_CODE"] = pd.to_numeric(df["DRG_APR_CODE"], errors="coerce")

    # Binary procedure flag
    df["HAS_PROCEDURE"] = (df["OPERATION_COUNT"] > 0).astype(int)

    # Strip residual whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    sev = df["DRG_APR_SEVERITY"].fillna(2)
    cc  = df["NUM_CHRONIC_COND"].fillna(0)
    icu = df["ICU_DAYS"]
    mh  = df.get("MONITORING_HOURS",  pd.Series(0, index=df.index)).fillna(0)
    ci  = df.get("COMORBIDITY_INDEX", pd.Series(0, index=df.index)).fillna(0)
    cts = df.get("CARE_TEAM_SIZE",    pd.Series(1, index=df.index)).fillna(1)
    charges = df["ORDER_TOTAL_CHARGES"].fillna(
        df["ORDER_TOTAL_CHARGES"].median()
    ).clip(lower=1)

    df["AGE_x_SEVERITY"]       = df["PATIENT_AGE"] * sev
    df["CHRONIC_x_SEVERITY"]   = cc * sev
    df["ICU_x_CHRONIC"]        = icu * cc
    df["ICU_x_OPERATION"]      = icu * df["OPERATION_COUNT"]
    df["LOG_CHARGES"]          = np.log1p(charges)
    df["CHARGE_PER_ICU"]       = charges / (icu + 1)
    df["ICU_DAYS_SQRT"]        = np.sqrt(icu)
    df["PATIENT_AGE_SQ"]       = df["PATIENT_AGE"] ** 2
    df["ADMIT_QUARTER"]        = ((df["ADMIT_MTH"] - 1) // 3) + 1
    df["IS_SUMMER"]            = df["ADMIT_MTH"].isin([7, 8, 9]).astype(int)
    df["MONITOR_x_ICU"]        = mh * icu
    df["COMORBID_x_SEV"]       = ci * sev
    df["TEAM_x_COMORBID"]      = cts * ci
    df["LOG_MONITORING"]       = np.log1p(mh)
    df["MONITOR_PER_COMORBID"] = mh / (ci + 1)
    df["COMORBID_SQ"]          = ci ** 2
    df["TEAM_x_ICU"]           = cts * icu
    return df


# ─── OOF target encoding ─────────────────────────────────────────────────────

def _smoothed_encode(s_tr, y_tr, s_te, global_mean, smoothing):
    stats    = pd.Series(y_tr, index=s_tr.index).groupby(s_tr.values).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    enc_tr   = s_tr.map(smoothed).fillna(global_mean).values
    enc_te   = s_te.map(smoothed).fillna(global_mean).values
    return enc_tr, enc_te, smoothed


def apply_oof_encoding(X_tr, X_vl, X_te_in, y_tr, encode_cols):
    """Apply smoothed OOF TE to high-cardinality ID columns."""
    X_tr = X_tr.copy(); X_vl = X_vl.copy(); X_te = X_te_in.copy()
    gm   = float(np.mean(y_tr))
    for col, sw in encode_cols:
        if col not in X_tr.columns:
            continue
        enc_tr, enc_vl, smoothed = _smoothed_encode(X_tr[col], y_tr, X_vl[col], gm, sw)
        enc_te_vals = X_te[col].map(smoothed).fillna(gm).values
        X_tr[col] = enc_tr; X_vl[col] = enc_vl; X_te[col] = enc_te_vals
    return X_tr, X_vl, X_te


def add_new_te_features(X_tr, X_vl, X_te, y_tr, raw_tr, raw_vl, raw_te):
    """Add fold-safe OOF TE numeric columns for high-signal categoricals."""
    gm = float(np.mean(y_tr))
    X_tr = X_tr.copy(); X_vl = X_vl.copy(); X_te = X_te.copy()
    for col, sw in NEW_TE_COLS:
        new_col = f"{col}_TE"
        if col not in raw_tr.columns:
            continue
        enc_tr, enc_vl, smoothed = _smoothed_encode(
            raw_tr[col].astype(str), y_tr,
            raw_vl[col].astype(str), gm, sw
        )
        enc_te_vals = raw_te[col].astype(str).map(smoothed).fillna(gm).values
        X_tr[new_col] = enc_tr
        X_vl[new_col] = enc_vl
        X_te[new_col] = enc_te_vals
    return X_tr, X_vl, X_te


# ─── Data loading ────────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str):
    train_raw = pd.read_csv(train_path, low_memory=False, encoding="utf-8-sig")
    test_raw  = pd.read_csv(test_path,  low_memory=False, encoding="utf-8-sig")

    # Strip BOM characters from ENCOUNTER_KEY if present
    encounter_keys = (
        test_raw["ENCOUNTER_KEY"].astype(str)
        .str.replace("﻿", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    # Try to cast to int if possible
    try:
        encounter_keys = encounter_keys.astype("int64")
    except ValueError:
        pass

    y = train_raw["ADMIT_LOS"].values

    train = clean(train_raw.copy())
    test  = clean(test_raw.copy())
    train = engineer(train)
    test  = engineer(test)

    # Hospital frequency encoding (volume proxy; hospital mean LOS has tiny variance)
    hosp_freq = train["HOSPITAL"].value_counts().to_dict()
    med_freq  = float(np.median(list(hosp_freq.values())))
    for df in [train, test]:
        df["HOSPITAL_freq"] = df["HOSPITAL"].map(hosp_freq).fillna(med_freq)

    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    cat_cols_active = [c for c in CAT_COLS if c in feature_cols]
    for df in [train, test]:
        for col in cat_cols_active:
            df[col] = df[col].fillna("__MISSING__").astype(str)

    return (train[feature_cols].copy(), y,
            test[feature_cols].copy(), cat_cols_active,
            train_raw, test_raw, encounter_keys)


# ─── Per-seed model training ──────────────────────────────────────────────────

def run_one_seed(X, y, X_test, cat_cols, raw_train, raw_test, seed):
    print(f"\n  === Seed {seed} ===")
    kf    = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    y_log = np.log1p(y)

    cb_oof   = np.zeros(len(y)); cb_test   = np.zeros(len(X_test))
    lgb_oof  = np.zeros(len(y)); lgb_test  = np.zeros(len(X_test))
    xgb_oof  = np.zeros(len(y)); xgb_test  = np.zeros(len(X_test))
    hgbm_oof = np.zeros(len(y)); hgbm_test = np.zeros(len(X_test))

    for fold, (tr, vl) in enumerate(kf.split(X), 1):
        X_tr, X_vl, X_te = apply_oof_encoding(
            X.iloc[tr], X.iloc[vl], X_test, y[tr], BASE_ENCODE
        )
        X_tr, X_vl, X_te = add_new_te_features(
            X_tr, X_vl, X_te, y[tr],
            raw_train.iloc[tr], raw_train.iloc[vl], raw_test
        )
        active_cat = [c for c in cat_cols if c in X_tr.columns]

        # ── CatBoost ──────────────────────────────────────────────────────────
        cb_p = dict(CB_PARAMS_BASE, random_seed=seed)
        pool_tr = Pool(X_tr, y_log[tr], cat_features=active_cat)
        pool_vl = Pool(X_vl, y_log[vl], cat_features=active_cat)
        pool_te = Pool(X_te,            cat_features=active_cat)
        m_cb = CatBoostRegressor(**cb_p)
        m_cb.fit(pool_tr, eval_set=pool_vl, early_stopping_rounds=EARLY_STOP)
        cb_oof[vl]  = np.expm1(m_cb.predict(pool_vl))
        cb_test    += np.expm1(m_cb.predict(pool_te)) / N_FOLDS

        # ── LightGBM ──────────────────────────────────────────────────────────
        lgb_p = dict(LGB_PARAMS_BASE, random_state=seed)
        X_tr_l = X_tr.copy(); X_vl_l = X_vl.copy(); X_te_l = X_te.copy()
        for c in active_cat:
            if c in X_tr_l.columns:
                X_tr_l[c] = pd.Categorical(X_tr_l[c]).codes
                X_vl_l[c] = pd.Categorical(X_vl_l[c]).codes
                X_te_l[c] = pd.Categorical(X_te_l[c]).codes
        m_lgb = lgb.LGBMRegressor(**lgb_p)
        m_lgb.fit(X_tr_l, y_log[tr],
                  eval_set=[(X_vl_l, y_log[vl])],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                             lgb.log_evaluation(-1)])
        lgb_oof[vl]  = np.expm1(m_lgb.predict(X_vl_l))
        lgb_test    += np.expm1(m_lgb.predict(X_te_l)) / N_FOLDS

        # ── XGBoost ───────────────────────────────────────────────────────────
        xgb_p = dict(XGB_PARAMS_BASE, random_state=seed)
        X_tr_x = X_tr.copy(); X_vl_x = X_vl.copy(); X_te_x = X_te.copy()
        for c in active_cat:
            if c in X_tr_x.columns:
                codes = pd.Categorical(pd.concat([X_tr_x[c], X_vl_x[c], X_te_x[c]])).codes
                n1 = len(X_tr_x); n2 = len(X_vl_x)
                X_tr_x[c] = codes[:n1]
                X_vl_x[c] = codes[n1:n1+n2]
                X_te_x[c] = codes[n1+n2:]
        m_xgb = xgb.XGBRegressor(**xgb_p)
        m_xgb.fit(X_tr_x, y_log[tr], eval_set=[(X_vl_x, y_log[vl])], verbose=False)
        xgb_oof[vl]  = np.expm1(m_xgb.predict(X_vl_x))
        xgb_test    += np.expm1(m_xgb.predict(X_te_x)) / N_FOLDS

        # ── HistGradientBoosting ───────────────────────────────────────────────
        hgbm_p = dict(HGBM_PARAMS_BASE, random_state=seed)
        X_tr_h = X_tr.copy(); X_vl_h = X_vl.copy(); X_te_h = X_te.copy()
        for c in active_cat:
            if c in X_tr_h.columns:
                codes = pd.Categorical(pd.concat([X_tr_h[c], X_vl_h[c], X_te_h[c]])).codes
                n1 = len(X_tr_h); n2 = len(X_vl_h)
                X_tr_h[c] = codes[:n1]
                X_vl_h[c] = codes[n1:n1+n2]
                X_te_h[c] = codes[n1+n2:]
        m_hgbm = HistGradientBoostingRegressor(**hgbm_p)
        m_hgbm.fit(X_tr_h, y_log[tr])
        hgbm_oof[vl]  = np.expm1(m_hgbm.predict(X_vl_h))
        hgbm_test    += np.expm1(m_hgbm.predict(X_te_h)) / N_FOLDS

        rmse_cb  = np.sqrt(mean_squared_error(y[vl], cb_oof[vl]))
        rmse_lgb = np.sqrt(mean_squared_error(y[vl], lgb_oof[vl]))
        print(f"    Fold {fold}: CB={rmse_cb:.4f}  LGB={rmse_lgb:.4f}")

    return (cb_oof, lgb_oof, xgb_oof, hgbm_oof,
            cb_test, lgb_test, xgb_test, hgbm_test)


# ─── Ridge meta-learner ───────────────────────────────────────────────────────

def ridge_stack(all_oofs, y, all_tests):
    cb_oof_avg   = np.mean([o[0] for o in all_oofs], axis=0)
    lgb_oof_avg  = np.mean([o[1] for o in all_oofs], axis=0)
    xgb_oof_avg  = np.mean([o[2] for o in all_oofs], axis=0)
    hgbm_oof_avg = np.mean([o[3] for o in all_oofs], axis=0)
    cb_test_avg   = np.mean([t[0] for t in all_tests], axis=0)
    lgb_test_avg  = np.mean([t[1] for t in all_tests], axis=0)
    xgb_test_avg  = np.mean([t[2] for t in all_tests], axis=0)
    hgbm_test_avg = np.mean([t[3] for t in all_tests], axis=0)

    X_meta = np.column_stack([cb_oof_avg, lgb_oof_avg, xgb_oof_avg, hgbm_oof_avg])
    X_test = np.column_stack([cb_test_avg, lgb_test_avg, xgb_test_avg, hgbm_test_avg])

    # CV estimate of stacked RMSE
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_rmses = []
    for tr, vl in kf.split(X_meta):
        r = Ridge(alpha=1.0)
        r.fit(X_meta[tr], y[tr])
        fold_rmses.append(np.sqrt(mean_squared_error(y[vl], r.predict(X_meta[vl]))))

    # Full-data fit for test predictions
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_meta, y)
    stacked_test = ridge.predict(X_test)

    cv_rmse = float(np.mean(fold_rmses))
    cv_std  = float(np.std(fold_rmses))
    names   = ["CB", "LGB", "XGB", "HGBM"]
    coef_str = "  ".join(f"{n}:{c:.3f}" for n, c in zip(names, ridge.coef_))
    print(f"\n  Ridge coefs → {coef_str}")
    print(f"  Stack CV RMSE ({len(SEEDS)} seeds): {cv_rmse:.4f} ± {cv_std:.4f}")
    return stacked_test, cv_rmse, cv_std


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
    print(f"Seeds: {SEEDS}  |  Models: CB + LGB + XGB + HGBM")
    print(f"New OOF TE features: {[c for c, _ in NEW_TE_COLS]}")
    print("=" * 60)
    print(f"Train: {args.train}")
    print(f"Test:  {args.test}")
    print(f"Output: {args.out}")

    print("\nLoading and preprocessing data...")
    X, y, X_test, cat_cols, raw_train, raw_test, encounter_keys = load_data(
        args.train, args.test
    )
    print(f"Train shape: {X.shape}  |  Test shape: {X_test.shape}")
    print(f"Target — mean: {y.mean():.2f}  std: {y.std():.2f}  range: [{y.min()}, {y.max()}]")

    all_oofs  = []
    all_tests = []
    for seed in SEEDS:
        result = run_one_seed(X, y, X_test, cat_cols, raw_train, raw_test, seed)
        all_oofs.append(result[:4])
        all_tests.append(result[4:])
        print(f"  Seed {seed} complete.")

    stacked_test, cv_rmse, cv_std = ridge_stack(all_oofs, y, all_tests)

    # Clip to valid range — no rounding (raw floats give lower RMSE than integers)
    stacked_test = np.clip(stacked_test, 0, 51)

    submission = pd.DataFrame({
        "ENCOUNTER_KEY": encounter_keys.values,
        "ADMIT_LOS":     stacked_test,
    })
    submission.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Stack CV RMSE : {cv_rmse:.4f} ± {cv_std:.4f}")
    print(f"  Submission    : {args.out}  ({len(submission):,} rows)")
    print(f"  Predictions   : min={stacked_test.min():.2f}  max={stacked_test.max():.2f}  mean={stacked_test.mean():.2f}")


if __name__ == "__main__":
    main()
