"""
Train one 5-seed, 4-model stacked ensemble.

Each seed runs 5-fold CV with CatBoost, LightGBM, XGBoost and
HistGradientBoosting on a log1p target. Fold-safe smoothed target
encoding is applied inside each fold. Optional pseudo-labelled test
rows are appended to every fold's training data. Seed-averaged OOF
predictions feed a Ridge meta-learner.

The final leaderboard model averages three runs of this script with
disjoint seed sets:

    python train_stack.py --seeds 42 7 123 999 2025      --tag stack_a --pseudo preds_prev.csv
    python train_stack.py --seeds 100 200 300 400 500    --tag stack_b --pseudo preds_prev.csv
    python train_stack.py --seeds 555 1717 8080 3141 9999 --tag stack_c --pseudo preds_prev.csv

Outputs: artifacts/oof_<tag>.csv and artifacts/preds_<tag>.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

from preprocess import load_data

N_FOLDS = 5
EARLY_STOP = 150

CB_PARAMS = dict(
    iterations=5000, learning_rate=0.04050837781329675, depth=6,
    l2_leaf_reg=1.2151617026673374, bagging_temperature=1.42332830588,
    random_strength=2.9140800826863984, min_data_in_leaf=41,
    eval_metric="RMSE", verbose=0, allow_writing_files=False,
)
LGB_PARAMS = dict(
    n_estimators=5000, verbose=-1, num_leaves=65,
    learning_rate=0.037198143773777674, min_child_samples=5,
    subsample=0.683417948, colsample_bytree=0.658219841,
    reg_lambda=1.0, reg_alpha=3.65,
)
XGB_PARAMS = dict(
    n_estimators=5000, verbosity=0, eval_metric="rmse",
    early_stopping_rounds=EARLY_STOP, max_depth=7, learning_rate=0.03,
    reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
)
HGBM_PARAMS = dict(
    max_iter=5000, max_leaf_nodes=31, learning_rate=0.010489291,
    min_samples_leaf=40, l2_regularization=0.159447,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
)

TE_COLS = [("DX_CODE", 20), ("DIAGNOSIS_SUBCAT_CODE", 20), ("DEPARTMENT", 20)]
BASE_ENCODE = [("DOCTOR", 20), ("HOSPITAL", 20)]


def smoothed_encode(s_tr, y_tr, s_te, gm, sw):
    # sw is the smoothing weight: rare categories are pulled toward the global mean
    stats = pd.Series(y_tr, index=s_tr.index).groupby(s_tr.values).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + gm * sw) / (stats["count"] + sw)
    return (s_tr.map(smoothed).fillna(gm).values,
            s_te.map(smoothed).fillna(gm).values, smoothed)


# called inside each fold so validation rows never touch training labels during encoding
def apply_target_encoding(X_tr, X_vl, X_te, X_ps, y_tr, raw_tr, raw_vl, raw_te, raw_ps):
    X_tr = X_tr.copy(); X_vl = X_vl.copy(); X_te = X_te.copy(); X_ps = X_ps.copy()
    gm = float(np.mean(y_tr))
    for col, sw in BASE_ENCODE:
        if col not in X_tr.columns:
            continue
        e_tr, e_vl, sm = smoothed_encode(X_tr[col], y_tr, X_vl[col], gm, sw)
        X_tr[col] = e_tr; X_vl[col] = e_vl
        X_te[col] = X_te[col].map(sm).fillna(gm).values
        X_ps[col] = X_ps[col].map(sm).fillna(gm).values
    for col, sw in TE_COLS:
        new_col = f"{col}_TE"
        if col not in raw_tr.columns:
            continue
        e_tr, e_vl, sm = smoothed_encode(
            raw_tr[col].astype(str), y_tr, raw_vl[col].astype(str), gm, sw)
        X_tr[new_col] = e_tr; X_vl[new_col] = e_vl
        X_te[new_col] = raw_te[col].astype(str).map(sm).fillna(gm).values
        X_ps[new_col] = raw_ps[col].astype(str).map(sm).fillna(gm).values
    return X_tr, X_vl, X_te, X_ps


def run_one_seed(X, y, X_test, cat_cols, raw_train, raw_test, y_pseudo, seed):
    print(f"seed {seed}")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    y_log = np.log1p(y)
    oofs = {m: np.zeros(len(y)) for m in ["cb", "lgb", "xgb", "hgbm"]}
    tests = {m: np.zeros(len(X_test)) for m in ["cb", "lgb", "xgb", "hgbm"]}
    X_pseudo = X_test.copy()
    raw_pseudo = raw_test.copy()
    y_pseudo_log = np.log1p(np.clip(y_pseudo, 0, 51)) if y_pseudo is not None else None

    for fold, (tr, vl) in enumerate(kf.split(X), 1):
        X_tr, X_vl, X_te, X_ps = apply_target_encoding(
            X.iloc[tr], X.iloc[vl], X_test, X_pseudo, y[tr],
            raw_train.iloc[tr], raw_train.iloc[vl], raw_test, raw_pseudo)

        if y_pseudo_log is not None:
            X_tr_c = pd.concat([X_tr, X_ps], axis=0, ignore_index=True)
            y_log_c = np.concatenate([y_log[tr], y_pseudo_log])
        else:
            X_tr_c, y_log_c = X_tr, y_log[tr]

        cc = [c for c in cat_cols if c in X_tr_c.columns]

        m = CatBoostRegressor(**dict(CB_PARAMS, random_seed=seed))
        m.fit(Pool(X_tr_c, y_log_c, cat_features=cc),
              eval_set=Pool(X_vl, y_log[vl], cat_features=cc),
              early_stopping_rounds=EARLY_STOP)
        oofs["cb"][vl] = np.expm1(m.predict(Pool(X_vl, cat_features=cc)))
        tests["cb"] += np.expm1(m.predict(Pool(X_te, cat_features=cc))) / N_FOLDS

        # LGB needs integer codes; CatBoost handles raw strings natively
        Xl, Xvl, Xte = X_tr_c.copy(), X_vl.copy(), X_te.copy()
        for c in cc:
            Xl[c] = pd.Categorical(Xl[c]).codes
            Xvl[c] = pd.Categorical(Xvl[c]).codes
            Xte[c] = pd.Categorical(Xte[c]).codes
        m = lgb.LGBMRegressor(**dict(LGB_PARAMS, random_state=seed))
        m.fit(Xl, y_log_c, eval_set=[(Xvl, y_log[vl])],
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                         lgb.log_evaluation(-1)])
        oofs["lgb"][vl] = np.expm1(m.predict(Xvl))
        tests["lgb"] += np.expm1(m.predict(Xte)) / N_FOLDS

        # XGB needs consistent integer codes across train, val, and test within the fold
        Xx, Xxv, Xxe = X_tr_c.copy(), X_vl.copy(), X_te.copy()
        for c in cc:
            codes = pd.Categorical(pd.concat([Xx[c], Xxv[c], Xxe[c]])).codes
            n1, n2 = len(Xx), len(Xxv)
            Xx[c] = codes[:n1]; Xxv[c] = codes[n1:n1 + n2]; Xxe[c] = codes[n1 + n2:]
        m = xgb.XGBRegressor(**dict(XGB_PARAMS, random_state=seed))
        m.fit(Xx, y_log_c, eval_set=[(Xxv, y_log[vl])], verbose=False)
        oofs["xgb"][vl] = np.expm1(m.predict(Xxv))
        tests["xgb"] += np.expm1(m.predict(Xxe)) / N_FOLDS

        # same joint-encoding trick for HGBM
        Xh, Xhv, Xhe = X_tr_c.copy(), X_vl.copy(), X_te.copy()
        for c in cc:
            codes = pd.Categorical(pd.concat([Xh[c], Xhv[c], Xhe[c]])).codes
            n1, n2 = len(Xh), len(Xhv)
            Xh[c] = codes[:n1]; Xhv[c] = codes[n1:n1 + n2]; Xhe[c] = codes[n1 + n2:]
        m = HistGradientBoostingRegressor(**dict(HGBM_PARAMS, random_state=seed))
        m.fit(Xh, y_log_c)
        oofs["hgbm"][vl] = np.expm1(m.predict(Xhv))
        tests["hgbm"] += np.expm1(m.predict(Xhe)) / N_FOLDS

        rmse_cb = np.sqrt(mean_squared_error(y[vl], oofs["cb"][vl]))
        print(f"  fold {fold}: CB={rmse_cb:.4f}")

    return oofs, tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pseudo", default=None,
                    help="CSV of test predictions to use as pseudo-labels")
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    X, y, X_test, cat_cols, raw_train, raw_test, y_pseudo, keys = load_data(args.pseudo)

    all_oofs, all_tests = [], []
    for seed in args.seeds:
        oofs, tests = run_one_seed(X, y, X_test, cat_cols, raw_train, raw_test, y_pseudo, seed)
        all_oofs.append(oofs)
        all_tests.append(tests)

    models = ["cb", "lgb", "xgb", "hgbm"]
    oof_cols = [np.mean([o[m] for o in all_oofs], axis=0) for m in models]
    test_cols = [np.mean([t[m] for t in all_tests], axis=0) for m in models]
    X_meta_oof = np.column_stack(oof_cols)
    X_meta_test = np.column_stack(test_cols)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_rmses = []
    for tr, vl in kf.split(X_meta_oof):
        r = Ridge(alpha=1.0).fit(X_meta_oof[tr], y[tr])
        fold_rmses.append(np.sqrt(mean_squared_error(y[vl], r.predict(X_meta_oof[vl]))))

    ridge = Ridge(alpha=1.0).fit(X_meta_oof, y)
    stacked_test = np.clip(ridge.predict(X_meta_test), 0, 51)
    stacked_oof = ridge.predict(X_meta_oof)

    print(f"stack CV RMSE: {np.mean(fold_rmses):.4f} +- {np.std(fold_rmses):.4f}")

    pd.DataFrame({"stacked_oof": stacked_oof, "ADMIT_LOS": y}).to_csv(
        f"artifacts/oof_{args.tag}.csv", index=False)
    pd.DataFrame({"ENCOUNTER_KEY": keys.values, "ADMIT_LOS": stacked_test}).to_csv(
        f"artifacts/preds_{args.tag}.csv", index=False)
    print(f"saved artifacts/oof_{args.tag}.csv and artifacts/preds_{args.tag}.csv")


if __name__ == "__main__":
    main()
