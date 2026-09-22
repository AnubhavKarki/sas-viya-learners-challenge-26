"""
Build the final submission from trained model artifacts.

Each tree stack gets its own isotonic calibration curve, fitted on its
OOF predictions and applied to its test predictions. The three
calibrated stacks are averaged, then the calibrated MLP is mixed in
at weight 0.05 (tuned on cross-validated OOF, never on the leaderboard).

    python build_submission.py

Reads artifacts/oof_*.csv and artifacts/preds_*.csv, writes submission.csv.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

STACKS = ["stack_a", "stack_b", "stack_c"]
MLP_WEIGHT = 0.05
KF = KFold(n_splits=5, shuffle=True, random_state=42)


def calibrate(oof_pred, y, test_pred):
    # out_of_bounds="clip" handles test predictions outside the training range
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_pred, y)
    return np.clip(iso.predict(test_pred), 0, 51)


def honest_calibrated_oof(oof_pred, y):
    # cross-validate the isotonic fit so the calibrated OOF doesn't overfit its own labels
    cal = np.zeros(len(y))
    for tr, vl in KF.split(oof_pred.reshape(-1, 1)):
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof_pred[tr], y[tr])
        cal[vl] = iso.predict(oof_pred[vl])
    return cal


def main():
    y = None
    keys = None
    cal_tests, cal_oofs = [], []

    for tag in STACKS:
        oof = pd.read_csv(f"artifacts/oof_{tag}.csv")
        preds = pd.read_csv(f"artifacts/preds_{tag}.csv")
        if y is None:
            y = oof["ADMIT_LOS"].values
            keys = preds["ENCOUNTER_KEY"].values
        p_oof = oof["stacked_oof"].values
        p_test = preds.set_index("ENCOUNTER_KEY").loc[keys, "ADMIT_LOS"].values
        cal_tests.append(calibrate(p_oof, y, p_test))
        cal_oofs.append(honest_calibrated_oof(p_oof, y))

    mlp_oof = pd.read_csv("artifacts/oof_mlp.csv")["mlp_oof"].values
    mlp_test = pd.read_csv("artifacts/preds_mlp.csv").set_index(
        "ENCOUNTER_KEY").loc[keys, "ADMIT_LOS"].values
    cal_mlp_test = calibrate(mlp_oof, y, mlp_test)
    cal_mlp_oof = honest_calibrated_oof(mlp_oof, y)

    blend_oof = (1 - MLP_WEIGHT) * np.mean(cal_oofs, axis=0) + MLP_WEIGHT * cal_mlp_oof
    print(f"honest blended OOF RMSE: {np.sqrt(mean_squared_error(y, blend_oof)):.4f}")

    final = (1 - MLP_WEIGHT) * np.mean(cal_tests, axis=0) + MLP_WEIGHT * cal_mlp_test
    pd.DataFrame({"ENCOUNTER_KEY": keys, "ADMIT_LOS": final}).to_csv(
        "submission.csv", index=False)
    print("saved submission.csv")


if __name__ == "__main__":
    main()
