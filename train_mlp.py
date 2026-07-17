"""
Train the embedding MLP diversity model.

Categorical columns get learned embeddings (dimension min(50, (n+1)//2)),
continuous columns are standard-scaled per fold. The network is
128 -> 64 -> 32 -> 1 with BatchNorm, ReLU and dropout, trained on
log1p(ADMIT_LOS) with Adam and early stopping. Same 5-fold split
(random_state=42) as the tree stacks so OOF predictions are combinable.

The MLP's errors correlate ~0.925 with the tree stacks (vs 0.9998
between stacks), which is why it earns a place in the final blend.

    python train_mlp.py

Outputs: artifacts/oof_mlp.csv and artifacts/preds_mlp.csv
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import load_data

N_FOLDS = 5
KF_SEED = 42
MAX_EPOCHS = 60
PATIENCE = 8
BATCH = 1024
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(6)


class EmbMLP(nn.Module):
    def __init__(self, cat_dims, n_cont):
        super().__init__()
        self.embs = nn.ModuleList([
            nn.Embedding(n, min(50, (n + 1) // 2)) for n in cat_dims
        ])
        emb_total = sum(min(50, (n + 1) // 2) for n in cat_dims)
        self.net = nn.Sequential(
            nn.Linear(emb_total + n_cont, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_cat, x_cont):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.net(torch.cat(embs + [x_cont], dim=1)).squeeze(1)


def train_fold(Xc_tr, Xn_tr, y_tr, Xc_vl, Xn_vl, y_vl, cat_dims):
    model = EmbMLP(cat_dims, Xn_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    dl = DataLoader(
        TensorDataset(torch.LongTensor(Xc_tr), torch.FloatTensor(Xn_tr), torch.FloatTensor(y_tr)),
        batch_size=BATCH, shuffle=True)
    xc_vl = torch.LongTensor(Xc_vl)
    xn_vl = torch.FloatTensor(Xn_vl)
    y_vl_t = torch.FloatTensor(y_vl)

    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xc, xn, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xc, xn), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl_loss = lossf(model(xc_vl, xn_vl), y_vl_t).item()
        if vl_loss < best_loss - 1e-5:
            best_loss, bad = vl_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    os.makedirs("artifacts", exist_ok=True)
    X, y, X_test, cat_cols, _, _, _, keys = load_data()

    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    Xc = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    Xc_te = np.zeros((len(X_test), len(cat_cols)), dtype=np.int64)
    cat_dims = []
    for i, c in enumerate(cat_cols):
        joint = pd.Categorical(pd.concat([X[c], X_test[c]], axis=0).astype(str))
        codes = joint.codes.astype(np.int64)
        codes[codes < 0] = len(joint.categories)
        Xc[:, i] = codes[:len(X)]
        Xc_te[:, i] = codes[len(X):]
        cat_dims.append(int(codes.max()) + 1)

    Xn_raw = X[num_cols].apply(pd.to_numeric, errors="coerce")
    Xn_te_raw = X_test[num_cols].apply(pd.to_numeric, errors="coerce")
    y_log = np.log1p(y)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=KF_SEED)
    mlp_oof = np.zeros(len(y))
    mlp_test = np.zeros(len(X_test))

    for fold, (tr, vl) in enumerate(kf.split(Xc), 1):
        med = Xn_raw.iloc[tr].median()
        Xn_tr = Xn_raw.iloc[tr].fillna(med).values.astype(np.float32)
        Xn_vl = Xn_raw.iloc[vl].fillna(med).values.astype(np.float32)
        Xn_te = Xn_te_raw.fillna(med).values.astype(np.float32)
        sc = StandardScaler().fit(Xn_tr)
        Xn_tr, Xn_vl, Xn_te = sc.transform(Xn_tr), sc.transform(Xn_vl), sc.transform(Xn_te)

        model = train_fold(Xc[tr], Xn_tr, y_log[tr], Xc[vl], Xn_vl, y_log[vl], cat_dims)
        with torch.no_grad():
            mlp_oof[vl] = model(torch.LongTensor(Xc[vl]), torch.FloatTensor(Xn_vl)).numpy()
            mlp_test += model(torch.LongTensor(Xc_te), torch.FloatTensor(Xn_te)).numpy() / N_FOLDS

        fr = np.sqrt(mean_squared_error(y[vl], np.clip(np.expm1(mlp_oof[vl]), 0, 51)))
        print(f"fold {fold}: RMSE={fr:.4f}")

    mlp_oof_r = np.clip(np.expm1(mlp_oof), 0, 51)
    mlp_test_r = np.clip(np.expm1(mlp_test), 0, 51)
    print(f"MLP solo OOF RMSE: {np.sqrt(mean_squared_error(y, mlp_oof_r)):.4f}")

    pd.DataFrame({"mlp_oof": mlp_oof_r, "ADMIT_LOS": y}).to_csv(
        "artifacts/oof_mlp.csv", index=False)
    pd.DataFrame({"ENCOUNTER_KEY": keys.values, "ADMIT_LOS": mlp_test_r}).to_csv(
        "artifacts/preds_mlp.csv", index=False)
    print("saved artifacts/oof_mlp.csv and artifacts/preds_mlp.csv")


if __name__ == "__main__":
    main()
