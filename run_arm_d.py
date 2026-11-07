#!/usr/bin/env python
"""Arm D (RQ1 mechanism check): leak-resampling, but train only on real training-fold rows."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hdp.data import FEATURES, load_uci, schema_categories
from hdp.models import make_preprocessor
from hdp.resampling import smote
from run_smote_isolation import make_learner

SITES = ["cleveland", "hungarian", "switzerland", "va"]
K_FEATURES = 8


def run_fold(tr, te, X, y, cats, learner, seed):
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
    pre = make_preprocessor(list(FEATURES), cats).fit(Xtr)
    Zall = pre.transform(X)                         # all rows, indices 0..n-1
    n = len(Zall)
    Zs, ys = smote(Zall, y, np.random.RandomState(seed))   # originals first, then synthetic
    idx_orig = np.arange(n)                         # originals keep index i
    keep = np.zeros(len(Zs), bool)
    keep[:n] = np.isin(idx_orig, tr)                # originals: keep training-fold rows only
    # synthetic rows (indices n onward) stay False: they are dropped
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Zs[keep], ys[keep])
    clf = make_learner(learner, seed).fit(sel.transform(Zs[keep]), ys[keep])
    return roc_auc_score(yte, clf.predict_proba(sel.transform(pre.transform(Xte)))[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", choices=["LR", "RF", "GB"], default="RF")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(f"results/smote_isolation_{a.learner}")
    out.mkdir(parents=True, exist_ok=True)

    df = load_uci("data/uci", SITES)
    cats = schema_categories(df)
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats,
                                          random_state=a.seed).split(X, y))
    scores = []
    for i, (tr, te) in enumerate(splits):
        scores.append(run_fold(tr, te, X, y, cats, a.learner, a.seed))
        print(f"[D {a.learner}] fold {i + 1}/{len(splits)} done", flush=True)
    m, s = float(np.mean(scores)), float(np.std(scores))
    pd.DataFrame([dict(learner=a.learner, arm="D_leak_real_rows_only",
                       mean_auroc=round(m, 4), sd=round(s, 4))]).to_csv(
        out / "arm_D.csv", index=False)
    print(f"\n=== Arm D ({a.learner}): mean AUROC {m:.4f}  sd {s:.4f} ===")


if __name__ == "__main__":
    main()
