#!/usr/bin/env python
"""Arm F (RQ1): leaked SMOTE on all rows, with feature selection fit on
training-fold ORIGINAL rows only. The learner still trains on all resampled
rows, including synthetic copies of test patients. Same seed as arm B, so
the synthetic points are identical and the comparison is paired."""
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
    Xte, yte = X.iloc[te], y[te]
    pre = make_preprocessor(list(FEATURES), cats).fit(X.iloc[tr])
    Zall = pre.transform(X)
    Zs, ys = smote(Zall, y, np.random.RandomState(seed))   # leak: resample all rows
    # selection sees only training-fold originals (no synthetic rows, no test rows)
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Zall[tr], y[tr])
    clf = make_learner(learner, seed).fit(sel.transform(Zs), ys)
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
        print(f"[F {a.learner}] fold {i + 1}/{len(splits)} done", flush=True)
    m, s = float(np.mean(scores)), float(np.std(scores))
    pd.DataFrame([dict(learner=a.learner, arm="F_leak_smote_train_only_selection",
                       mean_auroc=round(m, 4), sd=round(s, 4))]).to_csv(
        out / "arm_F.csv", index=False)
    print(f"\n=== Arm F ({a.learner}): mean AUROC {m:.4f}  sd {s:.4f} ===")


if __name__ == "__main__":
    main()
