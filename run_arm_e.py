#!/usr/bin/env python
"""Arm E (RQ1): leaked SMOTE on all rows, but feature selection refit on
real training-fold rows only. Synthetic rows are dropped before selection,
so any inflation must come from the learner seeing synthetic test copies."""
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
    Xtr, Xte, yte = X.iloc[tr], X.iloc[te], y[te]
    pre = make_preprocessor(list(FEATURES), cats).fit(Xtr)
    Zall = pre.transform(X)
    n = len(Zall)
    Zs, ys = smote(Zall, y, np.random.RandomState(seed))   # leak: synthetic copies of test rows
    is_orig = np.zeros(len(Zs), bool)
    is_orig[:n] = True
    is_train = np.zeros(len(Zs), bool)
    is_train[:n] = np.isin(np.arange(n), tr)
    # selection and learner see ONLY real training-fold rows; synthetic rows are dropped
    Ztr_real, ytr_real = Zs[is_train], ys[is_train]
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Ztr_real, ytr_real)
    clf = make_learner(learner, seed).fit(sel.transform(Ztr_real), ytr_real)
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
        print(f"[E {a.learner}] fold {i + 1}/{len(splits)} done", flush=True)
    m, s = float(np.mean(scores)), float(np.std(scores))
    pd.DataFrame([dict(learner=a.learner, arm="E_leak_smote_refit_selection",
                       mean_auroc=round(m, 4), sd=round(s, 4))]).to_csv(
        out / "arm_E.csv", index=False)
    print(f"\n=== Arm E ({a.learner}): mean AUROC {m:.4f}  sd {s:.4f} ===")


if __name__ == "__main__":
    main()
