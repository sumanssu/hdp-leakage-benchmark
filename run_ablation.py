#!/usr/bin/env python
"""Leakage ablation (RQ1): how much do common pipeline errors inflate AUROC?

Correct protocol: all learned steps refit inside each training fold.
Flawed variants deliberately break one rule each, to measure the inflation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from hdp.data import FEATURES, load_uci, schema_categories
from hdp.models import build_pipeline, make_preprocessor
from hdp.resampling import smote

SITES = ["cleveland", "hungarian", "switzerland", "va"]


def _auc(y, p):
    return roc_auc_score(y, p)


def correct_fold(X_tr, y_tr, X_te, cats, seed):
    """Baseline: preprocessing and model fit on training fold only."""
    pipe, _ = build_pipeline("LR", cats, seed)
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_te)[:, 1]


def flawed_global_scaling(X_tr, y_tr, X_te, cats, seed, X_all):
    """Flaw 1: impute and scale using ALL data (incl. test) before splitting."""
    pre = make_preprocessor(list(FEATURES), cats).fit(X_all)  # leak: sees test rows
    clf = LogisticRegression(max_iter=2000).fit(pre.transform(X_tr), y_tr)
    return clf.predict_proba(pre.transform(X_te))[:, 1]


def flawed_feature_selection(X_tr, y_tr, X_te, cats, seed, X_all, y_all):
    """Flaw 2: choose the top features with labels from ALL data, then train normally."""
    pre = make_preprocessor(list(FEATURES), cats)
    Z_all = pre.fit_transform(X_all)
    sel = SelectKBest(f_classif, k=5).fit(Z_all, y_all)  # leak: uses test labels
    clf = LogisticRegression(max_iter=2000).fit(sel.transform(pre.transform(X_tr)), y_tr)
    return clf.predict_proba(sel.transform(pre.transform(X_te)))[:, 1]


def flawed_smote_before_split(X_tr, y_tr, X_te, cats, seed, X_all, y_all):
    """Flaw 3: oversample the WHOLE dataset before splitting, so synthetic
    copies of test patients can appear in training. Then train normally."""
    rng = np.random.RandomState(seed)
    pre = make_preprocessor(list(FEATURES), cats).fit(X_all)
    Z_all, y_res = smote(pre.transform(X_all), y_all, rng)   # leak: resample before split
    n_orig = len(X_all)
    # rows of Z_all that came from the test fold's original patients are the leak;
    # we train on the resampled full set, then score the real test rows
    clf = LogisticRegression(max_iter=2000).fit(Z_all, y_res)
    return clf.predict_proba(pre.transform(X_te))[:, 1]


def flawed_test_tuning(X_tr, y_tr, X_te, y_te, cats, seed):
    """Flaw 4: pick the best of a few configs by TEST AUROC, then report it."""
    best = -1.0
    for C in [0.01, 0.1, 1, 10]:
        pipe, _ = build_pipeline("LR", cats, seed)
        pipe.set_params(clf__C=C).fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        best = max(best, _auc(y_te, p))  # leak: selection uses test labels
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/uci")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/ablation")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_uci(a.data_dir, SITES)
    cats = schema_categories(df)
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()

    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats,
                                          random_state=a.seed).split(X, y))
    scores = {k: [] for k in ["correct", "global_scaling", "feature_selection",
                              "smote_before_split", "test_tuning"]}

    for i, (tr, te) in enumerate(splits):
        Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
        scores["correct"].append(_auc(yte, correct_fold(Xtr, ytr, Xte, cats, a.seed)))
        scores["global_scaling"].append(_auc(yte, flawed_global_scaling(Xtr, ytr, Xte, cats, a.seed, X)))
        scores["feature_selection"].append(_auc(yte, flawed_feature_selection(Xtr, ytr, Xte, cats, a.seed, X, y)))
        scores["smote_before_split"].append(_auc(yte, flawed_smote_before_split(Xtr, ytr, Xte, cats, a.seed, X, y)))
        scores["test_tuning"].append(flawed_test_tuning(Xtr, ytr, Xte, yte, cats, a.seed))
        print(f"fold {i + 1}/{len(splits)} done", flush=True)

    base = np.mean(scores["correct"])
    rows = []
    for k, v in scores.items():
        m = float(np.mean(v))
        rows.append(dict(protocol=k, mean_auroc=round(m, 4), sd=round(float(np.std(v)), 4),
                         inflation_vs_correct=round(m - base, 4)))
    res = pd.DataFrame(rows)
    res.to_csv(out / "ablation_results.csv", index=False)
    print("\n=== Leakage ablation (LR, repeated 5-fold, pooled UCI sites) ===")
    print(res.to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
