#!/usr/bin/env python
"""SMOTE isolation (RQ1). Only the SMOTE location changes between arms.
A: SMOTE inside training fold (correct)
B: SMOTE on all rows before splitting (leak)
C: no SMOTE (control)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hdp.data import FEATURES, load_uci, schema_categories
from hdp.models import make_preprocessor
from hdp.resampling import smote

SITES = ["cleveland", "hungarian", "switzerland", "va"]
K_FEATURES = 8


def make_learner(name, seed):
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=3000)
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                      random_state=seed, n_jobs=-1)
    return HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=20, random_state=seed)


def run_fold(Xtr, ytr, Xte, yte, Xall, yall, cats, learner, seed):
    pre = make_preprocessor(list(FEATURES), cats).fit(Xtr)
    Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
    out = {}

    # C: control, no oversampling
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Ztr, ytr)
    clf = make_learner(learner, seed).fit(sel.transform(Ztr), ytr)
    out["C_no_smote"] = roc_auc_score(yte, clf.predict_proba(sel.transform(Zte))[:, 1])

    # A: correct, SMOTE on training fold only
    Zs, ys = smote(sel.transform(Ztr), ytr, np.random.RandomState(seed))
    clf = make_learner(learner, seed).fit(Zs, ys)
    out["A_smote_in_fold"] = roc_auc_score(yte, clf.predict_proba(sel.transform(Zte))[:, 1])

    # B: leak, SMOTE on all rows (test patients' synthetic copies enter training)
    Zall = pre.transform(Xall)
    Zs_all, ys_all = smote(Zall, yall, np.random.RandomState(seed))
    sel_b = SelectKBest(f_classif, k=K_FEATURES).fit(Zs_all, ys_all)
    clf = make_learner(learner, seed).fit(sel_b.transform(Zs_all), ys_all)
    out["B_smote_all_rows"] = roc_auc_score(yte, clf.predict_proba(sel_b.transform(Zte))[:, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/uci")
    ap.add_argument("--learner", choices=["LR", "RF", "GB"], default="RF")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/smote_isolation_{a.learner}")
    out.mkdir(parents=True, exist_ok=True)

    df = load_uci(a.data_dir, SITES)
    cats = schema_categories(df)
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats,
                                          random_state=a.seed).split(X, y))
    keys = ["C_no_smote", "A_smote_in_fold", "B_smote_all_rows"]
    scores = {k: [] for k in keys}
    for i, (tr, te) in enumerate(splits):
        res = run_fold(X.iloc[tr], y[tr], X.iloc[te], y[te], X, y, cats, a.learner, a.seed)
        for k in keys:
            scores[k].append(res[k])
        print(f"[{a.learner}] fold {i + 1}/{len(splits)} done", flush=True)

    res_df = pd.DataFrame([dict(learner=a.learner, arm=k,
                                mean_auroc=round(float(np.mean(scores[k])), 4),
                                sd=round(float(np.std(scores[k])), 4)) for k in keys])
    res_df.to_csv(out / "smote_isolation.csv", index=False)
    print(f"\n=== SMOTE isolation: {a.learner} ===")
    print(res_df.drop(columns="learner").to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
