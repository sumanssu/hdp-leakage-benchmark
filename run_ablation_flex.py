#!/usr/bin/env python
"""Leakage ablation with a configurable model (RQ1).

Same protocols as run_ablation_v2.py, but the learner is a parameter:
LR (low capacity), RF or GB (flexible). Leakage is expected to matter more
for flexible learners, so comparing models tests that prediction.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
    if name == "GB":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                              min_samples_leaf=20, random_state=seed)
    raise ValueError(name)


def fit_predict(Xtr, ytr, Xte, cats, learner, seed, pre=None, sel_mask=None,
                resample=False, select=True):
    pre = pre or make_preprocessor(list(FEATURES), cats).fit(Xtr)
    Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
    if sel_mask is not None:
        Ztr, Zte = Ztr[:, sel_mask], Zte[:, sel_mask]
    elif select:
        sel = SelectKBest(f_classif, k=min(K_FEATURES, Ztr.shape[1])).fit(Ztr, ytr)
        Ztr, Zte = sel.transform(Ztr), sel.transform(Zte)
    if resample:
        Ztr, ytr = smote(Ztr, ytr, np.random.RandomState(seed))
    clf = make_learner(learner, seed).fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1]


def run_fold(Xtr, ytr, Xte, yte, Xall, yall, cats, learner, seed):
    out = {}
    out["correct"] = roc_auc_score(yte, fit_predict(Xtr, ytr, Xte, cats, learner, seed))
    out["correct_smote"] = roc_auc_score(
        yte, fit_predict(Xtr, ytr, Xte, cats, learner, seed, resample=True))

    pre_all = make_preprocessor(list(FEATURES), cats).fit(Xall)
    out["leak_scaling"] = roc_auc_score(
        yte, fit_predict(Xtr, ytr, Xte, cats, learner, seed, pre=pre_all))

    pre_tr = make_preprocessor(list(FEATURES), cats).fit(Xtr)
    Zs, ys = smote(pre_tr.transform(Xall), yall, np.random.RandomState(seed))
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Zs, ys)
    clf = make_learner(learner, seed).fit(sel.transform(Zs), ys)
    out["leak_smote"] = roc_auc_score(
        yte, clf.predict_proba(sel.transform(pre_tr.transform(Xte)))[:, 1])

    pre_all2 = make_preprocessor(list(FEATURES), cats).fit(Xall)
    sel_all = SelectKBest(f_classif, k=K_FEATURES).fit(pre_all2.transform(Xall), yall)
    out["leak_selection"] = roc_auc_score(
        yte, fit_predict(Xtr, ytr, Xte, cats, learner, seed,
                         pre=pre_all2, sel_mask=sel_all.get_support()))

    # test-set tuning over a small grid of the learner's main capacity knob
    grid = {"LR": [0.01, 0.1, 1, 10], "RF": [3, 5, 8, 12], "GB": [2, 3, 4, 5]}[learner]
    best = -1.0
    for v in grid:
        if learner == "LR":
            clf = LogisticRegression(C=v, max_iter=3000)
        elif learner == "RF":
            clf = RandomForestClassifier(n_estimators=300, max_depth=v, min_samples_leaf=3,
                                         random_state=seed, n_jobs=-1)
        else:
            clf = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=v,
                                                 min_samples_leaf=20, random_state=seed)
        pre = make_preprocessor(list(FEATURES), cats).fit(Xtr)
        Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
        sel = SelectKBest(f_classif, k=K_FEATURES).fit(Ztr, ytr)
        clf.fit(sel.transform(Ztr), ytr)
        best = max(best, roc_auc_score(yte, clf.predict_proba(sel.transform(Zte))[:, 1]))
    out["leak_test_tuning"] = best
    return out


def paired_test(a, b):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + 1 / 4) * var)  # Nadeau-Bengio, 5-fold
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/uci")
    ap.add_argument("--learner", choices=["LR", "RF", "GB"], default="RF")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"results/ablation_{a.learner}")
    out.mkdir(parents=True, exist_ok=True)

    df = load_uci(a.data_dir, SITES)
    cats = schema_categories(df)
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats,
                                          random_state=a.seed).split(X, y))
    keys = ["correct", "correct_smote", "leak_scaling", "leak_smote",
            "leak_selection", "leak_test_tuning"]
    scores = {k: [] for k in keys}
    for i, (tr, te) in enumerate(splits):
        res = run_fold(X.iloc[tr], y[tr], X.iloc[te], y[te], X, y, cats, a.learner, a.seed)
        for k in keys:
            scores[k].append(res[k])
        print(f"[{a.learner}] fold {i + 1}/{len(splits)} done", flush=True)

    pairs = {"leak_scaling": "correct", "leak_smote": "correct_smote",
             "leak_selection": "correct", "leak_test_tuning": "correct"}
    rows = []
    for k in keys:
        row = dict(learner=a.learner, protocol=k,
                   mean_auroc=round(float(np.mean(scores[k])), 4),
                   sd=round(float(np.std(scores[k])), 4),
                   inflation=np.nan, t=np.nan, p_paired=np.nan)
        if k in pairs:
            base = pairs[k]
            row["inflation"] = round(float(np.mean(scores[k]) - np.mean(scores[base])), 4)
            t, p = paired_test(scores[k], scores[base])
            row["t"], row["p_paired"] = round(t, 2), round(p, 4)
        rows.append(row)
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "ablation_results.csv", index=False)
    print(f"\n=== Leakage ablation: {a.learner} ===")
    print(res_df.drop(columns="learner").to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
