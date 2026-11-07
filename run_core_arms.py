#!/usr/bin/env python
"""Core SMOTE arms for LR, RF, GB with paired tests (RQ1, final numbers)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
    Ztr, Zte, Zall = pre.transform(X.iloc[tr]), pre.transform(Xte), pre.transform(X)
    out = {}

    # C: no SMOTE
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Ztr, y[tr])
    clf = make_learner(learner, seed).fit(sel.transform(Ztr), y[tr])
    out["C"] = roc_auc_score(yte, clf.predict_proba(sel.transform(Zte))[:, 1])

    # A: SMOTE inside training fold
    Zs, ys = smote(sel.transform(Ztr), y[tr], np.random.RandomState(seed))
    clf = make_learner(learner, seed).fit(Zs, ys)
    out["A"] = roc_auc_score(yte, clf.predict_proba(sel.transform(Zte))[:, 1])

    # B: SMOTE on all rows, selection on resampled all rows (leak)
    Zs_all, ys_all = smote(Zall, y, np.random.RandomState(seed))
    sel_b = SelectKBest(f_classif, k=K_FEATURES).fit(Zs_all, ys_all)
    clf = make_learner(learner, seed).fit(sel_b.transform(Zs_all), ys_all)
    out["B"] = roc_auc_score(yte, clf.predict_proba(sel_b.transform(Zte))[:, 1])

    # F: SMOTE on all rows, selection on training-fold originals only (leak)
    sel_f = SelectKBest(f_classif, k=K_FEATURES).fit(Zall[tr], y[tr])
    clf = make_learner(learner, seed).fit(sel_f.transform(Zs_all), ys_all)
    out["F"] = roc_auc_score(yte, clf.predict_proba(sel_f.transform(Zte))[:, 1])
    return out


def paired_test(a, b, n_ratio=0.25):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default="LR,RF,GB")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path("results/core_arms"); out.mkdir(parents=True, exist_ok=True)

    df = load_uci("data/uci", SITES)
    cats = schema_categories(df)
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats,
                                          random_state=a.seed).split(X, y))
    summary, tests = [], []
    for learner in a.learners.split(","):
        scores = {k: [] for k in ["C", "A", "B", "F"]}
        for i, (tr, te) in enumerate(splits):
            r = run_fold(tr, te, X, y, cats, learner, a.seed)
            for k in scores:
                scores[k].append(r[k])
            print(f"[{learner}] fold {i + 1}/{len(splits)}", flush=True)
        for k in scores:
            summary.append(dict(learner=learner, arm=k,
                                mean_auroc=round(float(np.mean(scores[k])), 4),
                                sd=round(float(np.std(scores[k])), 4)))
        for x_arm, y_arm in [("A", "C"), ("B", "A"), ("F", "B")]:
            t, p = paired_test(scores[x_arm], scores[y_arm])
            tests.append(dict(learner=learner, comparison=f"{x_arm} vs {y_arm}",
                              diff=round(float(np.mean(scores[x_arm]) - np.mean(scores[y_arm])), 4),
                              t=round(t, 2), p_paired=round(p, 4)))
    pd.DataFrame(summary).to_csv(out / f"core_summary_{a.learners.replace(',', '_')}.csv", index=False)
    pd.DataFrame(tests).to_csv(out / f"core_tests_{a.learners.replace(',', '_')}.csv", index=False)
    print("\n=== Arm means ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n=== Paired tests ===")
    print(pd.DataFrame(tests).to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
