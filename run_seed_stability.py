#!/usr/bin/env python
"""SMOTE-seed stability check (RQ1). Repeats the B vs A leak comparison
with several SMOTE random seeds, to show the effect is not seed-specific."""
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
from run_core_arms import paired_test
from run_smote_isolation import make_learner

SITES = ["cleveland", "hungarian", "switzerland", "va"]
K_FEATURES = 8


def fold_A_B(tr, te, X, y, cats, learner, smote_seed):
    Xte, yte = X.iloc[te], y[te]
    pre = make_preprocessor(list(FEATURES), cats).fit(X.iloc[tr])
    Ztr, Zte, Zall = pre.transform(X.iloc[tr]), pre.transform(Xte), pre.transform(X)
    # A: SMOTE inside the training fold
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Ztr, y[tr])
    Zs, ys = smote(sel.transform(Ztr), y[tr], np.random.RandomState(smote_seed))
    clf = make_learner(learner, 42).fit(Zs, ys)
    a = roc_auc_score(yte, clf.predict_proba(sel.transform(Zte))[:, 1])
    # B: SMOTE on all rows (leak)
    Zs_all, ys_all = smote(Zall, y, np.random.RandomState(smote_seed))
    sel_b = SelectKBest(f_classif, k=K_FEATURES).fit(Zs_all, ys_all)
    clf = make_learner(learner, 42).fit(sel_b.transform(Zs_all), ys_all)
    b = roc_auc_score(yte, clf.predict_proba(sel_b.transform(Zte))[:, 1])
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", choices=["LR", "RF", "GB"], default="RF")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    out = Path("results/seed_stability"); out.mkdir(parents=True, exist_ok=True)

    df = load_uci("data/uci", SITES)
    cats = schema_categories(df)
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats,
                                          random_state=42).split(X, y))
    rows = []
    for s in [int(v) for v in a.seeds.split(",")]:
        A, B = [], []
        for i, (tr, te) in enumerate(splits):
            av, bv = fold_A_B(tr, te, X, y, cats, a.learner, s)
            A.append(av); B.append(bv)
        t, p = paired_test(B, A)
        rows.append(dict(learner=a.learner, smote_seed=s,
                         mean_A=round(float(np.mean(A)), 4),
                         mean_B=round(float(np.mean(B)), 4),
                         leak_effect=round(float(np.mean(B) - np.mean(A)), 4),
                         t=round(t, 2), p_paired=round(p, 4)))
        print(f"[{a.learner}] seed {s} done: leak effect {rows[-1]['leak_effect']:+.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"seed_stability_{a.learner}.csv", index=False)
    print("\n=== SMOTE-seed stability ===")
    print(res.to_string(index=False))
    print(f"\nLeak effect across seeds: mean {res.leak_effect.mean():+.4f}, "
          f"range {res.leak_effect.min():+.4f} to {res.leak_effect.max():+.4f}")


if __name__ == "__main__":
    main()
