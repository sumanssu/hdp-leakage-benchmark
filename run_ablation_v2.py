#!/usr/bin/env python
"""Leakage ablation v2 (RQ1), corrected.

Every protocol uses the SAME model (logistic regression) and the SAME features.
Only the place where a learned step happens differs:
  correct            : scale, impute, select, SMOTE, fit all inside each training fold
  correct_smote      : SMOTE inside each training fold only (fair baseline for the SMOTE leak)
  leak_scaling       : scaler/imputer fit on all rows (including test) before splitting
  leak_smote         : SMOTE on all rows before splitting (synthetic copies of test patients enter training)
  leak_selection     : top-k features chosen with labels from all rows before splitting
  leak_test_tuning   : hyperparameter chosen by test-fold AUROC
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hdp.data import FEATURES, load_uci, schema_categories
from hdp.models import make_preprocessor
from hdp.resampling import smote

SITES = ["cleveland", "hungarian", "switzerland", "va"]
K_FEATURES = 8   # same k used in every selection step, for a fair comparison
C_GRID = [0.01, 0.1, 1, 10]


def fit_predict(Xtr, ytr, Xte, cats, C=1.0, select=True, resample=False, seed=0,
                pre=None, sel_mask=None):
    """Generic inner pipeline. Learned steps run only on the arrays passed in."""
    pre = pre or make_preprocessor(list(FEATURES), cats).fit(Xtr)
    Ztr, Zte = pre.transform(Xtr), pre.transform(Xte)
    if sel_mask is not None:
        Ztr, Zte = Ztr[:, sel_mask], Zte[:, sel_mask]
    elif select:
        sel = SelectKBest(f_classif, k=min(K_FEATURES, Ztr.shape[1])).fit(Ztr, ytr)
        Ztr, Zte = sel.transform(Ztr), sel.transform(Zte)
    if resample:
        Ztr, ytr = smote(Ztr, ytr, np.random.RandomState(seed))
    clf = LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr)
    return clf.predict_proba(Zte)[:, 1]


def run_fold(Xtr, ytr, Xte, yte, Xall, yall, cats, seed):
    out = {}
    # correct: everything inside the training fold
    out["correct"] = roc_auc_score(yte, fit_predict(Xtr, ytr, Xte, cats, seed=seed))
    # correct SMOTE: resample only the training fold
    out["correct_smote"] = roc_auc_score(yte, fit_predict(Xtr, ytr, Xte, cats, resample=True, seed=seed))

    # leak 1: scaler/imputer fit on ALL rows. Selection is still inside the fold.
    pre_all = make_preprocessor(list(FEATURES), cats).fit(Xall)
    out["leak_scaling"] = roc_auc_score(yte, fit_predict(Xtr, ytr, Xte, cats, seed=seed, pre=pre_all))

    # leak 2: SMOTE on ALL rows before splitting (test patients' synthetic copies reach training)
    pre_tr = make_preprocessor(list(FEATURES), cats).fit(Xtr)
    Zall = pre_tr.transform(Xall)
    Zs, ys = smote(Zall, yall, np.random.RandomState(seed))
    sel = SelectKBest(f_classif, k=K_FEATURES).fit(Zs, ys)
    clf = LogisticRegression(max_iter=3000).fit(sel.transform(Zs), ys)
    out["leak_smote"] = roc_auc_score(yte, clf.predict_proba(sel.transform(pre_tr.transform(Xte)))[:, 1])

    # leak 3: feature selection with labels from ALL rows, then honest fit
    pre_all2 = make_preprocessor(list(FEATURES), cats).fit(Xall)
    Zall2 = pre_all2.transform(Xall)
    sel_all = SelectKBest(f_classif, k=K_FEATURES).fit(Zall2, yall)
    mask = sel_all.get_support()
    out["leak_selection"] = roc_auc_score(
        yte, fit_predict(Xtr, ytr, Xte, cats, seed=seed, pre=pre_all2, sel_mask=mask))

    # leak 4: pick C by test-fold AUROC
    best = max(roc_auc_score(yte, fit_predict(Xtr, ytr, Xte, cats, C=C, seed=seed)) for C in C_GRID)
    out["leak_test_tuning"] = best
    return out


def paired_test(a, b):
    """Nadeau-Bengio corrected resampled t-test on paired fold scores."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    n_test_over_train = 1 / 4   # 5-fold: test/train size ratio
    t = d.mean() / np.sqrt((1 / J + n_test_over_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/uci")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/ablation_v2")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_uci(a.data_dir, SITES)
    cats = schema_categories(df)
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()

    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats,
                                          random_state=a.seed).split(X, y))
    keys = ["correct", "correct_smote", "leak_scaling", "leak_smote", "leak_selection", "leak_test_tuning"]
    scores = {k: [] for k in keys}
    for i, (tr, te) in enumerate(splits):
        res = run_fold(X.iloc[tr], y[tr], X.iloc[te], y[te], X, y, cats, a.seed)
        for k in keys:
            scores[k].append(res[k])
        print(f"fold {i + 1}/{len(splits)} done", flush=True)

    rows = []
    for k in keys:
        rows.append(dict(protocol=k, mean_auroc=round(float(np.mean(scores[k])), 4),
                         sd=round(float(np.std(scores[k])), 4)))
    res_df = pd.DataFrame(rows)
    # compare each leak against its matching correct baseline
    pairs = {"leak_scaling": "correct", "leak_smote": "correct_smote",
             "leak_selection": "correct", "leak_test_tuning": "correct"}
    infl, tvals, pvals = [], [], []
    for k in keys:
        if k in pairs:
            base = pairs[k]
            infl.append(round(np.mean(scores[k]) - np.mean(scores[base]), 4))
            t, p = paired_test(scores[k], scores[base])
            tvals.append(round(t, 2)); pvals.append(round(p, 4))
        else:
            infl.append(np.nan); tvals.append(np.nan); pvals.append(np.nan)
    res_df["inflation_vs_matched_baseline"] = infl
    res_df["t"] = tvals
    res_df["p_paired"] = pvals
    res_df.to_csv(out / "ablation_v2_results.csv", index=False)
    print("\n=== Leakage ablation v2 (LR, same features, repeated 5-fold) ===")
    print(res_df.to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
