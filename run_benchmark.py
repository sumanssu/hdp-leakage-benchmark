#!/usr/bin/env python
"""Leakage-free nested-CV benchmark (RQ2, RQ3).

Example:
    python run_benchmark.py --data-dir data/uci --sites cleveland --calibrate sigmoid
    python run_benchmark.py --demo --quick                     # smoke test, synthetic data
    python run_benchmark.py --demo --quick --permute-labels    # negative control: AUROC must be ~0.5
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from hdp.data import (ALL_SITES, FEATURES, audit_duplicates, load_uci, make_synthetic,
                      schema_categories, subsample_prevalence)
from hdp.metrics import bootstrap_ci, compute_metrics, reliability_plot
from hdp.models import make_search, model_zoo
from hdp.stats import pairwise_table


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/uci")
    ap.add_argument("--sites", default="cleveland", help="comma list or 'all'")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (code testing only)")
    ap.add_argument("--models", default="", help="comma list; default = all")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=12, help="tuning budget per model")
    ap.add_argument("--calibrate", choices=["none", "sigmoid", "isotonic"], default="none")
    ap.add_argument("--pos-fraction", type=float, default=0.0, help="subsample positives to this prevalence")
    ap.add_argument("--permute-labels", action="store_true", help="negative control")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--quick", action="store_true", help="repeats=2, n_iter=6, n_boot=200")
    a = ap.parse_args()
    if a.quick:
        a.repeats, a.n_iter, a.n_boot = 2, 6, 200
    return a


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    sites = list(ALL_SITES) if args.sites == "all" else args.sites.split(",")
    df = make_synthetic(tuple(sites), args.seed) if args.demo else load_uci(args.data_dir, sites)
    if args.pos_fraction:
        df = subsample_prevalence(df, args.pos_fraction, args.seed)
    dup_summary, dup_cross = audit_duplicates(df)
    print("Duplicate audit:\n", dup_summary.to_string(index=False), f"\ncross-site duplicates: {dup_cross}\n")

    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].to_numpy()
    if args.permute_labels:
        y = np.random.RandomState(args.seed).permutation(y)
        print(">>> NEGATIVE CONTROL: labels permuted. Every AUROC should be ~0.5.\n")
    cats = schema_categories(df)
    n, K, R = len(y), args.outer_splits, args.repeats
    print(f"n={n}, prevalence={y.mean():.3f}, features={len(FEATURES)}, outer={R}x{K}, tuning n_iter={args.n_iter}\n")

    names = [m for m in args.models.split(",") if m] or list(model_zoo(args.seed))
    splits = list(RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=args.seed).split(X, y))

    oof_all, fold_rows = {}, []
    for name in names:
        t0 = time.time()
        labels = [name] + ([f"{name}+{args.calibrate}"] if args.calibrate != "none" else [])
        oof = {lab: np.full((R, n), np.nan) for lab in labels}
        for i, (tr, te) in enumerate(splits):
            rep, fold = divmod(i, K)
            search = make_search(name, cats, args.seed, args.inner_splits, args.n_iter, args.n_jobs)
            search.fit(X.iloc[tr], y[tr])
            preds = {name: search.predict_proba(X.iloc[te])[:, 1]}
            if args.calibrate != "none":
                cal = CalibratedClassifierCV(clone(search.best_estimator_), method=args.calibrate, cv=3)
                cal.fit(X.iloc[tr], y[tr])
                preds[labels[1]] = cal.predict_proba(X.iloc[te])[:, 1]
            for lab, p in preds.items():
                oof[lab][rep, te] = p
                fold_rows.append(dict(model=lab, repeat=rep, fold=fold, n_train=len(tr), n_test=len(te),
                                      auroc=roc_auc_score(y[te], p), auprc=average_precision_score(y[te], p),
                                      brier=brier_score_loss(y[te], p), best_params=str(search.best_params_)))
        oof_all.update(oof)
        print(f"[{name}] done in {time.time() - t0:.0f}s  AUROC(pooled, rep0)={roc_auc_score(y, oof[name][0]):.3f}")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    np.savez_compressed(out / "oof_predictions.npz", y=y, **{k.replace("+", "_"): v for k, v in oof_all.items()})

    # ---- summary with bootstrap CIs
    rows = []
    for lab, oof in oof_all.items():
        per_rep = pd.DataFrame([compute_metrics(y, oof[r]) for r in range(R)]).mean()
        ci = bootstrap_ci(y, oof, args.n_boot, args.seed)
        row = {"model": lab}
        for m, (pt, lo, hi) in ci.items():
            row.update({m: pt, f"{m}_lo": lo, f"{m}_hi": hi})
        row.update({k: per_rep[k] for k in ("cal_slope", "cal_intercept", "mean_pred_minus_prev", "accuracy")})
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    summary.to_csv(out / "summary.csv", index=False)

    # ---- statistics on per-fold AUROC (same splits for all models => aligned)
    scores = {lab: g.sort_values(["repeat", "fold"])["auroc"].to_numpy() for lab, g in folds.groupby("model")}
    n_train, n_test = folds.n_train.mean(), folds.n_test.mean()
    pairwise_table(scores, n_train, n_test).to_csv(out / "tests_all_pairs.csv", index=False)
    if "LR_clinical" in scores:
        pairwise_table(scores, n_train, n_test, reference="LR_clinical").to_csv(out / "tests_vs_clinical_baseline.csv", index=False)

    reliability_plot(y, {k: v for k, v in oof_all.items() if k in names[:0] + [k for k in oof_all]}, str(out / "reliability.png"))

    (out / "run_config.json").write_text(json.dumps(dict(
        args=vars(args), n=n, prevalence=float(y.mean()), python=platform.python_version(),
        sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__), indent=2))

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY (pooled out-of-fold, mean over repeats, 95% bootstrap CI) ===")
    show = summary[["model", "auroc", "auroc_lo", "auroc_hi", "auprc", "brier", "sens_at_90spec",
                    "cal_slope", "cal_intercept", "accuracy"]]
    print(show.round(3).to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
