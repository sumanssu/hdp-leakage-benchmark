#!/usr/bin/env python
"""Leave-one-site-out (LOSO) evaluation for RQ3: train on 3 UCI sites, test on the 4th."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from hdp.data import FEATURES, load_uci, schema_categories
from hdp.metrics import compute_metrics
from hdp.models import make_search

SITES = ["cleveland", "hungarian", "switzerland", "va"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/uci")
    ap.add_argument("--models", default="LR,RF,GB,SVM,KNN,NB,DT,LR_clinical")
    ap.add_argument("--n-iter", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/loso")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = load_uci(a.data_dir, SITES)
    cats = schema_categories(df)
    rows = []
    for test_site in SITES:
        tr = df[df.site != test_site]
        te = df[df.site == test_site]
        for name in a.models.split(","):
            search = make_search(name, cats, a.seed, n_iter=a.n_iter)
            search.fit(tr[FEATURES], tr["target"].to_numpy())
            p = search.predict_proba(te[FEATURES])[:, 1]
            y = te["target"].to_numpy()
            m = compute_metrics(y, p)
            rows.append(dict(test_site=test_site, model=name, n_test=len(te),
                             prevalence=y.mean(), **m))
            print(f"[{test_site}] {name}: AUROC={m['auroc']:.3f}  cal_slope={m['cal_slope']:.2f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "loso_results.csv", index=False)
    summary = (res.groupby("model")[["auroc", "brier", "cal_slope"]]
                  .agg(["mean", "min", "max"]).round(3))
    summary.to_csv(out / "loso_summary.csv")
    print("\n=== LOSO summary (mean / min / max across the 4 held-out sites) ===")
    print(summary.to_string())
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
