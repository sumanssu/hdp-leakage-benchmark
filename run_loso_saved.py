#!/usr/bin/env python
"""LOSO with saved predictions (RQ3). Same models as run_loso.py, but writes
each held-out site's predicted probabilities so bootstrap CIs can be computed."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    a = ap.parse_args()
    out = Path("results/loso_saved"); out.mkdir(parents=True, exist_ok=True)

    df = load_uci(a.data_dir, SITES)
    cats = schema_categories(df)
    rows, preds = [], {}
    for test_site in SITES:
        tr = df[df.site != test_site]
        te = df[df.site == test_site]
        y = te["target"].to_numpy()
        preds[f"{test_site}__y"] = y
        for name in a.models.split(","):
            search = make_search(name, cats, a.seed, n_iter=a.n_iter)
            search.fit(tr[FEATURES], tr["target"].to_numpy())
            p = search.predict_proba(te[FEATURES])[:, 1]
            preds[f"{test_site}__{name}"] = p
            m = compute_metrics(y, p)
            rows.append(dict(test_site=test_site, model=name, n_test=len(te), **m))
            print(f"[{test_site}] {name}: AUROC={m['auroc']:.3f}", flush=True)
    np.savez_compressed(out / "loso_predictions.npz", **preds)
    pd.DataFrame(rows).to_csv(out / "loso_results.csv", index=False)
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
