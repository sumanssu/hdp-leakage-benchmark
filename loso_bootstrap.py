#!/usr/bin/env python
"""Per-site bootstrap CIs for LOSO AUROC (RQ3). Reads saved predictions; no retraining."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

SITES = ["cleveland", "hungarian", "switzerland", "va"]
MODELS = ["LR", "RF", "GB", "SVM", "KNN", "NB", "DT", "LR_clinical"]
N_BOOT = 2000
SEED = 42

src = Path("results/loso_saved/loso_predictions.npz")
data = np.load(src)
rng = np.random.RandomState(SEED)
rows = []
for site in SITES:
    y = data[f"{site}__y"]
    n = len(y)
    idx_all = [rng.randint(0, n, n) for _ in range(N_BOOT)]
    idx_all = [i for i in idx_all if len(np.unique(y[i])) == 2]
    for m in MODELS:
        p = data[f"{site}__{m}"]
        auc = roc_auc_score(y, p)
        br = brier_score_loss(y, p)
        boots = np.array([roc_auc_score(y[i], p[i]) for i in idx_all])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(site=site, n=n, prevalence=round(float(y.mean()), 3), model=m,
                         auroc=round(auc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                         brier=round(br, 3)))
res = pd.DataFrame(rows)
out = Path("results/loso_saved")
res.to_csv(out / "loso_site_cis.csv", index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
print(f"\nSaved to {out}/loso_site_cis.csv")
