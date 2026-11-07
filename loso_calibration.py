#!/usr/bin/env python
"""Per-site calibration for LOSO predictions (RQ3). Reads saved predictions; no retraining.
Also verifies negative-class counts for each held-out site."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

SITES = ["cleveland", "hungarian", "switzerland", "va"]
MODELS = ["LR", "RF", "GB", "SVM", "KNN", "NB", "DT", "LR_clinical"]
EPS = 1e-6

data = np.load(Path("results/loso_saved/loso_predictions.npz"))
rows, counts = [], []
for site in SITES:
    y = data[f"{site}__y"]
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    counts.append(dict(site=site, n=len(y), positives=n_pos, negatives=n_neg))
    for m in MODELS:
        p = np.clip(data[f"{site}__{m}"], EPS, 1 - EPS)
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
        rows.append(dict(site=site, model=m,
                         brier=round(brier_score_loss(y, p), 3),
                         cal_slope=round(float(lr.coef_[0, 0]), 2),
                         cal_intercept=round(float(lr.intercept_[0]), 2),
                         mean_pred=round(float(p.mean()), 3),
                         observed=round(float(y.mean()), 3)))
out = Path("results/loso_saved")
pd.DataFrame(counts).to_csv(out / "loso_class_counts.csv", index=False)
res = pd.DataFrame(rows)
res.to_csv(out / "loso_calibration.csv", index=False)
print("=== Class counts per held-out site ===")
print(pd.DataFrame(counts).to_string(index=False))
print("\n=== Per-site calibration ===")
print(res.to_string(index=False))
