"""Discrimination, calibration, bootstrap CIs and reliability plots."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score, roc_curve)

EPS = 1e-6
FAST = ("auroc", "auprc", "brier", "sens_at_90spec")


def sens_at_spec(y, p, spec: float = 0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    mask = fpr <= (1 - spec)
    return float(tpr[mask].max()) if mask.any() else 0.0


def _fast(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p),
                     brier_score_loss(y, p), sens_at_spec(y, p)])


def compute_metrics(y, p) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    if len(np.unique(y)) < 2:
        return {k: np.nan for k in (*FAST, "cal_slope", "cal_intercept", "mean_pred_minus_prev", "accuracy")}
    auroc, auprc, brier, sens = _fast(y, p)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return dict(auroc=auroc, auprc=auprc, brier=brier, sens_at_90spec=sens,
                cal_slope=float(lr.coef_[0, 0]),          # ideal = 1
                cal_intercept=float(lr.intercept_[0]),    # ideal = 0
                mean_pred_minus_prev=float(p.mean() - y.mean()),  # calibration-in-the-large, ideal = 0
                accuracy=float(((p >= 0.5) == y).mean()))  # reported only for literature comparison


def bootstrap_ci(y, oof, n_boot: int = 1000, seed: int = 0) -> dict:
    """Patient-level percentile bootstrap. `oof` has shape (n_repeats, n); the metric
    is averaged over repeats inside every bootstrap resample."""
    y = np.asarray(y)
    oof = np.atleast_2d(np.clip(oof, EPS, 1 - EPS))
    n = len(y)
    rng = np.random.RandomState(seed)

    def stat(idx):
        return np.mean([_fast(y[idx], oof[r, idx]) for r in range(oof.shape[0])], axis=0)

    point = stat(np.arange(n))
    boots = []
    while len(boots) < n_boot:
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(stat(idx))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return {m: (point[i], lo[i], hi[i]) for i, m in enumerate(FAST)}


def reliability_plot(y, oof_by_model: dict, path: str, n_bins: int = 8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, oof in oof_by_model.items():
        oof = np.atleast_2d(oof)
        p = oof.ravel()
        yy = np.tile(np.asarray(y), oof.shape[0])
        frac, mean_p = calibration_curve(yy, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=3, lw=1.2, label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.legend(fontsize=7)
    ax.set_title("Reliability (pooled out-of-fold predictions)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
