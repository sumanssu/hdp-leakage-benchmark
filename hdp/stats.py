"""Statistical comparison of models.

* Plain paired t-tests on CV folds are biased (folds share training data).
  Use the Nadeau-Bengio corrected resampled t-test, with Holm correction.
* For comparison ACROSS datasets / sites use Friedman + Nemenyi critical difference.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def corrected_resampled_ttest(a, b, n_train: float, n_test: float):
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pairwise_table(scores: dict, n_train: float, n_test: float, reference: str | None = None):
    """scores: {model: array of per-fold metric, aligned across models (same splits)}."""
    pairs = ([(m, reference) for m in scores if m != reference] if reference
             else list(itertools.combinations(scores, 2)))
    rows = []
    for a, b in pairs:
        t, p = corrected_resampled_ttest(scores[a], scores[b], n_train, n_test)
        rows.append(dict(model_a=a, model_b=b,
                         mean_diff=float(np.mean(scores[a]) - np.mean(scores[b])), t=t, p_raw=p))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p_raw"].values)
    return df


def friedman_nemenyi(matrix: np.ndarray, names: list[str], alpha: float = 0.05):
    """matrix: (N blocks x k models), higher = better. Blocks = datasets or held-out sites."""
    N, k = matrix.shape
    ranks = np.vstack([stats.rankdata(-row) for row in matrix])
    avg = ranks.mean(axis=0)
    stat, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(k)])
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * N)))
    return pd.DataFrame({"model": names, "avg_rank": avg}).sort_values("avg_rank"), float(stat), float(p), cd
