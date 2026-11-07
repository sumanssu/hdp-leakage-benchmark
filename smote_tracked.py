#!/usr/bin/env python
"""SMOTE with parent tracking (RQ1 mechanism check).

Returns the synthetic points AND, for each one, the indices of the two original
rows it was interpolated between (its base and its neighbour). This lets us count
exactly how many synthetic points have a test-fold patient as a parent.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote_tracked(X, y, rng, k=5):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    n_new = int(counts.max() - counts.min())
    min_idx = np.flatnonzero(y == minority)
    Xm = np.asarray(X)[min_idx]
    k = min(k, len(Xm) - 1)
    neigh = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.randint(0, len(Xm), n_new)
    nb = neigh[base, rng.randint(0, k, n_new)]
    lam = rng.rand(n_new, 1)
    X_new = Xm[base] + lam * (Xm[nb] - Xm[base])
    parents = np.column_stack([min_idx[base], min_idx[nb]])   # original row indices
    return (np.vstack([np.asarray(X), X_new]),
            np.concatenate([y, np.full(n_new, minority)]),
            parents)
