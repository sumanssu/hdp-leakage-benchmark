"""Minimal SMOTE (dependency-free). Only ever call this on TRAINING data."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(X: np.ndarray, y: np.ndarray, rng: np.random.RandomState, k: int = 5):
    """Oversample the minority class to parity by interpolating between minority
    neighbours. Note: interpolating one-hot columns yields fractional values (a known
    SMOTE limitation; SMOTE-NC is the alternative)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    minority = classes[np.argmin(counts)]
    n_new = int(counts.max() - counts.min())
    Xm = np.asarray(X)[y == minority]
    if n_new == 0 or len(Xm) < 2:
        return X, y
    k = min(k, len(Xm) - 1)
    neigh = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.randint(0, len(Xm), n_new)
    nb = neigh[base, rng.randint(0, k, n_new)]
    lam = rng.rand(n_new, 1)
    X_new = Xm[base] + lam * (Xm[nb] - Xm[base])
    return np.vstack([X, X_new]), np.concatenate([y, np.full(n_new, minority)])
