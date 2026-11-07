"""Data loading, cleaning, schema and auditing for UCI-style heart disease data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL
# Classical risk-factor baseline (Framingham-style inputs available in UCI)
CLINICAL_BASELINE = ["age", "sex", "trestbps", "chol", "fbs"]

UCI_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
ALL_SITES = tuple(UCI_FILES)


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def clean_uci(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning that does NOT learn anything from the data.

    In the UCI files, chol == 0 and trestbps == 0 are physiologically impossible
    and encode *missing* values (mostly in the Switzerland / VA sites).
    """
    df = df.copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["chol"] == 0, "chol"] = np.nan
    df.loc[df["trestbps"] == 0, "trestbps"] = np.nan
    return df


def load_uci(data_dir: str | Path, sites: list[str] | None = None) -> pd.DataFrame:
    """Load the four processed UCI files. Target = (num > 0)."""
    data_dir = Path(data_dir)
    sites = list(sites) if sites else list(ALL_SITES)
    frames = []
    for site in sites:
        path = data_dir / UCI_FILES[site]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Download the processed.*.data files from the UCI "
                f"Heart Disease repository into {data_dir} (see README)."
            )
        d = pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values="?")
        d["site"] = site
        frames.append(d)
    df = clean_uci(pd.concat(frames, ignore_index=True))
    df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
    return df.drop(columns="num").reset_index(drop=True)


def schema_categories(df: pd.DataFrame) -> dict[str, list]:
    """Category levels per categorical column. This is a *schema* (data dictionary),
    not a learned statistic, so using the full dataset here is not leakage. It keeps
    the one-hot column layout identical in every fold."""
    return {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}


def subsample_prevalence(df: pd.DataFrame, prevalence: float, seed: int = 0) -> pd.DataFrame:
    """Down-sample positives to create an imbalanced scenario (e.g. prevalence=0.15)."""
    rng = np.random.RandomState(seed)
    pos, neg = df[df.target == 1], df[df.target == 0]
    n_pos = int(round(prevalence * len(neg) / (1 - prevalence)))
    pos = pos.iloc[rng.permutation(len(pos))[: min(n_pos, len(pos))]]
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Auditing (duplicates / overlap between datasets)
# --------------------------------------------------------------------------- #
def _row_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    vals = df[cols].round(4).to_numpy(dtype=object)
    return pd.Series(["|".join("NA" if pd.isna(v) else str(v) for v in row) for row in vals], index=df.index)


def audit_duplicates(df: pd.DataFrame, cols: list[str] = FEATURES, site_col: str = "site"):
    """Exact duplicate feature-rows within each site and across sites."""
    d = df.assign(_key=_row_key(df, cols))
    rows = [
        {"site": s, "n": len(g), "exact_duplicates_within_site": int(g["_key"].duplicated().sum())}
        for s, g in d.groupby(site_col)
    ]
    cross = int((d.groupby("_key")[site_col].nunique() > 1).sum())
    return pd.DataFrame(rows), cross


def audit_overlap(reference: pd.DataFrame, other: pd.DataFrame, cols: list[str]) -> dict:
    """How many rows of `other` also appear in `reference` (same columns/units required).
    Use this to check whether a 'new' Kaggle dataset is really a copy/merge of UCI."""
    ref_keys = set(_row_key(reference, cols))
    hit = _row_key(other, cols).isin(ref_keys)
    return {"n_other": len(other), "n_in_reference": int(hit.sum()), "fraction": float(hit.mean())}


# --------------------------------------------------------------------------- #
# Synthetic data (only for testing the code offline - NEVER report results on it)
# --------------------------------------------------------------------------- #
def make_synthetic(sites=("cleveland",), seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    sizes = {"cleveland": 303, "hungarian": 294, "switzerland": 123, "va": 200}
    frames = []
    for i, site in enumerate(sites):
        n = sizes.get(site, 200)
        age = rng.normal(54 + 2 * i, 9, n).clip(29, 77).round()
        sex = (rng.rand(n) < 0.68).astype(float)
        cp = rng.choice([1, 2, 3, 4], n, p=[.08, .17, .28, .47]).astype(float)
        trestbps = rng.normal(131, 17, n).clip(94, 200).round()
        chol = rng.normal(246, 52, n).clip(126, 564).round()
        fbs = (rng.rand(n) < 0.15).astype(float)
        restecg = rng.choice([0, 1, 2], n, p=[.5, .01, .49]).astype(float)
        thalach = rng.normal(150 - 3 * i, 23, n).clip(71, 202).round()
        exang = (rng.rand(n) < 0.33).astype(float)
        oldpeak = rng.exponential(1.0, n).clip(0, 6).round(1)
        slope = rng.choice([1, 2, 3], n, p=[.47, .46, .07]).astype(float)
        ca = rng.choice([0, 1, 2, 3], n, p=[.59, .22, .13, .06]).astype(float)
        thal = rng.choice([3, 6, 7], n, p=[.55, .06, .39]).astype(float)
        z = (-3.0 + 0.03 * (age - 54) + 0.8 * sex + 0.9 * (cp == 4) - 0.03 * (thalach - 150)
             + 0.8 * exang + 0.5 * oldpeak + 0.7 * ca + 0.9 * (thal == 7) + 0.15 * i
             + rng.normal(0, 1.0, n))
        target = (rng.rand(n) < 1 / (1 + np.exp(-z))).astype(int)
        d = pd.DataFrame(dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs,
                              restecg=restecg, thalach=thalach, exang=exang, oldpeak=oldpeak,
                              slope=slope, ca=ca, thal=thal, site=site, target=target))
        for c in ("ca", "thal"):
            d.loc[rng.rand(n) < 0.02, c] = np.nan
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
