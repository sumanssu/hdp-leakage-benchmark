"""Preprocessing + model zoo. Every learned step lives INSIDE a Pipeline, so it is
refit on the training portion of each CV split only (no leakage)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import BINARY, CATEGORICAL, CLINICAL_BASELINE, FEATURES, NUMERIC


def make_preprocessor(columns, categories) -> ColumnTransformer:
    num = [c for c in columns if c in NUMERIC]
    binary = [c for c in columns if c in BINARY]
    cat = [c for c in columns if c in CATEGORICAL]
    parts = []
    if num:
        parts.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("sc", StandardScaler())]), num))
    if binary:
        parts.append(("bin", SimpleImputer(strategy="most_frequent", keep_empty_features=True), binary))
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("oh", OneHotEncoder(categories=[categories[c] for c in cat],
                                 handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


@dataclass
class ModelSpec:
    estimator: object
    grid: dict
    features: list | None = None  # None -> all features


def model_zoo(seed: int = 0) -> dict[str, ModelSpec]:
    zoo = {
        "LR_clinical": ModelSpec(LogisticRegression(max_iter=2000),
                                 {"clf__C": [0.01, 0.1, 1, 10]}, CLINICAL_BASELINE),
        "LR": ModelSpec(LogisticRegression(max_iter=2000), {"clf__C": [0.01, 0.1, 1, 10]}),
        "DT": ModelSpec(DecisionTreeClassifier(random_state=seed),
                        {"clf__max_depth": [2, 3, 4, 5, None], "clf__min_samples_leaf": [1, 5, 10]}),
        "KNN": ModelSpec(KNeighborsClassifier(),
                         {"clf__n_neighbors": [3, 5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
        "NB": ModelSpec(GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2]}),
        "SVM": ModelSpec(SVC(probability=True, random_state=seed, cache_size=1000),
                         {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
        "RF": ModelSpec(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
                        {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3, 5],
                         "clf__max_features": ["sqrt", 0.5]}),
        "GB": ModelSpec(HistGradientBoostingClassifier(random_state=seed),
                        {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3, None],
                         "clf__min_samples_leaf": [10, 20], "clf__l2_regularization": [0.0, 1.0]}),
        "MLP": ModelSpec(MLPClassifier(max_iter=2000, random_state=seed),
                         {"clf__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                          "clf__alpha": [1e-4, 1e-2, 1e-1]}),
    }
    try:  # optional
        from xgboost import XGBClassifier
        zoo["XGB"] = ModelSpec(
            XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0),
            {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 3, 4],
             "clf__learning_rate": [0.03, 0.1], "clf__subsample": [0.7, 1.0]})
    except ImportError:
        pass
    return zoo


def build_pipeline(name: str, categories: dict, seed: int = 0):
    spec = model_zoo(seed)[name]
    cols = spec.features or FEATURES
    pipe = Pipeline([("pre", make_preprocessor(cols, categories)), ("clf", clone(spec.estimator))])
    return pipe, spec.grid


def make_search(name, categories, seed=0, inner_splits=3, n_iter=12, n_jobs=1, cv=None):
    """Inner-loop hyper-parameter search. The SAME budget (n_iter) is given to every model;
    grids smaller than n_iter are searched exhaustively."""
    pipe, grid = build_pipeline(name, categories, seed)
    cv = cv or StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    warnings.filterwarnings("ignore", message="The total space of parameters")
    return RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="roc_auc", cv=cv,
                              refit=True, random_state=seed, n_jobs=n_jobs, error_score=float("nan"))
