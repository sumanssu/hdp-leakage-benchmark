#!/usr/bin/env python
"""Count synthetic points whose parents include test-fold patients (RQ1)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from hdp.data import FEATURES, load_uci, schema_categories
from hdp.models import make_preprocessor
from smote_tracked import smote_tracked

SITES = ["cleveland", "hungarian", "switzerland", "va"]

df = load_uci("data/uci", SITES)
cats = schema_categories(df)
X = df[FEATURES].reset_index(drop=True)
y = df["target"].to_numpy()
n = len(y)
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, y))

rows = []
for i, (tr, te) in enumerate(splits):
    pre = make_preprocessor(list(FEATURES), cats).fit(X.iloc[tr])
    Zall = pre.transform(X)
    Zs, ys, parents = smote_tracked(Zall, y, np.random.RandomState(42))
    n_syn = len(parents)
    test_set = np.zeros(n, bool)
    test_set[te] = True
    p_base_test = test_set[parents[:, 0]]
    p_nb_test = test_set[parents[:, 1]]
    any_test = (p_base_test | p_nb_test).sum()
    rows.append(dict(fold=i + 1, n_synthetic=n_syn,
                     synthetic_with_test_parent=int(any_test),
                     fraction_with_test_parent=round(float(any_test / n_syn), 3)))
res = pd.DataFrame(rows)
res.to_csv("results/mechanism_count.csv", index=False)
print(res.to_string(index=False))
print(f"\nMean fraction of synthetic points with a test-fold parent: "
      f"{res.fraction_with_test_parent.mean():.3f}")
