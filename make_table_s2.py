#!/usr/bin/env python
"""Build Supplementary Table S2: per-site class counts and calibration for LOSO.
Reads loso_class_counts.csv and loso_calibration.csv; writes a manuscript-ready CSV
and a Markdown version."""
from pathlib import Path

import pandas as pd

out = Path("results/loso_saved")
counts = pd.read_csv(out / "loso_class_counts.csv")
cal = pd.read_csv(out / "loso_calibration.csv")

merged = cal.merge(counts, on="site")
merged["prevalence"] = (merged["positives"] / merged["n"]).round(2)
table = merged[["site", "n", "positives", "negatives", "model", "mean_pred",
                "observed", "cal_slope", "cal_intercept", "brier"]]
table = table.rename(columns={"mean_pred": "mean_predicted", "observed": "observed_prev",
                              "cal_slope": "slope", "cal_intercept": "intercept"})
table.to_csv(out / "table_S2.csv", index=False)
with open(out / "table_S2.md", "w") as f:
    f.write(table.to_markdown(index=False))
print(table.to_string(index=False))
print(f"\nSaved to {out}/table_S2.csv and table_S2.md")
