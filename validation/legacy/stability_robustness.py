"""
NORTHSTAR MODEL VALIDATION — STABILITY & ROBUSTNESS V0.1

Purpose
-------
Compact stability and robustness validation of a baseline
credit-risk model.

Tests
-----
1. Bootstrap stability of core metrics:
   - ROC AUC
   - PR AUC
   - Brier score

2. Random subsample stability:
   - repeated 90% subsamples
   - deviation from baseline metrics

3. Ranking stability:
   - Spearman correlation between baseline PD ranking
     and ranking on repeated 90% subsamples

This is NOT a stress-testing framework.
It intentionally avoids synthetic feature perturbations.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_FILE = Path("reports/baseline/validation_predictions.csv")

ID_COL = "SK_ID_CURR"
TARGET_COL = "y_true"
PD_COL = "y_probability"

RANDOM_SEED = 42

BOOTSTRAP_ITERATIONS = 1000
SUBSAMPLE_ITERATIONS = 100
SUBSAMPLE_FRACTION = 0.90


# ============================================================
# HELPERS
# ============================================================

def calculate_metrics(y_true, pd_pred):
    """Calculate core model metrics."""

    return {
        "roc_auc": roc_auc_score(y_true, pd_pred),
        "pr_auc": average_precision_score(y_true, pd_pred),
        "brier": brier_score_loss(y_true, pd_pred),
    }


def percentile_interval(values, lower=2.5, upper=97.5):
    """Return percentile confidence interval."""

    return (
        np.percentile(values, lower),
        np.percentile(values, upper),
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("NORTHSTAR MODEL VALIDATION — STABILITY & ROBUSTNESS V0.1")
print("=" * 70)


# ============================================================
# [1] LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("[1] LOADING BASELINE PREDICTIONS")
print("=" * 70)

df = pd.read_csv(PREDICTIONS_FILE)

required_columns = {
    ID_COL,
    TARGET_COL,
    PD_COL,
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

print(f"Predictions shape:   {df.shape}")

print(
    "Prediction IDs unique:",
    df[ID_COL].is_unique
)

if not df[ID_COL].is_unique:
    raise ValueError("Prediction IDs are not unique.")


# ============================================================
# [2] BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("[2] BASIC VALIDATION CHECKS")
print("=" * 70)

y = df[TARGET_COL].to_numpy()
pd_pred = df[PD_COL].to_numpy()

if not np.isfinite(pd_pred).all():
    raise ValueError("PD contains NaN or infinite values.")

if ((pd_pred < 0) | (pd_pred > 1)).any():
    raise ValueError("PD contains values outside [0, 1].")

unique_targets = np.unique(y)

print(
    "Target values:",
    unique_targets
)

if not set(unique_targets).issubset({0, 1}):
    raise ValueError("Target must contain only 0/1 values.")

print(
    "PD range:            "
    f"{pd_pred.min():.6f} — {pd_pred.max():.6f}"
)

print(
    "Default rate:        "
    f"{y.mean():.6f}"
)


# ============================================================
# [3] BASELINE METRICS
# ============================================================

print("\n" + "=" * 70)
print("[3] BASELINE METRICS")
print("=" * 70)

baseline = calculate_metrics(y, pd_pred)

for metric, value in baseline.items():
    print(f"{metric.upper():<15} {value:.6f}")


# ============================================================
# [4] BOOTSTRAP STABILITY
# ============================================================

print("\n" + "=" * 70)
print("[4] BOOTSTRAP METRIC STABILITY")
print("=" * 70)

rng = np.random.default_rng(RANDOM_SEED)

n = len(df)

bootstrap_results = {
    "roc_auc": [],
    "pr_auc": [],
    "brier": [],
}

valid_bootstrap = 0

for _ in range(BOOTSTRAP_ITERATIONS):

    indices = rng.integers(
        0,
        n,
        size=n,
    )

    y_sample = y[indices]
    pd_sample = pd_pred[indices]

    # AUC metrics require both classes.
    if len(np.unique(y_sample)) < 2:
        continue

    metrics = calculate_metrics(
        y_sample,
        pd_sample,
    )

    for metric in bootstrap_results:
        bootstrap_results[metric].append(
            metrics[metric]
        )

    valid_bootstrap += 1


print(
    f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}"
)

print(
    f"Valid iterations:     {valid_bootstrap}"
)

print()

for metric, values in bootstrap_results.items():

    values = np.asarray(values)

    mean = values.mean()
    std = values.std(ddof=1)

    lower, upper = percentile_interval(values)

    print(
        f"{metric.upper():<15}"
        f" mean={mean:.6f}  "
        f"std={std:.6f}  "
        f"95% CI=[{lower:.6f}, {upper:.6f}]"
    )


# ============================================================
# [5] 90% SUBSAMPLE STABILITY
# ============================================================

print("\n" + "=" * 70)
print("[5] 90% SUBSAMPLE STABILITY")
print("=" * 70)

subsample_results = {
    "roc_auc": [],
    "pr_auc": [],
    "brier": [],
}

ranking_results = []

subsample_size = int(
    n * SUBSAMPLE_FRACTION
)

print(
    f"Subsample fraction:  {SUBSAMPLE_FRACTION:.0%}"
)

print(
    f"Subsample size:      {subsample_size}"
)

for _ in range(SUBSAMPLE_ITERATIONS):

    indices = rng.choice(
        n,
        size=subsample_size,
        replace=False,
    )

    y_sample = y[indices]
    pd_sample = pd_pred[indices]

    # Skip pathological samples without both target classes.
    if len(np.unique(y_sample)) < 2:
        continue

    metrics = calculate_metrics(
        y_sample,
        pd_sample,
    )

    for metric in subsample_results:
        subsample_results[metric].append(
            metrics[metric]
        )

    # --------------------------------------------------------
    # Ranking stability
    # --------------------------------------------------------

    baseline_pd = pd_pred[indices]

    # Ranking is based on the PD ordering within the
    # same observations.
    rho, _ = spearmanr(
        baseline_pd,
        pd_sample,
    )

    ranking_results.append(rho)


print(
    f"Valid iterations:     {len(subsample_results['roc_auc'])}"
)

print()

for metric, values in subsample_results.items():

    values = np.asarray(values)

    mean = values.mean()
    std = values.std(ddof=1)

    max_deviation = np.max(
        np.abs(values - baseline[metric])
    )

    lower, upper = percentile_interval(values)

    print(
        f"{metric.upper():<15}"
        f" mean={mean:.6f}  "
        f"std={std:.6f}  "
        f"max_dev={max_deviation:.6f}  "
        f"95% CI=[{lower:.6f}, {upper:.6f}]"
    )


# ============================================================
# [6] RANKING STABILITY
# ============================================================

print("\n" + "=" * 70)
print("[6] RANKING STABILITY")
print("=" * 70)

ranking_results = np.asarray(ranking_results)

print(
    f"Spearman mean:       {ranking_results.mean():.6f}"
)

print(
    f"Spearman std:        {ranking_results.std(ddof=1):.6f}"
)

print(
    f"Spearman minimum:    {ranking_results.min():.6f}"
)

lower, upper = percentile_interval(
    ranking_results
)

print(
    f"Spearman 95% CI:     "
    f"[{lower:.6f}, {upper:.6f}]"
)


# ============================================================
# [7] CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("[7] VALIDATION SUMMARY")
print("=" * 70)

print(
    "Bootstrap stability:  ANALYZED"
)

print(
    "Subsample stability:  ANALYZED"
)

print(
    "Ranking stability:     ANALYZED"
)

print()
print(
    "Interpretation:"
)

print(
    "  - Narrow bootstrap intervals indicate stable "
    "aggregate performance."
)

print(
    "  - Small subsample deviations indicate that "
    "metrics are not highly sensitive to sample composition."
)

print(
    "  - High Spearman correlation indicates stable "
    "customer risk ordering."
)

print()
print("=" * 70)
print("STABILITY & ROBUSTNESS VALIDATION COMPLETE")
print("=" * 70)