"""
NORTHSTAR MODEL VALIDATION
Feature Stability / Data Drift Analysis
Version: 0.1

Purpose
-------
Independent validation test for detecting feature distribution drift
between a reference (development) period and an OOT period.

Tests
-----
Numerical features:
    - PSI
    - KS statistic
    - KS p-value
    - Mean shift
    - Median shift
    - Missing-rate shift

Categorical features:
    - PSI
    - Chi-square statistic
    - Chi-square p-value
    - Missing-rate shift

Outputs
-------
1. Console summary
2. Detailed feature-level CSV report
3. Top drifting features printed to console

Important
---------
Drift does NOT automatically imply model deterioration.

This test identifies data distribution changes. Materiality and
impact on model performance should be assessed separately.

Author: Northstar Model Validation
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path("data/raw/application_train.csv")
PREDICTIONS_PATH = Path("data/raw/validation_predictions.csv")

OUTPUT_PATH = Path("reports/feature_stability_report.csv")


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
#
# These values should correspond to the temporal/OOT split established in
# temporal_check.py.
#
# IMPORTANT:
# Adjust these dates if temporal_check.py uses different boundaries.
#

DATE_COLUMN = "DAYS_BIRTH"

# DAYS_BIRTH is negative age in days and therefore NOT a real observation
# date. For the Home Credit dataset there is no application date column.
#
# Therefore v0.1 supports two modes:
#
#   1. Explicit period column, if one exists.
#   2. Index-based / externally defined split.
#
# For the current dataset, use ROW_SPLIT mode unless a derived temporal
# variable has been created.
#

SPLIT_MODE = "ROW"

REFERENCE_FRACTION = 0.80


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

ID_COLUMNS = [
    "SK_ID_CURR",
]

TARGET_COLUMN = "TARGET"


# Features with extremely high cardinality are treated as categorical only
# if their dtype is non-numeric. Numeric high-cardinality variables remain
# numerical.
#
# Categorical PSI is calculated using the union of categories from both
# samples.

MAX_CATEGORIES_FOR_CATEGORICAL = 100


# ---------------------------------------------------------------------------
# Drift thresholds
# ---------------------------------------------------------------------------
#
# PSI:
#   < 0.10       Stable
#   0.10 - 0.25  Watch
#   >= 0.25      Drift
#
# These are common practical thresholds, not universal regulatory limits.
#

PSI_STABLE_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.25


# KS:
# KS is reported but does not independently determine the final status
# in v0.1. PSI is used as the primary drift classification metric.

KS_ALPHA = 0.05


# Minimum observations required in both samples

MIN_SAMPLE_SIZE = 100


# Number of bins for numerical PSI

N_BINS = 10


# Small value preventing log(0)

EPSILON = 1e-6


# Number of top features displayed

TOP_N = 15


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def print_header(title: str) -> None:
    """Print a formatted section header."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def safe_divide(a, b):
    """Safe division."""

    if b == 0:
        return np.nan

    return a / b


# ============================================================================
# PSI
# ============================================================================


def calculate_psi(
    reference: pd.Series,
    oot: pd.Series,
    n_bins: int = N_BINS,
) -> float:
    """
    Calculate Population Stability Index.

    For numerical data, bins are defined using quantiles of the reference
    population.

    For categorical data, values are treated as categories.

    PSI = sum((actual_pct - expected_pct)
              * ln(actual_pct / expected_pct))
    """

    reference = pd.Series(reference).dropna()
    oot = pd.Series(oot).dropna()

    if len(reference) == 0 or len(oot) == 0:
        return np.nan

    # ------------------------------------------------------------------
    # Numerical PSI
    # ------------------------------------------------------------------

    if pd.api.types.is_numeric_dtype(reference):

        # If constant feature, there is no distributional information.
        if reference.nunique() <= 1:
            return 0.0

        try:
            quantiles = np.linspace(0, 1, n_bins + 1)

            breakpoints = np.unique(
                reference.quantile(quantiles).values
            )

            # Not enough unique breakpoints
            if len(breakpoints) < 3:
                return 0.0

            # Expand boundaries slightly
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf

            reference_bins = pd.cut(
                reference,
                bins=breakpoints,
                include_lowest=True,
            )

            oot_bins = pd.cut(
                oot,
                bins=breakpoints,
                include_lowest=True,
            )

            reference_dist = (
                reference_bins.value_counts(
                    normalize=True,
                    sort=False,
                )
                .values
            )

            oot_dist = (
                oot_bins.value_counts(
                    normalize=True,
                    sort=False,
                )
                .values
            )

        except Exception:
            return np.nan

    # ------------------------------------------------------------------
    # Categorical PSI
    # ------------------------------------------------------------------

    else:

        categories = sorted(
            set(reference.astype(str).unique())
            | set(oot.astype(str).unique())
        )

        reference_dist = (
            reference.astype(str)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
            .values
        )

        oot_dist = (
            oot.astype(str)
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
            .values
        )

    reference_dist = np.maximum(reference_dist, EPSILON)
    oot_dist = np.maximum(oot_dist, EPSILON)

    psi = np.sum(
        (oot_dist - reference_dist)
        * np.log(oot_dist / reference_dist)
    )

    return float(psi)


# ============================================================================
# NUMERICAL FEATURE TEST
# ============================================================================


def analyze_numeric_feature(
    reference: pd.Series,
    oot: pd.Series,
) -> dict:
    """Calculate stability statistics for a numerical feature."""

    reference_nonnull = reference.dropna()
    oot_nonnull = oot.dropna()

    if (
        len(reference_nonnull) < MIN_SAMPLE_SIZE
        or len(oot_nonnull) < MIN_SAMPLE_SIZE
    ):
        return {
            "psi": np.nan,
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "reference_mean": np.nan,
            "oot_mean": np.nan,
            "mean_relative_shift": np.nan,
            "reference_median": np.nan,
            "oot_median": np.nan,
            "median_relative_shift": np.nan,
        }

    psi = calculate_psi(
        reference_nonnull,
        oot_nonnull,
    )

    try:
        ks_stat, ks_pvalue = ks_2samp(
            reference_nonnull,
            oot_nonnull,
        )
    except Exception:
        ks_stat = np.nan
        ks_pvalue = np.nan

    reference_mean = reference_nonnull.mean()
    oot_mean = oot_nonnull.mean()

    reference_median = reference_nonnull.median()
    oot_median = oot_nonnull.median()

    if abs(reference_mean) > EPSILON:
        mean_relative_shift = (
            (oot_mean - reference_mean)
            / abs(reference_mean)
        )
    else:
        mean_relative_shift = np.nan

    if abs(reference_median) > EPSILON:
        median_relative_shift = (
            (oot_median - reference_median)
            / abs(reference_median)
        )
    else:
        median_relative_shift = np.nan

    return {
        "psi": psi,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "reference_mean": reference_mean,
        "oot_mean": oot_mean,
        "mean_relative_shift": mean_relative_shift,
        "reference_median": reference_median,
        "oot_median": oot_median,
        "median_relative_shift": median_relative_shift,
    }


# ============================================================================
# CATEGORICAL FEATURE TEST
# ============================================================================


def analyze_categorical_feature(
    reference: pd.Series,
    oot: pd.Series,
) -> dict:
    """Calculate stability statistics for a categorical feature."""

    reference_nonnull = reference.dropna().astype(str)
    oot_nonnull = oot.dropna().astype(str)

    if (
        len(reference_nonnull) < MIN_SAMPLE_SIZE
        or len(oot_nonnull) < MIN_SAMPLE_SIZE
    ):
        return {
            "psi": np.nan,
            "chi2_stat": np.nan,
            "chi2_pvalue": np.nan,
            "reference_mean": np.nan,
            "oot_mean": np.nan,
            "mean_relative_shift": np.nan,
            "reference_median": np.nan,
            "oot_median": np.nan,
            "median_relative_shift": np.nan,
        }

    psi = calculate_psi(
        reference_nonnull,
        oot_nonnull,
    )

    categories = sorted(
        set(reference_nonnull.unique())
        | set(oot_nonnull.unique())
    )

    reference_counts = (
        reference_nonnull
        .value_counts()
        .reindex(categories, fill_value=0)
    )

    oot_counts = (
        oot_nonnull
        .value_counts()
        .reindex(categories, fill_value=0)
    )

    contingency = np.vstack(
        [
            reference_counts.values,
            oot_counts.values,
        ]
    )

    try:

        chi2_stat, chi2_pvalue, _, _ = chi2_contingency(
            contingency
        )

    except Exception:

        chi2_stat = np.nan
        chi2_pvalue = np.nan

    return {
        "psi": psi,
        "chi2_stat": chi2_stat,
        "chi2_pvalue": chi2_pvalue,
        "reference_mean": np.nan,
        "oot_mean": np.nan,
        "mean_relative_shift": np.nan,
        "reference_median": np.nan,
        "oot_median": np.nan,
        "median_relative_shift": np.nan,
    }


# ============================================================================
# FEATURE TYPE DETECTION
# ============================================================================


def detect_feature_type(series: pd.Series) -> str:
    """
    Determine whether a feature should be treated as numerical or
    categorical.

    Numeric dtypes -> numerical
    Everything else -> categorical
    """

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


# ============================================================================
# DRIFT CLASSIFICATION
# ============================================================================


def classify_drift(psi: float) -> str:
    """Classify drift based on PSI."""

    if pd.isna(psi):
        return "INSUFFICIENT_DATA"

    if psi < PSI_STABLE_THRESHOLD:
        return "STABLE"

    if psi < PSI_DRIFT_THRESHOLD:
        return "WATCH"

    return "DRIFT"


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main():

    print_header(
        "NORTHSTAR MODEL VALIDATION — "
        "FEATURE STABILITY / DATA DRIFT V0.1"
    )

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================

    print_header("[1] LOADING DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape:       {df.shape}")

    # ========================================================================
    # 2. BASIC VALIDATION
    # ========================================================================

    print_header("[2] BASIC VALIDATION CHECKS")

    print(
        f"Target column present: "
        f"{TARGET_COLUMN in df.columns}"
    )

    print(
        f"ID columns present:    "
        f"{all(c in df.columns for c in ID_COLUMNS)}"
    )

    if TARGET_COLUMN in df.columns:

        print(
            f"Missing target:        "
            f"{df[TARGET_COLUMN].isna().sum()}"
        )

    # ========================================================================
    # 3. DEFINE REFERENCE / OOT SAMPLES
    # ========================================================================

    print_header("[3] DEFINING REFERENCE / OOT PERIODS")

    if SPLIT_MODE == "ROW":

        split_index = int(
            len(df) * REFERENCE_FRACTION
        )

        reference = df.iloc[:split_index].copy()
        oot = df.iloc[split_index:].copy()

        print(
            "Split mode:           ROW"
        )

        print(
            f"Reference fraction:   "
            f"{REFERENCE_FRACTION:.0%}"
        )

    else:

        raise ValueError(
            f"Unsupported SPLIT_MODE: {SPLIT_MODE}"
        )

    print(
        f"Reference rows:       {len(reference):,}"
    )

    print(
        f"OOT rows:             {len(oot):,}"
    )

    # ========================================================================
    # 4. FEATURE SELECTION
    # ========================================================================

    print_header("[4] PREPARING FEATURES")

    excluded_columns = (
        ID_COLUMNS
        + [TARGET_COLUMN]
    )

    features = [
        c
        for c in df.columns
        if c not in excluded_columns
    ]

    print(
        f"Total features:       {len(features)}"
    )

    print(
        f"Excluded columns:     {excluded_columns}"
    )

    # ========================================================================
    # 5. FEATURE ANALYSIS
    # ========================================================================

    print_header("[5] FEATURE STABILITY ANALYSIS")

    results = []

    for i, feature in enumerate(features, start=1):

        series = df[feature]

        feature_type = detect_feature_type(series)

        reference_feature = reference[feature]
        oot_feature = oot[feature]

        reference_missing_rate = (
            reference_feature.isna().mean()
        )

        oot_missing_rate = (
            oot_feature.isna().mean()
        )

        missing_rate_shift = (
            oot_missing_rate
            - reference_missing_rate
        )

        if feature_type == "numeric":

            metrics = analyze_numeric_feature(
                reference_feature,
                oot_feature,
            )

            chi2_stat = np.nan
            chi2_pvalue = np.nan

        else:

            metrics = analyze_categorical_feature(
                reference_feature,
                oot_feature,
            )

            chi2_stat = metrics.pop(
                "chi2_stat",
                np.nan,
            )

            chi2_pvalue = metrics.pop(
                "chi2_pvalue",
                np.nan,
            )

        psi = metrics["psi"]

        drift_status = classify_drift(psi)

        results.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "reference_n": len(reference_feature),
                "oot_n": len(oot_feature),
                "reference_missing_rate":
                    reference_missing_rate,
                "oot_missing_rate":
                    oot_missing_rate,
                "missing_rate_shift":
                    missing_rate_shift,
                "psi": psi,
                "drift_status": drift_status,
                "ks_stat":
                    metrics.get(
                        "ks_stat",
                        np.nan,
                    ),
                "ks_pvalue":
                    metrics.get(
                        "ks_pvalue",
                        np.nan,
                    ),
                "chi2_stat": chi2_stat,
                "chi2_pvalue": chi2_pvalue,
                "reference_mean":
                    metrics.get(
                        "reference_mean",
                        np.nan,
                    ),
                "oot_mean":
                    metrics.get(
                        "oot_mean",
                        np.nan,
                    ),
                "mean_relative_shift":
                    metrics.get(
                        "mean_relative_shift",
                        np.nan,
                    ),
                "reference_median":
                    metrics.get(
                        "reference_median",
                        np.nan,
                    ),
                "oot_median":
                    metrics.get(
                        "oot_median",
                        np.nan,
                    ),
                "median_relative_shift":
                    metrics.get(
                        "median_relative_shift",
                        np.nan,
                    ),
            }
        )

        if i % 20 == 0 or i == len(features):

            print(
                f"Processed: {i:>3} / "
                f"{len(features)} features"
            )

    report = pd.DataFrame(results)

    # ========================================================================
    # 6. SORT BY PSI
    # ========================================================================

    report = report.sort_values(
        by="psi",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    # ========================================================================
    # 7. SUMMARY
    # ========================================================================

    print_header("[6] DRIFT SUMMARY")

    status_counts = (
        report["drift_status"]
        .value_counts()
    )

    stable_count = status_counts.get(
        "STABLE",
        0,
    )

    watch_count = status_counts.get(
        "WATCH",
        0,
    )

    drift_count = status_counts.get(
        "DRIFT",
        0,
    )

    insufficient_count = status_counts.get(
        "INSUFFICIENT_DATA",
        0,
    )

    print(
        f"STABLE:              {stable_count}"
    )

    print(
        f"WATCH:               {watch_count}"
    )

    print(
        f"DRIFT:               {drift_count}"
    )

    print(
        f"INSUFFICIENT DATA:   {insufficient_count}"
    )

    # ========================================================================
    # 8. MISSINGNESS SUMMARY
    # ========================================================================

    print_header("[7] MISSINGNESS DRIFT")

    missingness_report = report.copy()

    missingness_report["abs_missing_rate_shift"] = (
        missingness_report["missing_rate_shift"]
        .abs()
    )

    missingness_report = (
        missingness_report
        .sort_values(
            by="abs_missing_rate_shift",
            ascending=False,
        )
    )

    print(
        missingness_report[
            [
                "feature",
                "reference_missing_rate",
                "oot_missing_rate",
                "missing_rate_shift",
            ]
        ]
        .head(TOP_N)
        .to_string(index=False)
    )

    # ========================================================================
    # 9. TOP PSI FEATURES
    # ========================================================================

    print_header(
        f"[8] TOP {TOP_N} FEATURES BY PSI"
    )

    display_columns = [
        "feature",
        "feature_type",
        "psi",
        "drift_status",
        "ks_stat",
        "ks_pvalue",
        "chi2_pvalue",
        "missing_rate_shift",
        "mean_relative_shift",
        "median_relative_shift",
    ]

    print(
        report[
            display_columns
        ]
        .head(TOP_N)
        .to_string(index=False)
    )

    # ========================================================================
    # 10. STATISTICALLY SIGNIFICANT KS TESTS
    # ========================================================================

    print_header(
        "[9] NUMERICAL FEATURES — "
        "KS TEST SIGNIFICANCE"
    )

    numeric_report = report[
        report["feature_type"] == "numeric"
    ].copy()

    ks_significant = numeric_report[
        numeric_report["ks_pvalue"] < KS_ALPHA
    ]

    print(
        f"Numerical features:       "
        f"{len(numeric_report)}"
    )

    print(
        f"KS p-value < {KS_ALPHA}:     "
        f"{len(ks_significant)}"
    )

    if len(ks_significant) > 0:

        print()

        print(
            ks_significant[
                [
                    "feature",
                    "psi",
                    "ks_stat",
                    "ks_pvalue",
                    "drift_status",
                ]
            ]
            .head(TOP_N)
            .to_string(index=False)
        )

    # ========================================================================
    # 11. CATEGORICAL CHI-SQUARE
    # ========================================================================

    print_header(
        "[10] CATEGORICAL FEATURES — "
        "CHI-SQUARE SIGNIFICANCE"
    )

    categorical_report = report[
        report["feature_type"] == "categorical"
    ].copy()

    chi2_significant = categorical_report[
        categorical_report["chi2_pvalue"] < KS_ALPHA
    ]

    print(
        f"Categorical features:     "
        f"{len(categorical_report)}"
    )

    print(
        f"Chi-square p-value < "
        f"{KS_ALPHA}: "
        f"{len(chi2_significant)}"
    )

    if len(chi2_significant) > 0:

        print()

        print(
            chi2_significant[
                [
                    "feature",
                    "psi",
                    "chi2_stat",
                    "chi2_pvalue",
                    "drift_status",
                ]
            ]
            .head(TOP_N)
            .to_string(index=False)
        )

    # ========================================================================
    # 12. SAVE REPORT
    # ========================================================================

    print_header("[11] SAVING REPORT")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Report saved to:     "
        f"{OUTPUT_PATH}"
    )

    # ========================================================================
    # 13. FINAL VALIDATION CONCLUSION
    # ========================================================================

    print_header("[12] VALIDATION CONCLUSION")

    if drift_count == 0:

        print(
            "No material feature drift detected "
            "according to the PSI thresholds."
        )

    else:

        print(
            f"Material feature drift detected in "
            f"{drift_count} feature(s)."
        )

        print(
            "These features require further investigation."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Feature drift does not by itself demonstrate "
        "model performance deterioration."
    )

    print(
        "Next validation step should assess whether "
        "identified drift is associated with changes "
        "in model discrimination, calibration, or "
        "risk ranking."
    )

    print()
    print("=" * 70)
    print(
        "FEATURE STABILITY ANALYSIS COMPLETE"
    )
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()