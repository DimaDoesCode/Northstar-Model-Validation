from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


# ======================================================================
# NORTHSTAR MODEL VALIDATION — FEATURE STABILITY / DATA DRIFT V0.2
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "application_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "feature_stability"
)

FEATURE_STABILITY_RESULTS_PATH = (
    OUTPUT_DIR
    / "feature_stability_analysis.csv"
)

SUMMARY_RESULTS_PATH = (
    OUTPUT_DIR
    / "feature_stability_summary.csv"
)


# ======================================================================
# SPLIT CONFIGURATION
# ======================================================================

# IMPORTANT:
#
# This is NOT a formal temporal / OOT split.
#
# The current Home Credit dataset does not contain a genuine application
# or observation date.
#
# Therefore v0.2 uses the row order only as a comparison sample.
#
# Reference:
#     first 80% of rows
#
# Comparison:
#     last 20% of rows
#
# This should be interpreted as a stability sanity check, not as formal
# production-time drift evidence.

REFERENCE_FRACTION = 0.80


# ======================================================================
# FEATURE CONFIGURATION
# ======================================================================

ID_COLUMNS = [
    ID_COLUMN,
]

EXCLUDED_COLUMNS = (
    ID_COLUMNS
    + [TARGET]
)


# ======================================================================
# PSI CONFIGURATION
# ======================================================================

N_BINS = 10

EPSILON = 1e-6

PSI_STABLE_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.25


# ======================================================================
# STATISTICAL TEST CONFIGURATION
# ======================================================================

KS_ALPHA = 0.05

MIN_SAMPLE_SIZE = 100


# ======================================================================
# DISPLAY CONFIGURATION
# ======================================================================

TOP_N = 15


# ======================================================================
# HELPERS
# ======================================================================


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# PSI
# ======================================================================


def calculate_psi(
    reference,
    comparison,
    n_bins=10,
):
    """
    Calculate Population Stability Index.

    Numerical features:
        Bins are defined using reference quantiles.

    Categorical features:
        Categories are defined as the union of categories
        present in reference and comparison samples.
    """

    reference = pd.Series(reference).dropna()
    comparison = pd.Series(comparison).dropna()

    if len(reference) == 0 or len(comparison) == 0:
        return np.nan

    # ------------------------------------------------------------------
    # Numerical feature
    # ------------------------------------------------------------------

    if pd.api.types.is_numeric_dtype(reference):

        if reference.nunique() <= 1:
            return 0.0

        quantiles = np.linspace(
            0,
            1,
            n_bins + 1,
        )

        breakpoints = (
            reference
            .quantile(quantiles)
            .values
        )

        breakpoints = np.unique(
            breakpoints
        )

        if len(breakpoints) < 3:
            return 0.0

        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        reference_bins = pd.cut(
            reference,
            bins=breakpoints,
            include_lowest=True,
        )

        comparison_bins = pd.cut(
            comparison,
            bins=breakpoints,
            include_lowest=True,
        )

        reference_distribution = (
            reference_bins
            .value_counts(
                normalize=True,
                sort=False,
            )
        )

        comparison_distribution = (
            comparison_bins
            .value_counts(
                normalize=True,
                sort=False,
            )
        )

    # ------------------------------------------------------------------
    # Categorical feature
    # ------------------------------------------------------------------

    else:

        reference = reference.astype(str)
        comparison = comparison.astype(str)

        categories = sorted(
            set(reference.unique())
            | set(comparison.unique())
        )

        reference_distribution = (
            reference
            .value_counts(
                normalize=True
            )
            .reindex(
                categories,
                fill_value=0,
            )
        )

        comparison_distribution = (
            comparison
            .value_counts(
                normalize=True
            )
            .reindex(
                categories,
                fill_value=0,
            )
        )

    reference_distribution = (
        reference_distribution
        .clip(lower=EPSILON)
    )

    comparison_distribution = (
        comparison_distribution
        .clip(lower=EPSILON)
    )

    psi = (
        (
            comparison_distribution
            - reference_distribution
        )
        * np.log(
            comparison_distribution
            / reference_distribution
        )
    ).sum()

    return float(psi)


# ======================================================================
# NUMERICAL FEATURE ANALYSIS
# ======================================================================


def analyze_numeric_feature(
    reference,
    comparison,
):
    """
    Calculate stability statistics for a numerical feature.
    """

    reference_nonnull = (
        reference.dropna()
    )

    comparison_nonnull = (
        comparison.dropna()
    )

    if (
        len(reference_nonnull)
        < MIN_SAMPLE_SIZE
        or len(comparison_nonnull)
        < MIN_SAMPLE_SIZE
    ):
        return {
            "psi": np.nan,
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "reference_mean": np.nan,
            "comparison_mean": np.nan,
            "mean_relative_shift": np.nan,
            "reference_median": np.nan,
            "comparison_median": np.nan,
            "median_relative_shift": np.nan,
        }

    psi = calculate_psi(
        reference_nonnull,
        comparison_nonnull,
        n_bins=N_BINS,
    )

    try:

        ks_stat, ks_pvalue = ks_2samp(
            reference_nonnull,
            comparison_nonnull,
        )

    except Exception:

        ks_stat = np.nan
        ks_pvalue = np.nan

    reference_mean = (
        reference_nonnull.mean()
    )

    comparison_mean = (
        comparison_nonnull.mean()
    )

    reference_median = (
        reference_nonnull.median()
    )

    comparison_median = (
        comparison_nonnull.median()
    )

    if abs(reference_mean) > EPSILON:

        mean_relative_shift = (
            comparison_mean
            - reference_mean
        ) / abs(reference_mean)

    else:

        mean_relative_shift = np.nan

    if abs(reference_median) > EPSILON:

        median_relative_shift = (
            comparison_median
            - reference_median
        ) / abs(reference_median)

    else:

        median_relative_shift = np.nan

    return {
        "psi": psi,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "reference_mean": reference_mean,
        "comparison_mean": comparison_mean,
        "mean_relative_shift": mean_relative_shift,
        "reference_median": reference_median,
        "comparison_median": comparison_median,
        "median_relative_shift": median_relative_shift,
    }


# ======================================================================
# CATEGORICAL FEATURE ANALYSIS
# ======================================================================


def analyze_categorical_feature(
    reference,
    comparison,
):
    """
    Calculate stability statistics for a categorical feature.

    Chi-square test evaluates whether category proportions differ
    between reference and comparison samples.
    """

    reference_nonnull = (
        reference
        .dropna()
        .astype(str)
    )

    comparison_nonnull = (
        comparison
        .dropna()
        .astype(str)
    )

    if (
        len(reference_nonnull)
        < MIN_SAMPLE_SIZE
        or len(comparison_nonnull)
        < MIN_SAMPLE_SIZE
    ):
        return {
            "psi": np.nan,
            "chi2_stat": np.nan,
            "chi2_pvalue": np.nan,
        }

    psi = calculate_psi(
        reference_nonnull,
        comparison_nonnull,
        n_bins=N_BINS,
    )

    categories = sorted(
        set(reference_nonnull.unique())
        | set(comparison_nonnull.unique())
    )

    reference_counts = (
        reference_nonnull
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    comparison_counts = (
        comparison_nonnull
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    contingency_table = np.vstack(
        [
            reference_counts.values,
            comparison_counts.values,
        ]
    )

    try:

        chi2_stat, chi2_pvalue, _, _ = (
            chi2_contingency(
                contingency_table
            )
        )

    except Exception:

        chi2_stat = np.nan
        chi2_pvalue = np.nan

    return {
        "psi": psi,
        "chi2_stat": chi2_stat,
        "chi2_pvalue": chi2_pvalue,
    }


# ======================================================================
# FEATURE TYPE DETECTION
# ======================================================================


def detect_feature_type(series):
    """
    Determine whether feature is numerical or categorical.

    Numeric pandas dtypes:
        numerical

    Everything else:
        categorical
    """

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "categorical"


# ======================================================================
# DRIFT CLASSIFICATION
# ======================================================================


def classify_drift(psi):
    """
    Classify feature stability based on PSI.

    PSI < 0.10:
        STABLE

    0.10 <= PSI < 0.25:
        WATCH

    PSI >= 0.25:
        DRIFT
    """

    if pd.isna(psi):
        return "INSUFFICIENT_DATA"

    if psi < PSI_STABLE_THRESHOLD:
        return "STABLE"

    if psi < PSI_DRIFT_THRESHOLD:
        return "WATCH"

    return "DRIFT"


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print(
    "NORTHSTAR MODEL VALIDATION — "
    "FEATURE STABILITY / DATA DRIFT V0.2"
)
print("=" * 70)


# ======================================================================
# [1] LOADING DATA
# ======================================================================

print_section("[1] LOADING DATA")

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape:       {df.shape}"
)


# ======================================================================
# [2] BASIC VALIDATION CHECKS
# ======================================================================

print_section("[2] BASIC VALIDATION CHECKS")

target_present = (
    TARGET in df.columns
)

id_columns_present = all(
    column in df.columns
    for column in ID_COLUMNS
)

print(
    f"Target column present: "
    f"{target_present}"
)

print(
    f"ID columns present:    "
    f"{id_columns_present}"
)

if not target_present:

    raise ValueError(
        f"Target column '{TARGET}' "
        "not found."
    )

if not id_columns_present:

    raise ValueError(
        "One or more ID columns are missing."
    )

missing_target = (
    df[TARGET]
    .isna()
    .sum()
)

print(
    f"Missing target:        "
    f"{missing_target}"
)

if missing_target > 0:

    raise ValueError(
        "Missing target values detected."
    )


# ======================================================================
# [3] DEFINING REFERENCE / COMPARISON SAMPLES
# ======================================================================

print_section(
    "[3] DEFINING REFERENCE / COMPARISON SAMPLES"
)

split_index = int(
    len(df)
    * REFERENCE_FRACTION
)

reference = (
    df.iloc[:split_index]
    .copy()
)

comparison = (
    df.iloc[split_index:]
    .copy()
)

print(
    "Split mode:           ROW"
)

print(
    f"Reference fraction:   "
    f"{REFERENCE_FRACTION:.0%}"
)

print(
    f"Reference rows:       "
    f"{len(reference):,}"
)

print(
    f"Comparison rows:      "
    f"{len(comparison):,}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is a stability sanity check."
)

print(
    "It is NOT a formal temporal / OOT split."
)


# ======================================================================
# [4] PREPARING FEATURES
# ======================================================================

print_section("[4] PREPARING FEATURES")

features = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]

print(
    f"Total features:       "
    f"{len(features)}"
)

print(
    f"Excluded columns:     "
    f"{EXCLUDED_COLUMNS}"
)


# ======================================================================
# [5] FEATURE STABILITY ANALYSIS
# ======================================================================

print_section(
    "[5] FEATURE STABILITY ANALYSIS"
)

results = []

for index, feature in enumerate(
    features,
    start=1,
):

    series = df[feature]

    feature_type = (
        detect_feature_type(series)
    )

    reference_feature = (
        reference[feature]
    )

    comparison_feature = (
        comparison[feature]
    )

    # --------------------------------------------------------------
    # Missingness
    # --------------------------------------------------------------

    reference_missing_rate = (
        reference_feature
        .isna()
        .mean()
    )

    comparison_missing_rate = (
        comparison_feature
        .isna()
        .mean()
    )

    missing_rate_shift = (
        comparison_missing_rate
        - reference_missing_rate
    )

    # --------------------------------------------------------------
    # Feature-specific statistics
    # --------------------------------------------------------------

    if feature_type == "numeric":

        metrics = (
            analyze_numeric_feature(
                reference_feature,
                comparison_feature,
            )
        )

        chi2_stat = np.nan
        chi2_pvalue = np.nan

    else:

        metrics = (
            analyze_categorical_feature(
                reference_feature,
                comparison_feature,
            )
        )

        chi2_stat = metrics.get(
            "chi2_stat",
            np.nan,
        )

        chi2_pvalue = metrics.get(
            "chi2_pvalue",
            np.nan,
        )

    psi = metrics["psi"]

    drift_status = classify_drift(
        psi
    )

    # --------------------------------------------------------------
    # Result row
    # --------------------------------------------------------------

    result = {
        "feature": feature,
        "feature_type": feature_type,
        "reference_n": len(
            reference_feature
        ),
        "comparison_n": len(
            comparison_feature
        ),
        "reference_missing_rate":
            reference_missing_rate,
        "comparison_missing_rate":
            comparison_missing_rate,
        "missing_rate_shift":
            missing_rate_shift,
        "psi": psi,
        "drift_status": drift_status,
        "ks_stat": metrics.get(
            "ks_stat",
            np.nan,
        ),
        "ks_pvalue": metrics.get(
            "ks_pvalue",
            np.nan,
        ),
        "chi2_stat": chi2_stat,
        "chi2_pvalue": chi2_pvalue,
        "reference_mean": metrics.get(
            "reference_mean",
            np.nan,
        ),
        "comparison_mean": metrics.get(
            "comparison_mean",
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
        "comparison_median":
            metrics.get(
                "comparison_median",
                np.nan,
            ),
        "median_relative_shift":
            metrics.get(
                "median_relative_shift",
                np.nan,
            ),
    }

    results.append(result)

    if (
        index % 20 == 0
        or index == len(features)
    ):

        print(
            f"Processed: {index:>3} / "
            f"{len(features)} features"
        )


# ======================================================================
# [6] BUILD REPORT
# ======================================================================

print_section("[6] BUILDING REPORT")

report = pd.DataFrame(
    results
)

report = (
    report
    .sort_values(
        by="psi",
        ascending=False,
        na_position="last",
    )
    .reset_index(drop=True)
)

print(
    f"Report rows:         "
    f"{len(report)}"
)


# ======================================================================
# [7] DRIFT SUMMARY
# ======================================================================

print_section("[7] DRIFT SUMMARY")

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

insufficient_count = (
    status_counts.get(
        "INSUFFICIENT_DATA",
        0,
    )
)

print(
    f"STABLE:              "
    f"{stable_count}"
)

print(
    f"WATCH:               "
    f"{watch_count}"
)

print(
    f"DRIFT:               "
    f"{drift_count}"
)

print(
    f"INSUFFICIENT DATA:   "
    f"{insufficient_count}"
)


# ======================================================================
# [8] MISSINGNESS DRIFT
# ======================================================================

print_section(
    "[8] MISSINGNESS DRIFT"
)

missingness_report = (
    report
    .copy()
)

missingness_report[
    "abs_missing_rate_shift"
] = (
    missingness_report[
        "missing_rate_shift"
    ]
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
            "comparison_missing_rate",
            "missing_rate_shift",
        ]
    ]
    .head(TOP_N)
    .to_string(
        index=False
    )
)


# ======================================================================
# [9] TOP FEATURES BY PSI
# ======================================================================

print_section(
    f"[9] TOP {TOP_N} FEATURES BY PSI"
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
    .to_string(
        index=False
    )
)


# ======================================================================
# [10] NUMERICAL FEATURES — KS TEST
# ======================================================================

print_section(
    "[10] NUMERICAL FEATURES — KS TEST"
)

numeric_report = (
    report[
        report["feature_type"]
        == "numeric"
    ]
    .copy()
)

ks_significant = (
    numeric_report[
        numeric_report["ks_pvalue"]
        < KS_ALPHA
    ]
)

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
        .to_string(
            index=False
        )
    )


# ======================================================================
# [11] CATEGORICAL FEATURES — CHI-SQUARE TEST
# ======================================================================

print_section(
    "[11] CATEGORICAL FEATURES — "
    "CHI-SQUARE TEST"
)

categorical_report = (
    report[
        report["feature_type"]
        == "categorical"
    ]
    .copy()
)

chi2_significant = (
    categorical_report[
        categorical_report["chi2_pvalue"]
        < KS_ALPHA
    ]
)

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
        .to_string(
            index=False
        )
    )


# ======================================================================
# [12] FEATURE TYPE SUMMARY
# ======================================================================

print_section(
    "[12] FEATURE TYPE SUMMARY"
)

numeric_stable = len(
    numeric_report[
        numeric_report["drift_status"]
        == "STABLE"
    ]
)

numeric_watch = len(
    numeric_report[
        numeric_report["drift_status"]
        == "WATCH"
    ]
)

numeric_drift = len(
    numeric_report[
        numeric_report["drift_status"]
        == "DRIFT"
    ]
)

categorical_stable = len(
    categorical_report[
        categorical_report["drift_status"]
        == "STABLE"
    ]
)

categorical_watch = len(
    categorical_report[
        categorical_report["drift_status"]
        == "WATCH"
    ]
)

categorical_drift = len(
    categorical_report[
        categorical_report["drift_status"]
        == "DRIFT"
    ]
)

print()
print("Numerical:")
print(
    f"  STABLE:             "
    f"{numeric_stable}"
)
print(
    f"  WATCH:              "
    f"{numeric_watch}"
)
print(
    f"  DRIFT:              "
    f"{numeric_drift}"
)

print()
print("Categorical:")
print(
    f"  STABLE:             "
    f"{categorical_stable}"
)
print(
    f"  WATCH:              "
    f"{categorical_watch}"
)
print(
    f"  DRIFT:              "
    f"{categorical_drift}"
)


# ======================================================================
# [13] OVERALL STATISTICS
# ======================================================================

print_section(
    "[13] OVERALL STABILITY STATISTICS"
)

valid_psi = (
    report["psi"]
    .dropna()
)

if len(valid_psi) > 0:

    print(
        f"Mean PSI:             "
        f"{valid_psi.mean():.6f}"
    )

    print(
        f"Median PSI:           "
        f"{valid_psi.median():.6f}"
    )

    print(
        f"Maximum PSI:          "
        f"{valid_psi.max():.6f}"
    )

    print(
        f"Features PSI >= 0.10: "
        f"{(valid_psi >= 0.10).sum()}"
    )

    print(
        f"Features PSI >= 0.25: "
        f"{(valid_psi >= 0.25).sum()}"
    )


# ======================================================================
# [14] SAVING RESULTS
# ======================================================================

print_section("[14] SAVING RESULTS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

report.to_csv(
    FEATURE_STABILITY_RESULTS_PATH,
    index=False,
)


# ----------------------------------------------------------------------
# Summary file
# ----------------------------------------------------------------------

summary_rows = [
    {
        "metric": "total_features",
        "value": len(report),
    },
    {
        "metric": "stable_features",
        "value": stable_count,
    },
    {
        "metric": "watch_features",
        "value": watch_count,
    },
    {
        "metric": "drift_features",
        "value": drift_count,
    },
    {
        "metric": "insufficient_data_features",
        "value": insufficient_count,
    },
    {
        "metric": "numerical_features",
        "value": len(numeric_report),
    },
    {
        "metric": "categorical_features",
        "value": len(categorical_report),
    },
    {
        "metric": "ks_significant_features",
        "value": len(ks_significant),
    },
    {
        "metric": "chi2_significant_features",
        "value": len(chi2_significant),
    },
    {
        "metric": "mean_psi",
        "value": valid_psi.mean()
        if len(valid_psi) > 0
        else np.nan,
    },
    {
        "metric": "median_psi",
        "value": valid_psi.median()
        if len(valid_psi) > 0
        else np.nan,
    },
    {
        "metric": "max_psi",
        "value": valid_psi.max()
        if len(valid_psi) > 0
        else np.nan,
    },
]

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_RESULTS_PATH,
    index=False,
)

print()
print(
    "Feature stability report saved to:"
)

print(
    f"  {FEATURE_STABILITY_RESULTS_PATH}"
)

print()
print(
    "Feature stability summary saved to:"
)

print(
    f"  {SUMMARY_RESULTS_PATH}"
)


# ======================================================================
# [15] VALIDATION INTERPRETATION
# ======================================================================

print_section(
    "[15] VALIDATION INTERPRETATION"
)

print()
print(
    "This analysis evaluates feature distribution "
    "stability between a reference sample and a "
    "comparison sample."
)

print()
print(
    "The current reference/comparison split is based "
    "on row order and is NOT a formal temporal / OOT split."
)

print()
print(
    "Therefore, the results should be interpreted as "
    "a feature stability sanity check."
)

print()
print(
    "PSI interpretation:"
)

print(
    f"  PSI < {PSI_STABLE_THRESHOLD:.2f}"
    "  -> STABLE"
)

print(
    f"  {PSI_STABLE_THRESHOLD:.2f}"
    f" <= PSI < "
    f"{PSI_DRIFT_THRESHOLD:.2f}"
    "  -> WATCH"
)

print(
    f"  PSI >= {PSI_DRIFT_THRESHOLD:.2f}"
    "  -> DRIFT"
)

print()
print(
    "Important:"
)

print(
    "Statistical significance does not automatically "
    "imply material model risk."
)

print(
    "Likewise, feature drift does not automatically "
    "demonstrate model performance deterioration."
)

print()
print(
    "Any material feature drift should be investigated "
    "for business relevance and potential impact on "
    "model discrimination, calibration, and risk ranking."
)


# ======================================================================
# [16] FINAL CONCLUSION
# ======================================================================

print_section(
    "[16] VALIDATION CONCLUSION"
)

if drift_count == 0 and watch_count == 0:

    print()
    print(
        "No material feature distribution drift was "
        "detected in the current reference/comparison "
        "sanity check."
    )

elif drift_count == 0:

    print()
    print(
        f"No PSI-level DRIFT was detected, but "
        f"{watch_count} feature(s) require monitoring."
    )

else:

    print()
    print(
        f"Material feature drift was detected in "
        f"{drift_count} feature(s)."
    )

    print(
        "Further investigation is required."
    )

print()
print(
    "This result should NOT be interpreted as formal "
    "production OOT stability evidence."
)

print()
print("=" * 70)
print(
    "FEATURE STABILITY ANALYSIS COMPLETE"
)
print("=" * 70)