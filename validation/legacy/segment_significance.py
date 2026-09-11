"""
NORTHSTAR MODEL VALIDATION
Segment Significance V0.1

Purpose:
    Evaluate whether observed differences in model
    discrimination between population segments and their
    complementary populations are statistically significant.

The analysis compares:

    Segment
        vs
    Complement (all observations outside the segment)

Statistical significance is evaluated using bootstrap
confidence intervals for the difference in ROC-AUC.

This analysis does NOT by itself establish business
materiality or formal model risk.

It is a statistical follow-up to segment_performance.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ======================================================================
# CONFIGURATION
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

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline"
    / "validation_predictions.csv"
)

SEGMENT_PERFORMANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "segment_performance"
    / "segment_performance_analysis.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "segment_significance"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "segment_significance_analysis.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "segment_significance_summary.csv"
)


# ======================================================================
# SEGMENT CONFIGURATION
# ======================================================================

N_BINS = 4

SEGMENT_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "DAYS_BIRTH",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]


# ======================================================================
# STATISTICAL CONFIGURATION
# ======================================================================

N_BOOTSTRAPS = 1000

CONFIDENCE_LEVEL = 0.95

RANDOM_STATE = 42

MIN_SEGMENT_SIZE = 1000

MIN_BAD_COUNT = 30


# ======================================================================
# SIGNIFICANCE CONFIGURATION
# ======================================================================

ALPHA = 1 - CONFIDENCE_LEVEL


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


def calculate_auc(y_true, y_score):
    """
    Calculate ROC-AUC safely.

    ROC-AUC is undefined when only one target class
    is present.
    """

    if len(y_true) == 0:
        return np.nan

    if pd.Series(y_true).nunique() < 2:
        return np.nan

    try:

        return float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

    except Exception:

        return np.nan


def bootstrap_auc_difference(
    y_segment,
    score_segment,
    y_complement,
    score_complement,
    n_bootstraps,
    confidence_level,
    random_state,
):
    """
    Estimate the bootstrap confidence interval for:

        AUC(segment) - AUC(complement)

    Independent bootstrap samples are drawn from the
    segment and complement populations separately.

    Returns:

        observed_difference,
        lower_ci,
        upper_ci
    """

    observed_segment_auc = calculate_auc(
        y_segment,
        score_segment,
    )

    observed_complement_auc = calculate_auc(
        y_complement,
        score_complement,
    )

    if (
        pd.isna(observed_segment_auc)
        or pd.isna(observed_complement_auc)
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    observed_difference = (
        observed_segment_auc
        - observed_complement_auc
    )

    rng = np.random.default_rng(
        random_state
    )

    y_segment = np.asarray(
        y_segment
    )

    score_segment = np.asarray(
        score_segment
    )

    y_complement = np.asarray(
        y_complement
    )

    score_complement = np.asarray(
        score_complement
    )

    segment_n = len(
        y_segment
    )

    complement_n = len(
        y_complement
    )

    bootstrap_differences = []

    for _ in range(
        n_bootstraps
    ):

        segment_indices = (
            rng.integers(
                0,
                segment_n,
                size=segment_n,
            )
        )

        complement_indices = (
            rng.integers(
                0,
                complement_n,
                size=complement_n,
            )
        )

        bootstrap_segment_y = (
            y_segment[
                segment_indices
            ]
        )

        bootstrap_segment_score = (
            score_segment[
                segment_indices
            ]
        )

        bootstrap_complement_y = (
            y_complement[
                complement_indices
            ]
        )

        bootstrap_complement_score = (
            score_complement[
                complement_indices
            ]
        )

        bootstrap_segment_auc = (
            calculate_auc(
                bootstrap_segment_y,
                bootstrap_segment_score,
            )
        )

        bootstrap_complement_auc = (
            calculate_auc(
                bootstrap_complement_y,
                bootstrap_complement_score,
            )
        )

        if (
            pd.isna(
                bootstrap_segment_auc
            )
            or pd.isna(
                bootstrap_complement_auc
            )
        ):

            continue

        bootstrap_difference = (
            bootstrap_segment_auc
            - bootstrap_complement_auc
        )

        bootstrap_differences.append(
            bootstrap_difference
        )

    if not bootstrap_differences:

        return (
            observed_difference,
            np.nan,
            np.nan,
        )

    bootstrap_differences = np.asarray(
        bootstrap_differences
    )

    alpha = (
        1 - confidence_level
    )

    lower_percentile = (
        100 * alpha / 2
    )

    upper_percentile = (
        100 * (1 - alpha / 2)
    )

    lower_ci = np.percentile(
        bootstrap_differences,
        lower_percentile,
    )

    upper_ci = np.percentile(
        bootstrap_differences,
        upper_percentile,
    )

    return (
        float(observed_difference),
        float(lower_ci),
        float(upper_ci),
    )


def create_quantile_segments(
    series,
    n_bins,
):
    """
    Create quantile-based segments.

    Duplicate quantile boundaries are removed.
    """

    try:

        return pd.qcut(
            series,
            q=n_bins,
            duplicates="drop",
        )

    except Exception:

        return pd.Series(
            pd.NA,
            index=series.index,
            dtype="object",
        )


def classify_significance(
    auc_difference,
    lower_ci,
    upper_ci,
):
    """
    Determine whether the bootstrap confidence interval
    excludes zero.

    SIGNIFICANT_NEGATIVE:
        Upper confidence bound < 0.

    SIGNIFICANT_POSITIVE:
        Lower confidence bound > 0.

    NOT_SIGNIFICANT:
        Confidence interval contains zero.

    INSUFFICIENT_DATA:
        Statistical estimate unavailable.
    """

    if (
        pd.isna(auc_difference)
        or pd.isna(lower_ci)
        or pd.isna(upper_ci)
    ):

        return "INSUFFICIENT_DATA"

    if upper_ci < 0:

        return "SIGNIFICANT_NEGATIVE"

    if lower_ci > 0:

        return "SIGNIFICANT_POSITIVE"

    return "NOT_SIGNIFICANT"


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)

print(
    "NORTHSTAR MODEL VALIDATION — "
    "SEGMENT SIGNIFICANCE V0.1"
)

print("=" * 70)


# ======================================================================
# [1] LOADING DATA
# ======================================================================

print_section(
    "[1] LOADING DATA"
)


if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


if not PREDICTIONS_PATH.exists():

    raise FileNotFoundError(
        f"Predictions not found: "
        f"{PREDICTIONS_PATH}"
    )


df = pd.read_csv(
    DATA_PATH
)

predictions = pd.read_csv(
    PREDICTIONS_PATH
)


print(
    f"Dataset shape:       "
    f"{df.shape}"
)

print(
    f"Predictions shape:   "
    f"{predictions.shape}"
)


# ======================================================================
# [2] BASIC VALIDATION CHECKS
# ======================================================================

print_section(
    "[2] BASIC VALIDATION CHECKS"
)


required_prediction_columns = [
    ID_COLUMN,
    "y_true",
    "y_probability",
]


missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]


if missing_prediction_columns:

    raise ValueError(
        "Missing prediction columns: "
        f"{missing_prediction_columns}"
    )


required_data_columns = [
    ID_COLUMN,
    TARGET,
]


missing_data_columns = [
    column
    for column in required_data_columns
    if column not in df.columns
]


if missing_data_columns:

    raise ValueError(
        "Missing dataset columns: "
        f"{missing_data_columns}"
    )


print(
    "Required prediction columns: OK"
)

print(
    "Required dataset columns:    OK"
)


prediction_ids_unique = (
    predictions[ID_COLUMN]
    .is_unique
)


print(
    f"Prediction IDs unique:       "
    f"{prediction_ids_unique}"
)


if not prediction_ids_unique:

    raise ValueError(
        "Prediction IDs are not unique."
    )


# ======================================================================
# [3] MERGING PREDICTIONS WITH DATA
# ======================================================================

print_section(
    "[3] MERGING PREDICTIONS WITH DATA"
)


available_segment_features = [
    feature
    for feature in SEGMENT_FEATURES
    if feature in df.columns
]


missing_segment_features = [
    feature
    for feature in SEGMENT_FEATURES
    if feature not in df.columns
]


if missing_segment_features:

    print(
        "WARNING: Missing segment features:"
    )

    for feature in missing_segment_features:

        print(
            f"  {feature}"
        )


analysis_columns = [
    ID_COLUMN,
    TARGET,
] + available_segment_features


analysis = (
    predictions
    .merge(
        df[analysis_columns],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
)


print(
    f"Merged shape:        "
    f"{analysis.shape}"
)

print(
    f"Segment features:    "
    f"{len(available_segment_features)}"
)


# ======================================================================
# [4] TARGET CONSISTENCY CHECK
# ======================================================================

print_section(
    "[4] TARGET CONSISTENCY CHECK"
)


missing_target = (
    analysis[TARGET]
    .isna()
    .sum()
)


print(
    f"Missing target:      "
    f"{missing_target}"
)


if missing_target > 0:

    raise ValueError(
        "Some prediction IDs could not be "
        "matched to TARGET."
    )


target_consistent = (
    analysis["y_true"].astype(int)
    == analysis[TARGET].astype(int)
).all()


print(
    f"Prediction target consistent: "
    f"{target_consistent}"
)


if not target_consistent:

    raise ValueError(
        "y_true does not match TARGET."
    )


# ======================================================================
# [5] OVERALL PERFORMANCE
# ======================================================================

print_section(
    "[5] OVERALL PERFORMANCE"
)


y_true = (
    analysis[TARGET]
    .astype(int)
)

y_probability = (
    analysis["y_probability"]
)


overall_auc = calculate_auc(
    y_true,
    y_probability,
)


print(
    f"N:                    "
    f"{len(analysis):,}"
)

print(
    f"Bad count:             "
    f"{int(y_true.sum()):,}"
)

print(
    f"Bad rate:              "
    f"{y_true.mean():.4f}"
)

print(
    f"Overall ROC-AUC:       "
    f"{overall_auc:.4f}"
)


# ======================================================================
# [6] CREATING SEGMENTS
# ======================================================================

print_section(
    "[6] CREATING SEGMENTS"
)


for feature in available_segment_features:

    print()

    print(
        f"Creating segments for: "
        f"{feature}"
    )

    segment_column = (
        f"{feature}_BIN"
    )

    analysis[
        segment_column
    ] = create_quantile_segments(
        analysis[feature],
        N_BINS,
    )

    segment_count = (
        analysis[segment_column]
        .nunique(
            dropna=True
        )
    )

    print(
        f"  Segments created: "
        f"{segment_count}"
    )


# ======================================================================
# [7] STATISTICAL SIGNIFICANCE ANALYSIS
# ======================================================================

print_section(
    "[7] STATISTICAL SIGNIFICANCE ANALYSIS"
)


results = []


for feature in available_segment_features:

    print()

    print(
        f"Analyzing: {feature}"
    )

    segment_column = (
        f"{feature}_BIN"
    )

    segment_data = (
        analysis[
            analysis[segment_column]
            .notna()
        ]
        .copy()
    )

    for segment_name, group in (
        segment_data
        .groupby(
            segment_column,
            observed=True,
        )
    ):

        segment_mask = (
            analysis[segment_column]
            == segment_name
        )

        segment = (
            analysis[
                segment_mask
            ]
            .copy()
        )

        complement = (
            analysis[
                ~segment_mask
            ]
            .copy()
        )

        segment_n = len(
            segment
        )

        complement_n = len(
            complement
        )

        segment_bad_count = int(
            segment[TARGET]
            .sum()
        )

        complement_bad_count = int(
            complement[TARGET]
            .sum()
        )

        segment_bad_rate = (
            segment[TARGET]
            .mean()
        )

        complement_bad_rate = (
            complement[TARGET]
            .mean()
        )

        segment_auc = calculate_auc(
            segment[TARGET],
            segment["y_probability"],
        )

        complement_auc = calculate_auc(
            complement[TARGET],
            complement["y_probability"],
        )

        if (
            segment_n < MIN_SEGMENT_SIZE
            or segment_bad_count < MIN_BAD_COUNT
            or complement_n < MIN_SEGMENT_SIZE
            or complement_bad_count < MIN_BAD_COUNT
        ):

            observed_difference = np.nan
            lower_ci = np.nan
            upper_ci = np.nan

            significance_status = (
                "INSUFFICIENT_DATA"
            )

        else:

            (
                observed_difference,
                lower_ci,
                upper_ci,
            ) = bootstrap_auc_difference(
                y_segment=segment[TARGET],
                score_segment=segment[
                    "y_probability"
                ],
                y_complement=complement[
                    TARGET
                ],
                score_complement=complement[
                    "y_probability"
                ],
                n_bootstraps=N_BOOTSTRAPS,
                confidence_level=CONFIDENCE_LEVEL,
                random_state=(
                    RANDOM_STATE
                    + len(results)
                ),
            )

            significance_status = (
                classify_significance(
                    observed_difference,
                    lower_ci,
                    upper_ci,
                )
            )

        results.append(
            {
                "segment_feature":
                    feature,

                "segment":
                    str(segment_name),

                "segment_n":
                    segment_n,

                "segment_bad_count":
                    segment_bad_count,

                "segment_bad_rate":
                    segment_bad_rate,

                "segment_roc_auc":
                    segment_auc,

                "complement_n":
                    complement_n,

                "complement_bad_count":
                    complement_bad_count,

                "complement_bad_rate":
                    complement_bad_rate,

                "complement_roc_auc":
                    complement_auc,

                "auc_difference":
                    observed_difference,

                "ci_lower":
                    lower_ci,

                "ci_upper":
                    upper_ci,

                "confidence_level":
                    CONFIDENCE_LEVEL,

                "n_bootstraps":
                    N_BOOTSTRAPS,

                "significance_status":
                    significance_status,
            }
        )

        print(
            f"  {str(segment_name):30s} "
            f"AUC={segment_auc:.4f} "
            f"vs "
            f"{complement_auc:.4f} "
            f"diff={observed_difference:.4f} "
            f"[{lower_ci:.4f}, "
            f"{upper_ci:.4f}] "
            f"{significance_status}"
        )


# ======================================================================
# [8] BUILD REPORT
# ======================================================================

print_section(
    "[8] BUILDING REPORT"
)


report = pd.DataFrame(
    results
)


if report.empty:

    raise ValueError(
        "No statistical significance results "
        "were generated."
    )


report = (
    report
    .sort_values(
        by=[
            "segment_feature",
            "segment",
        ]
    )
    .reset_index(drop=True)
)


print(
    f"Report rows:         "
    f"{len(report)}"
)


# ======================================================================
# [9] SIGNIFICANCE SUMMARY
# ======================================================================

print_section(
    "[9] SIGNIFICANCE SUMMARY"
)


status_counts = (
    report[
        "significance_status"
    ]
    .value_counts()
)


significant_negative = (
    status_counts.get(
        "SIGNIFICANT_NEGATIVE",
        0,
    )
)

significant_positive = (
    status_counts.get(
        "SIGNIFICANT_POSITIVE",
        0,
    )
)

not_significant = (
    status_counts.get(
        "NOT_SIGNIFICANT",
        0,
    )
)

insufficient_data = (
    status_counts.get(
        "INSUFFICIENT_DATA",
        0,
    )
)


print(
    f"SIGNIFICANT_NEGATIVE: "
    f"{significant_negative}"
)

print(
    f"SIGNIFICANT_POSITIVE: "
    f"{significant_positive}"
)

print(
    f"NOT_SIGNIFICANT:      "
    f"{not_significant}"
)

print(
    f"INSUFFICIENT_DATA:    "
    f"{insufficient_data}"
)


# ======================================================================
# [10] WORST SEGMENTS
# ======================================================================

print_section(
    f"[10] WORST {TOP_N} SEGMENTS BY AUC DIFFERENCE"
)


worst_segments = (
    report[
        report["auc_difference"]
        .notna()
    ]
    .sort_values(
        by="auc_difference",
        ascending=True,
    )
)


display_columns = [
    "segment_feature",
    "segment",
    "segment_n",
    "segment_bad_count",
    "segment_bad_rate",
    "segment_roc_auc",
    "complement_n",
    "complement_bad_rate",
    "complement_roc_auc",
    "auc_difference",
    "ci_lower",
    "ci_upper",
    "significance_status",
]


if len(worst_segments) > 0:

    print(
        worst_segments[
            display_columns
        ]
        .head(TOP_N)
        .to_string(
            index=False
        )
    )


# ======================================================================
# [11] SIGNIFICANT NEGATIVE SEGMENTS
# ======================================================================

print_section(
    "[11] SIGNIFICANT NEGATIVE SEGMENTS"
)


significant_negative_segments = (
    report[
        report[
            "significance_status"
        ]
        == "SIGNIFICANT_NEGATIVE"
    ]
    .sort_values(
        by="auc_difference",
        ascending=True,
    )
)


if (
    len(
        significant_negative_segments
    )
    > 0
):

    print(
        significant_negative_segments[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "No statistically significant "
        "negative AUC differences detected."
    )


# ======================================================================
# [12] FEATURE-LEVEL SUMMARY
# ======================================================================

print_section(
    "[12] FEATURE-LEVEL SUMMARY"
)


feature_summary_rows = []


for feature in available_segment_features:

    feature_report = (
        report[
            report["segment_feature"]
            == feature
        ]
    )

    valid_differences = (
        feature_report[
            "auc_difference"
        ]
        .dropna()
    )

    feature_summary_rows.append(
        {
            "segment_feature":
                feature,

            "segments":
                len(feature_report),

            "valid_tests":
                len(valid_differences),

            "min_auc_difference":
                valid_differences.min()
                if len(valid_differences) > 0
                else np.nan,

            "max_auc_difference":
                valid_differences.max()
                if len(valid_differences) > 0
                else np.nan,

            "significant_negative":
                (
                    feature_report[
                        "significance_status"
                    ]
                    == "SIGNIFICANT_NEGATIVE"
                ).sum(),

            "significant_positive":
                (
                    feature_report[
                        "significance_status"
                    ]
                    == "SIGNIFICANT_POSITIVE"
                ).sum(),

            "not_significant":
                (
                    feature_report[
                        "significance_status"
                    ]
                    == "NOT_SIGNIFICANT"
                ).sum(),
        }
    )


feature_summary = pd.DataFrame(
    feature_summary_rows
)


print(
    feature_summary.to_string(
        index=False
    )
)


# ======================================================================
# [13] SAVING RESULTS
# ======================================================================

print_section(
    "[13] SAVING RESULTS"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


report.to_csv(
    RESULTS_PATH,
    index=False,
)


summary_rows = [
    {
        "metric":
            "total_segments",

        "value":
            len(report),
    },

    {
        "metric":
            "significant_negative",

        "value":
            significant_negative,
    },

    {
        "metric":
            "significant_positive",

        "value":
            significant_positive,
    },

    {
        "metric":
            "not_significant",

        "value":
            not_significant,
    },

    {
        "metric":
            "insufficient_data",

        "value":
            insufficient_data,
    },

    {
        "metric":
            "bootstrap_iterations",

        "value":
            N_BOOTSTRAPS,
    },

    {
        "metric":
            "confidence_level",

        "value":
            CONFIDENCE_LEVEL,
    },

    {
        "metric":
            "random_state",

        "value":
            RANDOM_STATE,
    },
]


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


print()

print(
    "Statistical significance report saved to:"
)

print(
    f"  {RESULTS_PATH}"
)

print()

print(
    "Statistical significance summary saved to:"
)

print(
    f"  {SUMMARY_PATH}"
)


# ======================================================================
# [14] VALIDATION INTERPRETATION
# ======================================================================

print_section(
    "[14] VALIDATION INTERPRETATION"
)


print()

print(
    "This analysis compares ROC-AUC within each "
    "population segment against the complementary "
    "validation population."
)

print()

print(
    "Statistical significance is evaluated using "
    "a non-parametric bootstrap confidence interval "
    "for the difference in ROC-AUC."
)

print()

print(
    "The tested quantity is:"
)

print(
    "  AUC(segment) - AUC(complement)"
)

print()

print(
    "A statistically significant negative difference "
    "means that model discrimination is lower in the "
    "segment than in the complementary population."
)

print()

print(
    f"Confidence level: "
    f"{CONFIDENCE_LEVEL:.0%}"
)

print()

print(
    "A confidence interval entirely below zero is "
    "classified as SIGNIFICANT_NEGATIVE."
)

print()

print(
    "A confidence interval entirely above zero is "
    "classified as SIGNIFICANT_POSITIVE."
)

print()

print(
    "If the confidence interval contains zero, the "
    "observed AUC difference is classified as "
    "NOT_SIGNIFICANT."
)

print()

print(
    "Important:"
)

print(
    "Statistical significance does NOT establish "
    "business materiality or formal model risk."
)

print(
    "Multiple segment tests are performed, so the "
    "results should be interpreted as an investigation "
    "screen rather than as a formal hypothesis-testing "
    "framework with multiplicity adjustment."
)


# ======================================================================
# [15] FINAL CONCLUSION
# ======================================================================

print_section(
    "[15] VALIDATION CONCLUSION"
)


if significant_negative > 0:

    print()

    print(
        f"Statistically significant negative "
        f"AUC differences were detected in "
        f"{significant_negative} segment(s)."
    )

    print()

    print(
        "These segments require further investigation "
        "for potential model performance weakness."
    )

elif not_significant > 0:

    print()

    print(
        "No statistically significant negative "
        "AUC differences were detected."
    )

    print(
        "Observed segment performance differences "
        "may be attributable to sampling variation."
    )

else:

    print()

    print(
        "No interpretable statistical significance "
        "results were generated."
    )


print()

print(
    "Statistical significance should be considered "
    "together with segment size, bad rate, business "
    "materiality, model purpose, and other validation "
    "evidence."
)

print()

print("=" * 70)

print(
    "SEGMENT SIGNIFICANCE ANALYSIS COMPLETE"
)

print("=" * 70)