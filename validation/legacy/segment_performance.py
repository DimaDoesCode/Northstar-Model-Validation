"""
NORTHSTAR MODEL VALIDATION
Segment Performance V0.1

Purpose:
    Evaluate baseline model performance across predefined
    population segments.

The analysis is intended to identify potential differences
in model discrimination and default rate across segments.

This is a descriptive segment performance analysis.

It does NOT determine statistical significance.
Statistical significance is evaluated separately by
segment_significance.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "segment_performance"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "segment_performance_analysis.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "segment_performance_summary.csv"
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
# VALIDATION CONFIGURATION
# ======================================================================

MIN_SEGMENT_SIZE = 1000

MIN_BAD_COUNT = 30

DEGRADATION_AUC_THRESHOLD = 0.02

SEVERE_DEGRADATION_AUC_THRESHOLD = 0.05


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


# ----------------------------------------------------------------------
# Safe ROC-AUC
# ----------------------------------------------------------------------


def calculate_roc_auc(y_true, y_score):
    """
    Calculate ROC-AUC safely.

    ROC-AUC is undefined when only one target class
    is present in the segment.
    """

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


# ----------------------------------------------------------------------
# Safe PR-AUC
# ----------------------------------------------------------------------


def calculate_pr_auc(y_true, y_score):
    """
    Calculate PR-AUC safely.

    Average precision is defined even for some highly
    imbalanced segments, but we still guard against
    unexpected calculation errors.
    """

    if len(y_true) == 0:
        return np.nan

    try:
        return float(
            average_precision_score(
                y_true,
                y_score,
            )
        )

    except Exception:
        return np.nan


# ----------------------------------------------------------------------
# Gini
# ----------------------------------------------------------------------


def calculate_gini(roc_auc):
    """
    Convert ROC-AUC to Gini coefficient.
    """

    if pd.isna(roc_auc):
        return np.nan

    return float(
        2 * roc_auc - 1
    )


# ----------------------------------------------------------------------
# Performance classification
# ----------------------------------------------------------------------


def classify_performance(
    auc_diff,
    segment_n,
    bad_count,
):
    """
    Classify segment performance.

    Classification is intentionally descriptive.

    OK:
        AUC degradation is below configured threshold.

    DEGRADED:
        AUC degradation exceeds configured threshold.

    SEVERE_DEGRADATION:
        AUC degradation exceeds severe threshold.

    INSUFFICIENT_DATA:
        Segment is too small or contains too few bads.
    """

    if (
        segment_n < MIN_SEGMENT_SIZE
        or bad_count < MIN_BAD_COUNT
    ):
        return "INSUFFICIENT_DATA"

    if pd.isna(auc_diff):
        return "INSUFFICIENT_DATA"

    if auc_diff <= -SEVERE_DEGRADATION_AUC_THRESHOLD:
        return "SEVERE_DEGRADATION"

    if auc_diff <= -DEGRADATION_AUC_THRESHOLD:
        return "DEGRADED"

    return "OK"


# ----------------------------------------------------------------------
# Create quantile segments
# ----------------------------------------------------------------------


def create_quantile_segments(
    series,
    n_bins,
):
    """
    Create quantile-based segments.

    Duplicate quantile boundaries are removed.

    If the feature does not provide enough unique values,
    the function returns a reduced number of bins.
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


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print(
    "NORTHSTAR MODEL VALIDATION — "
    "SEGMENT PERFORMANCE V0.1"
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
    f"Dataset shape:       {df.shape}"
)

print(
    f"Predictions shape:   {predictions.shape}"
)


# ======================================================================
# [2] BASIC VALIDATION CHECKS
# ======================================================================

print_section("[2] BASIC VALIDATION CHECKS")


required_prediction_columns = [
    ID_COLUMN,
    "y_true",
    "y_probability",
    "y_prediction",
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


# ----------------------------------------------------------------------
# Prediction ID uniqueness
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Target consistency
# ----------------------------------------------------------------------

target_check = (
    predictions
    .merge(
        df[
            [
                ID_COLUMN,
                TARGET,
            ]
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
)


missing_target_after_merge = (
    target_check[TARGET]
    .isna()
    .sum()
)


print(
    f"Missing target after merge:  "
    f"{missing_target_after_merge}"
)


if missing_target_after_merge > 0:

    raise ValueError(
        "Some prediction IDs could not be "
        "matched to the dataset."
    )


target_consistent = (
    target_check["y_true"].astype(int)
    == target_check[TARGET].astype(int)
).all()


print(
    f"Prediction target consistent: "
    f"{target_consistent}"
)


if not target_consistent:

    raise ValueError(
        "y_true in predictions does not match "
        "TARGET in source dataset."
    )


# ======================================================================
# [3] MERGING PREDICTIONS WITH FEATURES
# ======================================================================

print_section(
    "[3] MERGING PREDICTIONS WITH FEATURES"
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
    f"Merged shape:        {analysis.shape}"
)

print(
    f"Segment features:    "
    f"{len(available_segment_features)}"
)


# ======================================================================
# [4] OVERALL MODEL PERFORMANCE
# ======================================================================

print_section(
    "[4] OVERALL MODEL PERFORMANCE"
)


y_true = (
    analysis[TARGET]
    .astype(int)
)

y_probability = (
    analysis["y_probability"]
)


overall_roc_auc = calculate_roc_auc(
    y_true,
    y_probability,
)

overall_pr_auc = calculate_pr_auc(
    y_true,
    y_probability,
)

overall_gini = calculate_gini(
    overall_roc_auc
)

overall_bad_rate = (
    y_true.mean()
)


print(
    f"N:                    "
    f"{len(analysis):,}"
)

print(
    f"Bad count:             "
    f"{y_true.sum():,}"
)

print(
    f"Bad rate:              "
    f"{overall_bad_rate:.4f}"
)

print(
    f"ROC-AUC:              "
    f"{overall_roc_auc:.4f}"
)

print(
    f"PR-AUC:               "
    f"{overall_pr_auc:.4f}"
)

print(
    f"Gini:                 "
    f"{overall_gini:.4f}"
)


# ======================================================================
# [5] CREATING SEGMENTS
# ======================================================================

print_section(
    "[5] CREATING SEGMENTS"
)


results = []


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
# [6] SEGMENT PERFORMANCE ANALYSIS
# ======================================================================

print_section(
    "[6] SEGMENT PERFORMANCE ANALYSIS"
)


for feature in available_segment_features:

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

        segment_n = len(group)

        segment_bad_count = int(
            group[TARGET]
            .sum()
        )

        segment_bad_rate = (
            group[TARGET]
            .mean()
        )

        segment_roc_auc = (
            calculate_roc_auc(
                group[TARGET],
                group["y_probability"],
            )
        )

        segment_pr_auc = (
            calculate_pr_auc(
                group[TARGET],
                group["y_probability"],
            )
        )

        segment_gini = (
            calculate_gini(
                segment_roc_auc
            )
        )

        auc_diff = (
            segment_roc_auc
            - overall_roc_auc
            if not pd.isna(
                segment_roc_auc
            )
            else np.nan
        )

        pr_auc_diff = (
            segment_pr_auc
            - overall_pr_auc
            if not pd.isna(
                segment_pr_auc
            )
            else np.nan
        )

        bad_rate_diff = (
            segment_bad_rate
            - overall_bad_rate
        )

        performance_status = (
            classify_performance(
                auc_diff,
                segment_n,
                segment_bad_count,
            )
        )

        results.append(
            {
                "segment_feature": feature,
                "segment": str(
                    segment_name
                ),
                "segment_n": segment_n,
                "bad_count": segment_bad_count,
                "bad_rate": segment_bad_rate,
                "bad_rate_diff":
                    bad_rate_diff,
                "roc_auc":
                    segment_roc_auc,
                "auc_diff_vs_overall":
                    auc_diff,
                "pr_auc":
                    segment_pr_auc,
                "pr_auc_diff_vs_overall":
                    pr_auc_diff,
                "gini":
                    segment_gini,
                "performance_status":
                    performance_status,
            }
        )


# ======================================================================
# [7] BUILD REPORT
# ======================================================================

print_section(
    "[7] BUILDING REPORT"
)


report = pd.DataFrame(
    results
)


if report.empty:

    raise ValueError(
        "No segment performance results "
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
# [8] SEGMENT PERFORMANCE SUMMARY
# ======================================================================

print_section(
    "[8] SEGMENT PERFORMANCE SUMMARY"
)


status_counts = (
    report["performance_status"]
    .value_counts()
)


ok_count = status_counts.get(
    "OK",
    0,
)

degraded_count = status_counts.get(
    "DEGRADED",
    0,
)

severe_count = status_counts.get(
    "SEVERE_DEGRADATION",
    0,
)

insufficient_count = (
    status_counts.get(
        "INSUFFICIENT_DATA",
        0,
    )
)


print(
    f"OK:                  "
    f"{ok_count}"
)

print(
    f"DEGRADED:            "
    f"{degraded_count}"
)

print(
    f"SEVERE DEGRADATION:  "
    f"{severe_count}"
)

print(
    f"INSUFFICIENT DATA:   "
    f"{insufficient_count}"
)


# ======================================================================
# [9] WORST SEGMENTS BY AUC
# ======================================================================

print_section(
    f"[9] WORST {TOP_N} SEGMENTS BY AUC DIFFERENCE"
)


worst_segments = (
    report[
        report["auc_diff_vs_overall"]
        .notna()
    ]
    .sort_values(
        by="auc_diff_vs_overall",
        ascending=True,
    )
)


display_columns = [
    "segment_feature",
    "segment",
    "segment_n",
    "bad_count",
    "bad_rate",
    "roc_auc",
    "auc_diff_vs_overall",
    "pr_auc",
    "pr_auc_diff_vs_overall",
    "performance_status",
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
# [10] HIGHEST BAD-RATE SEGMENTS
# ======================================================================

print_section(
    f"[10] TOP {TOP_N} SEGMENTS BY BAD RATE"
)


highest_bad_rate = (
    report
    .sort_values(
        by="bad_rate",
        ascending=False,
    )
)


print(
    highest_bad_rate[
        display_columns
    ]
    .head(TOP_N)
    .to_string(
        index=False
    )
)


# ======================================================================
# [11] SEGMENT SIZE CHECK
# ======================================================================

print_section(
    "[11] SEGMENT SIZE CHECK"
)


small_segments = (
    report[
        report["segment_n"]
        < MIN_SEGMENT_SIZE
    ]
)


low_bad_segments = (
    report[
        report["bad_count"]
        < MIN_BAD_COUNT
    ]
)


print(
    f"Segments below minimum N "
    f"({MIN_SEGMENT_SIZE:,}): "
    f"{len(small_segments)}"
)

print(
    f"Segments below minimum bad count "
    f"({MIN_BAD_COUNT}): "
    f"{len(low_bad_segments)}"
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

    valid_auc = (
        feature_report[
            "roc_auc"
        ]
        .dropna()
    )

    feature_summary_rows.append(
        {
            "segment_feature": feature,
            "segments": len(
                feature_report
            ),
            "valid_auc_segments":
                len(valid_auc),
            "min_roc_auc":
                valid_auc.min()
                if len(valid_auc) > 0
                else np.nan,
            "max_roc_auc":
                valid_auc.max()
                if len(valid_auc) > 0
                else np.nan,
            "min_auc_diff":
                feature_report[
                    "auc_diff_vs_overall"
                ].min(),
            "max_auc_diff":
                feature_report[
                    "auc_diff_vs_overall"
                ].max(),
            "degraded_segments":
                (
                    feature_report[
                        "performance_status"
                    ]
                    == "DEGRADED"
                ).sum(),
            "severe_degradation_segments":
                (
                    feature_report[
                        "performance_status"
                    ]
                    == "SEVERE_DEGRADATION"
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
        "metric": "overall_n",
        "value": len(analysis),
    },
    {
        "metric": "overall_bad_count",
        "value": int(y_true.sum()),
    },
    {
        "metric": "overall_bad_rate",
        "value": overall_bad_rate,
    },
    {
        "metric": "overall_roc_auc",
        "value": overall_roc_auc,
    },
    {
        "metric": "overall_pr_auc",
        "value": overall_pr_auc,
    },
    {
        "metric": "overall_gini",
        "value": overall_gini,
    },
    {
        "metric": "total_segments",
        "value": len(report),
    },
    {
        "metric": "ok_segments",
        "value": ok_count,
    },
    {
        "metric": "degraded_segments",
        "value": degraded_count,
    },
    {
        "metric": "severe_degradation_segments",
        "value": severe_count,
    },
    {
        "metric": "insufficient_data_segments",
        "value": insufficient_count,
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
    "Segment performance report saved to:"
)

print(
    f"  {RESULTS_PATH}"
)

print()
print(
    "Segment performance summary saved to:"
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
    "This analysis evaluates baseline model "
    "performance across predefined population segments."
)

print()
print(
    "Segments are created using quartiles calculated "
    "on the validation prediction sample."
)

print()
print(
    "The analysis is descriptive and does NOT "
    "evaluate statistical significance."
)

print()
print(
    "A negative AUC difference indicates that model "
    "discrimination is lower in the segment than "
    "in the overall validation population."
)

print()
print(
    "Configured AUC degradation thresholds:"
)

print(
    f"  AUC difference > "
    f"-{DEGRADATION_AUC_THRESHOLD:.2f}"
    "  -> OK"
)

print(
    f"  AUC difference <= "
    f"-{DEGRADATION_AUC_THRESHOLD:.2f}"
    "  -> DEGRADED"
)

print(
    f"  AUC difference <= "
    f"-{SEVERE_DEGRADATION_AUC_THRESHOLD:.2f}"
    "  -> SEVERE_DEGRADATION"
)

print()
print(
    "Important:"
)

print(
    "A segment classified as DEGRADED or "
    "SEVERE_DEGRADATION is a screening signal, "
    "not evidence of statistically significant "
    "model weakness."
)

print()
print(
    "Segments flagged by this analysis should be "
    "investigated further using statistical significance "
    "testing and business/materiality assessment."
)


# ======================================================================
# [15] FINAL CONCLUSION
# ======================================================================

print_section(
    "[15] VALIDATION CONCLUSION"
)


if severe_count > 0:

    print()
    print(
        f"Severe performance degradation was detected "
        f"in {severe_count} segment(s)."
    )

    print(
        "Further investigation is required."
    )

elif degraded_count > 0:

    print()
    print(
        f"Performance degradation was detected "
        f"in {degraded_count} segment(s)."
    )

    print(
        "Further investigation is recommended."
    )

else:

    print()
    print(
        "No material AUC degradation was detected "
        "using the configured descriptive thresholds."
    )


print()
print(
    "This result does NOT establish statistical "
    "significance or formal model risk."
)

print()
print("=" * 70)
print(
    "SEGMENT PERFORMANCE ANALYSIS COMPLETE"
)
print("=" * 70)