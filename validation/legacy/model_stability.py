from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ======================================================================
# NORTHSTAR MODEL VALIDATION — MODEL STABILITY / SEGMENT PERFORMANCE V0.1
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

BASELINE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "baseline"
)

PREDICTIONS_PATH = (
    BASELINE_DIR
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_stability"
)

SEGMENT_RESULTS_PATH = (
    OUTPUT_DIR
    / "segment_performance.csv"
)

SUMMARY_RESULTS_PATH = (
    OUTPUT_DIR
    / "model_stability_summary.csv"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

MIN_SEGMENT_SIZE = 1000

TOP_N = 20

AGE_BINS = [
    18,
    25,
    35,
    45,
    55,
    65,
    100,
]

CHILDREN_BINS = [
    -1,
    0,
    1,
    2,
    100,
]

NUMERIC_SEGMENT_FEATURES = [
    "AGE_YEARS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
]

CATEGORICAL_SEGMENT_FEATURES = [
    "CODE_GENDER",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]


# ======================================================================
# HELPERS
# ======================================================================


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def calculate_gini(roc_auc):
    if pd.isna(roc_auc):
        return np.nan

    return 2 * roc_auc - 1


def calculate_ks(y_true, y_score):
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_score": y_score,
        }
    ).sort_values(
        "y_score",
        ascending=False,
    )

    total_bad = data["y_true"].sum()
    total_good = len(data) - total_bad

    if total_bad == 0 or total_good == 0:
        return np.nan

    data["cum_bad"] = (
        data["y_true"].cumsum()
        / total_bad
    )

    data["cum_good"] = (
        (1 - data["y_true"]).cumsum()
        / total_good
    )

    return (
        data["cum_bad"]
        - data["cum_good"]
    ).abs().max()


def calculate_metrics(group):
    y_true = group["y_true"]
    y_probability = group["y_probability"]

    n = len(group)

    target_rate = y_true.mean()
    mean_probability = y_probability.mean()

    calibration_gap = (
        mean_probability
        - target_rate
    )

    if y_true.nunique() < 2:

        return {
            "n": n,
            "target_rate": target_rate,
            "mean_probability": mean_probability,
            "calibration_gap": calibration_gap,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "gini": np.nan,
            "ks": np.nan,
            "brier_score": brier_score_loss(
                y_true,
                y_probability,
            ),
            "log_loss": log_loss(
                y_true,
                y_probability,
                labels=[0, 1],
            ),
        }

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability,
    )

    ks = calculate_ks(
        y_true,
        y_probability,
    )

    brier = brier_score_loss(
        y_true,
        y_probability,
    )

    ll = log_loss(
        y_true,
        y_probability,
        labels=[0, 1],
    )

    return {
        "n": n,
        "target_rate": target_rate,
        "mean_probability": mean_probability,
        "calibration_gap": calibration_gap,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "gini": calculate_gini(roc_auc),
        "ks": ks,
        "brier_score": brier,
        "log_loss": ll,
    }


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print(
    "NORTHSTAR MODEL VALIDATION — "
    "MODEL STABILITY / SEGMENT PERFORMANCE V0.1"
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

df = pd.read_csv(DATA_PATH)

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

required_prediction_columns = {
    ID_COLUMN,
    "y_true",
    "y_probability",
    "y_prediction",
}

missing_columns = (
    required_prediction_columns
    - set(predictions.columns)
)

if missing_columns:

    raise ValueError(
        "Missing prediction columns: "
        f"{sorted(missing_columns)}"
    )

print("Required prediction columns: OK")

prediction_ids_unique = (
    predictions[ID_COLUMN].is_unique
)

print(
    f"Prediction IDs unique: "
    f"{prediction_ids_unique}"
)

if not prediction_ids_unique:

    raise ValueError(
        "Prediction IDs are not unique."
    )

missing_y_true = (
    predictions["y_true"]
    .isna()
    .sum()
)

missing_y_probability = (
    predictions["y_probability"]
    .isna()
    .sum()
)

print(
    f"Missing y_true:        "
    f"{missing_y_true}"
)

print(
    f"Missing y_probability: "
    f"{missing_y_probability}"
)

if (
    missing_y_true > 0
    or missing_y_probability > 0
):

    raise ValueError(
        "Missing values detected in predictions."
    )

probability_outside_range = (
    (
        predictions["y_probability"]
        < 0
    )
    | (
        predictions["y_probability"]
        > 1
    )
).sum()

print(
    f"Probability outside [0, 1]: "
    f"{probability_outside_range}"
)

if probability_outside_range > 0:

    raise ValueError(
        "Invalid probability values detected."
    )


# ======================================================================
# [3] MERGING PREDICTIONS WITH ORIGINAL DATA
# ======================================================================

print_section(
    "[3] MERGING PREDICTIONS WITH ORIGINAL DATA"
)

required_features = (
    CATEGORICAL_SEGMENT_FEATURES
    + [
        "DAYS_BIRTH",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
    ]
)

missing_features = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing segmentation features: "
        f"{missing_features}"
    )

merge_columns = [
    ID_COLUMN,
    TARGET,
] + required_features

original_subset = (
    df[merge_columns]
    .copy()
)

merged = predictions.merge(
    original_subset,
    on=ID_COLUMN,
    how="left",
    validate="one_to_one",
)

print(
    f"Merged shape:        "
    f"{merged.shape}"
)

if merged[TARGET].isna().any():

    raise ValueError(
        "Some prediction IDs could not be "
        "matched to the original dataset."
    )

target_mismatches = (
    merged[TARGET]
    != merged["y_true"]
).sum()

print(
    f"Target mismatches:    "
    f"{target_mismatches}"
)

if target_mismatches > 0:

    raise ValueError(
        "Target mismatch between predictions "
        "and original dataset."
    )

print("Prediction/data merge: OK")


# ======================================================================
# [4] OVERALL MODEL PERFORMANCE
# ======================================================================

print_section(
    "[4] OVERALL MODEL PERFORMANCE"
)

overall_metrics = calculate_metrics(
    merged
)

overall_n = overall_metrics["n"]
overall_target_rate = (
    overall_metrics["target_rate"]
)
overall_mean_probability = (
    overall_metrics["mean_probability"]
)
overall_calibration_gap = (
    overall_metrics["calibration_gap"]
)
overall_roc_auc = (
    overall_metrics["roc_auc"]
)
overall_pr_auc = (
    overall_metrics["pr_auc"]
)
overall_gini = (
    overall_metrics["gini"]
)
overall_ks = (
    overall_metrics["ks"]
)
overall_brier = (
    overall_metrics["brier_score"]
)
overall_log_loss = (
    overall_metrics["log_loss"]
)

print(
    f"N:                    "
    f"{overall_n:,}"
)

print(
    f"Target rate:          "
    f"{overall_target_rate:.4%}"
)

print(
    f"Mean probability:     "
    f"{overall_mean_probability:.4%}"
)

print(
    f"Calibration gap:      "
    f"{overall_calibration_gap:+.4%}"
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

print(
    f"KS:                   "
    f"{overall_ks:.4f}"
)

print(
    f"Brier score:          "
    f"{overall_brier:.6f}"
)

print(
    f"Log Loss:             "
    f"{overall_log_loss:.6f}"
)


# ======================================================================
# [5] DERIVING SEGMENTATION VARIABLES
# ======================================================================

print_section(
    "[5] DERIVING SEGMENTATION VARIABLES"
)

# ----------------------------------------------------------------------
# Age
# ----------------------------------------------------------------------

merged["AGE_YEARS"] = (
    -merged["DAYS_BIRTH"]
    / 365.25
)

merged["AGE_BAND"] = pd.cut(
    merged["AGE_YEARS"],
    bins=AGE_BINS,
    right=False,
)

# ----------------------------------------------------------------------
# Children
# ----------------------------------------------------------------------

merged["CHILDREN_BAND"] = pd.cut(
    merged["CNT_CHILDREN"],
    bins=CHILDREN_BINS,
    right=True,
)

# ----------------------------------------------------------------------
# Income quantiles
# ----------------------------------------------------------------------

merged["INCOME_BAND"] = pd.qcut(
    merged["AMT_INCOME_TOTAL"],
    q=5,
    duplicates="drop",
)

# ----------------------------------------------------------------------
# Credit quantiles
# ----------------------------------------------------------------------

merged["CREDIT_BAND"] = pd.qcut(
    merged["AMT_CREDIT"],
    q=5,
    duplicates="drop",
)


# ======================================================================
# [6] DEFINING SEGMENTS
# ======================================================================

print_section(
    "[6] DEFINING SEGMENTS"
)

segments = {
    "CODE_GENDER": "CODE_GENDER",
    "NAME_EDUCATION_TYPE": "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS": "NAME_FAMILY_STATUS",
    "FLAG_OWN_CAR": "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY": "FLAG_OWN_REALTY",
    "AGE_BAND": "AGE_BAND",
    "CHILDREN_BAND": "CHILDREN_BAND",
    "INCOME_BAND": "INCOME_BAND",
    "CREDIT_BAND": "CREDIT_BAND",
}

for segment_name, column in segments.items():

    n_groups = (
        merged[column]
        .nunique(dropna=False)
    )

    print(
        f"{segment_name:<25} "
        f"{n_groups} groups"
    )


# ======================================================================
# [7] SEGMENT PERFORMANCE ANALYSIS
# ======================================================================

print_section(
    "[7] SEGMENT PERFORMANCE ANALYSIS"
)

results = []

for segment_name, column in segments.items():

    grouped = merged.groupby(
        column,
        dropna=False,
        observed=True,
    )

    for segment_value, group in grouped:

        metrics = calculate_metrics(
            group
        )

        n = metrics["n"]

        if n < MIN_SEGMENT_SIZE:

            status = "INSUFFICIENT_SAMPLE"

        elif pd.isna(metrics["roc_auc"]):

            status = "UNDEFINED_METRIC"

        else:

            status = "EVALUATED"

        roc_auc_delta = (
            metrics["roc_auc"]
            - overall_roc_auc
            if not pd.isna(
                metrics["roc_auc"]
            )
            else np.nan
        )

        pr_auc_delta = (
            metrics["pr_auc"]
            - overall_pr_auc
            if not pd.isna(
                metrics["pr_auc"]
            )
            else np.nan
        )

        brier_delta = (
            metrics["brier_score"]
            - overall_brier
            if not pd.isna(
                metrics["brier_score"]
            )
            else np.nan
        )

        calibration_gap_delta = (
            metrics["calibration_gap"]
            - overall_calibration_gap
        )

        result = {
            "segment": segment_name,
            "segment_value": str(
                segment_value
            ),
            "n": n,
            "sample_share": (
                n / overall_n
            ),
            "target_rate": (
                metrics["target_rate"]
            ),
            "mean_probability": (
                metrics["mean_probability"]
            ),
            "calibration_gap": (
                metrics["calibration_gap"]
            ),
            "calibration_gap_delta": (
                calibration_gap_delta
            ),
            "roc_auc": (
                metrics["roc_auc"]
            ),
            "roc_auc_delta": (
                roc_auc_delta
            ),
            "pr_auc": (
                metrics["pr_auc"]
            ),
            "pr_auc_delta": (
                pr_auc_delta
            ),
            "gini": (
                metrics["gini"]
            ),
            "ks": (
                metrics["ks"]
            ),
            "brier_score": (
                metrics["brier_score"]
            ),
            "brier_delta": (
                brier_delta
            ),
            "log_loss": (
                metrics["log_loss"]
            ),
            "status": status,
        }

        results.append(result)


segment_report = pd.DataFrame(
    results
)


# ======================================================================
# [8] SEGMENT SUMMARY
# ======================================================================

print_section(
    "[8] SEGMENT SUMMARY"
)

evaluated = segment_report[
    segment_report["status"]
    == "EVALUATED"
]

insufficient = segment_report[
    segment_report["status"]
    == "INSUFFICIENT_SAMPLE"
]

undefined = segment_report[
    segment_report["status"]
    == "UNDEFINED_METRIC"
]

print(
    f"Total segment groups:     "
    f"{len(segment_report)}"
)

print(
    f"Evaluated:                "
    f"{len(evaluated)}"
)

print(
    f"Insufficient sample:      "
    f"{len(insufficient)}"
)

print(
    f"Undefined metric:         "
    f"{len(undefined)}"
)


# ======================================================================
# [9] LOWEST AUC SEGMENTS
# ======================================================================

print_section(
    f"[9] LOWEST ROC-AUC SEGMENTS — TOP {TOP_N}"
)

lowest_auc = (
    evaluated
    .sort_values(
        by="roc_auc",
        ascending=True,
    )
)

print(
    lowest_auc[
        [
            "segment",
            "segment_value",
            "n",
            "target_rate",
            "roc_auc",
            "roc_auc_delta",
            "pr_auc",
            "pr_auc_delta",
        ]
    ]
    .head(TOP_N)
    .to_string(index=False)
)


# ======================================================================
# [10] CALIBRATION
# ======================================================================

print_section(
    f"[10] LARGEST CALIBRATION GAPS — TOP {TOP_N}"
)

calibration_report = (
    evaluated
    .copy()
)

calibration_report[
    "abs_calibration_gap"
] = (
    calibration_report[
        "calibration_gap"
    ]
    .abs()
)

calibration_report = (
    calibration_report
    .sort_values(
        by="abs_calibration_gap",
        ascending=False,
    )
)

print(
    calibration_report[
        [
            "segment",
            "segment_value",
            "n",
            "target_rate",
            "mean_probability",
            "calibration_gap",
        ]
    ]
    .head(TOP_N)
    .to_string(index=False)
)


# ======================================================================
# [11] PERFORMANCE RANGE
# ======================================================================

print_section(
    "[11] PERFORMANCE HETEROGENEITY"
)

if len(evaluated) > 0:

    auc_min = (
        evaluated["roc_auc"]
        .min()
    )

    auc_max = (
        evaluated["roc_auc"]
        .max()
    )

    auc_range = (
        auc_max
        - auc_min
    )

    pr_auc_min = (
        evaluated["pr_auc"]
        .min()
    )

    pr_auc_max = (
        evaluated["pr_auc"]
        .max()
    )

    pr_auc_range = (
        pr_auc_max
        - pr_auc_min
    )

    print(
        f"ROC-AUC minimum:          "
        f"{auc_min:.4f}"
    )

    print(
        f"ROC-AUC maximum:          "
        f"{auc_max:.4f}"
    )

    print(
        f"ROC-AUC range:            "
        f"{auc_range:.4f}"
    )

    print()

    print(
        f"PR-AUC minimum:           "
        f"{pr_auc_min:.4f}"
    )

    print(
        f"PR-AUC maximum:           "
        f"{pr_auc_max:.4f}"
    )

    print(
        f"PR-AUC range:             "
        f"{pr_auc_range:.4f}"
    )


# ======================================================================
# [12] SAMPLE SIZE CONTROL
# ======================================================================

print_section(
    "[12] SAMPLE SIZE CONTROL"
)

if len(insufficient) > 0:

    print(
        f"Segments below minimum sample "
        f"size ({MIN_SEGMENT_SIZE:,}): "
        f"{len(insufficient)}"
    )

    print()

    print(
        insufficient[
            [
                "segment",
                "segment_value",
                "n",
            ]
        ]
        .sort_values(
            by="n",
            ascending=True,
        )
        .head(TOP_N)
        .to_string(index=False)
    )

else:

    print(
        "All segment groups satisfy the "
        "minimum sample size requirement."
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

segment_report = (
    segment_report
    .sort_values(
        by=[
            "segment",
            "roc_auc",
        ],
        ascending=[
            True,
            True,
        ],
        na_position="last",
    )
    .reset_index(drop=True)
)

segment_report.to_csv(
    SEGMENT_RESULTS_PATH,
    index=False,
)


# ----------------------------------------------------------------------
# Summary file
# ----------------------------------------------------------------------

summary_rows = [
    {
        "metric": "overall_n",
        "value": overall_n,
    },
    {
        "metric": "overall_target_rate",
        "value": overall_target_rate,
    },
    {
        "metric": "overall_mean_probability",
        "value": overall_mean_probability,
    },
    {
        "metric": "overall_calibration_gap",
        "value": overall_calibration_gap,
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
        "metric": "overall_ks",
        "value": overall_ks,
    },
    {
        "metric": "overall_brier_score",
        "value": overall_brier,
    },
    {
        "metric": "overall_log_loss",
        "value": overall_log_loss,
    },
    {
        "metric": "total_segment_groups",
        "value": len(segment_report),
    },
    {
        "metric": "evaluated_segment_groups",
        "value": len(evaluated),
    },
    {
        "metric": "insufficient_sample_groups",
        "value": len(insufficient),
    },
    {
        "metric": "undefined_metric_groups",
        "value": len(undefined),
    },
]

if len(evaluated) > 0:

    summary_rows.extend(
        [
            {
                "metric": "segment_auc_min",
                "value": auc_min,
            },
            {
                "metric": "segment_auc_max",
                "value": auc_max,
            },
            {
                "metric": "segment_auc_range",
                "value": auc_range,
            },
            {
                "metric": "segment_pr_auc_min",
                "value": pr_auc_min,
            },
            {
                "metric": "segment_pr_auc_max",
                "value": pr_auc_max,
            },
            {
                "metric": "segment_pr_auc_range",
                "value": pr_auc_range,
            },
        ]
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY_RESULTS_PATH,
    index=False,
)

print()
print(
    "Segment performance report saved to:"
)

print(
    f"  {SEGMENT_RESULTS_PATH}"
)

print()
print(
    "Model stability summary saved to:"
)

print(
    f"  {SUMMARY_RESULTS_PATH}"
)


# ======================================================================
# [14] VALIDATION INTERPRETATION
# ======================================================================

print_section(
    "[14] VALIDATION INTERPRETATION"
)

print()
print(
    "This analysis evaluates model performance "
    "and calibration across predefined population "
    "segments."
)

print()
print(
    "The objective is to identify material "
    "performance heterogeneity between segments."
)

print()
print(
    "Primary validation dimensions:"
)

print(
    "  - discrimination"
)

print(
    "  - calibration"
)

print(
    "  - performance degradation"
)

print(
    "  - sample size sufficiency"
)

print()
print(
    "Segment-level differences do not automatically "
    "constitute model failure or model risk."
)

print()
print(
    "Observed differences should be investigated "
    "in the context of segment size, population "
    "characteristics, business relevance, and "
    "statistical uncertainty."
)

print()
print(
    "Insufficiently large segments should not be "
    "used as standalone evidence of model weakness."
)


# ======================================================================
# [15] VALIDATION CONCLUSION
# ======================================================================

print_section(
    "[15] VALIDATION CONCLUSION"
)

if len(evaluated) == 0:

    print()
    print(
        "No segment groups met the minimum sample "
        "size requirement for performance evaluation."
    )

else:

    worst_auc_row = (
        evaluated
        .sort_values(
            by="roc_auc",
            ascending=True,
        )
        .iloc[0]
    )

    largest_calibration_row = (
        evaluated
        .assign(
            abs_calibration_gap=lambda x:
                x["calibration_gap"].abs()
        )
        .sort_values(
            by="abs_calibration_gap",
            ascending=False,
        )
        .iloc[0]
    )

    print()
    print(
        "Lowest observed segment ROC-AUC:"
    )

    print(
        f"  Segment:             "
        f"{worst_auc_row['segment']}"
    )

    print(
        f"  Value:               "
        f"{worst_auc_row['segment_value']}"
    )

    print(
        f"  N:                   "
        f"{int(worst_auc_row['n']):,}"
    )

    print(
        f"  ROC-AUC:             "
        f"{worst_auc_row['roc_auc']:.4f}"
    )

    print(
        f"  Delta vs overall:    "
        f"{worst_auc_row['roc_auc_delta']:+.4f}"
    )

    print()
    print(
        "Largest absolute calibration gap:"
    )

    print(
        f"  Segment:             "
        f"{largest_calibration_row['segment']}"
    )

    print(
        f"  Value:               "
        f"{largest_calibration_row['segment_value']}"
    )

    print(
        f"  N:                   "
        f"{int(largest_calibration_row['n']):,}"
    )

    print(
        f"  Calibration gap:     "
        f"{largest_calibration_row['calibration_gap']:+.4%}"
    )

    print()
    print(
        "These observations are diagnostic findings "
        "and require contextual investigation before "
        "being classified as material model risk."
    )


print()
print("=" * 70)
print(
    "MODEL STABILITY / SEGMENT PERFORMANCE COMPLETE"
)
print("=" * 70)