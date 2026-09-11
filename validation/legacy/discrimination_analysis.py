"""
======================================================================
NORTHSTAR MODEL VALIDATION — DISCRIMINATION & THRESHOLD ANALYSIS V0.1
======================================================================

Purpose:
    Independent validation of model discrimination, risk ranking,
    risk-band monotonicity, threshold behaviour, and segment
    discrimination.

Inputs:
    - Original dataset containing:
        SK_ID_CURR
        TARGET
    - Prediction file containing:
        SK_ID_CURR
        predicted probability

Outputs:
    - Console validation report
    - Risk-band table
    - Threshold analysis table
    - Segment discrimination table
    - CSV output files

Notes:
    - No arbitrary AUC pass/fail threshold is imposed.
    - Calibration is intentionally NOT repeated here.
    - This module focuses on discrimination and ranking behaviour.
======================================================================
"""

from pathlib import Path

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

DATA_PATH = "data/raw/application_train.csv"
PREDICTIONS_PATH = "reports/baseline/validation_predictions.csv"

OUTPUT_DIR = "reports/discrimination"

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

# Candidate prediction column names.
# The script will automatically detect the first one present.
PREDICTION_CANDIDATES = [
    "prediction",
    "predicted_probability",
    "predicted_prob",
    "probability",
    "pd",
    "y_pred_proba",
    "prediction_probability",
    "y_probability",
]

N_RISK_BANDS = 10

# Thresholds are intentionally broad.
# They are analytical operating points, not recommended business cutoffs.
THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
]

# Minimum observations required for segment-level discrimination.
MIN_SEGMENT_SIZE = 500

# Segment definitions.
# These are deliberately simple and based on variables available
# in the Home Credit dataset.
SEGMENTS = {
    "CODE_GENDER": "CODE_GENDER",
    "NAME_CONTRACT_TYPE": "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE": "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE": "NAME_EDUCATION_TYPE",
}


# ======================================================================
# HELPERS
# ======================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def detect_prediction_column(df):
    """
    Detect prediction probability column.
    """

    for column in PREDICTION_CANDIDATES:
        if column in df.columns:
            return column

    # Fallback:
    # Look for a numeric column that is not the ID.
    excluded = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    numeric_candidates = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column not in excluded
    ]

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Could not automatically identify prediction probability column. "
        f"Available columns: {list(df.columns)}"
    )


def gini_from_auc(auc):
    return 2.0 * auc - 1.0


def ks_statistic(y_true, y_score):
    """
    Kolmogorov-Smirnov statistic for binary classification.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    differences = tpr - fpr

    index = np.argmax(differences)

    return {
        "ks": differences[index],
        "threshold": thresholds[index],
    }


def safe_auc(y_true, y_score):
    """
    AUC requires both classes to be present.
    """

    if y_true.nunique() < 2:
        return np.nan

    return roc_auc_score(y_true, y_score)


def safe_pr_auc(y_true, y_score):
    """
    PR-AUC requires at least one positive observation.
    """

    if y_true.sum() == 0:
        return np.nan

    return average_precision_score(y_true, y_score)


# ======================================================================
# [1] LOADING DATA
# ======================================================================

print_header("[1] LOADING DATA")

dataset = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

print(f"Dataset shape:       {dataset.shape}")
print(f"Predictions shape:   {predictions.shape}")


# ======================================================================
# [2] BASIC VALIDATION CHECKS
# ======================================================================

print_header("[2] BASIC VALIDATION CHECKS")

# ----------------------------------------------------------------------
# Required columns
# ----------------------------------------------------------------------

if ID_COLUMN not in dataset.columns:
    raise ValueError(
        f"Required dataset column missing: {ID_COLUMN}"
    )

if TARGET_COLUMN not in dataset.columns:
    raise ValueError(
        f"Required dataset column missing: {TARGET_COLUMN}"
    )

if ID_COLUMN not in predictions.columns:
    raise ValueError(
        f"Required prediction ID column missing: {ID_COLUMN}"
    )

prediction_column = detect_prediction_column(predictions)

print(f"Prediction column:   {prediction_column}")
print("Required columns:    OK")


# ----------------------------------------------------------------------
# Prediction ID uniqueness
# ----------------------------------------------------------------------

prediction_ids_unique = predictions[ID_COLUMN].is_unique

print(f"Prediction IDs unique: {prediction_ids_unique}")

if not prediction_ids_unique:
    raise ValueError("Prediction IDs are not unique.")


# ----------------------------------------------------------------------
# Target validation
# ----------------------------------------------------------------------

target_values = set(dataset[TARGET_COLUMN].dropna().unique())

print(f"Target values:       {sorted(target_values)}")

if not target_values.issubset({0, 1}):
    raise ValueError(
        f"TARGET must contain only 0/1. Found: {target_values}"
    )


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------

required_prediction_columns = [
    ID_COLUMN,
    prediction_column,
]

analysis = dataset[
    [ID_COLUMN, TARGET_COLUMN]
    + [
        column
        for column in SEGMENTS.values()
        if column in dataset.columns
    ]
].merge(
    predictions[required_prediction_columns],
    on=ID_COLUMN,
    how="inner",
    validate="one_to_one",
)

print(f"Merged observations: {len(analysis):,}")


# ----------------------------------------------------------------------
# Coverage
# ----------------------------------------------------------------------

coverage = len(analysis) / len(predictions)

print(f"Prediction coverage: {coverage:.4%}")

if len(analysis) == 0:
    raise ValueError("No observations after merging dataset and predictions.")


# ----------------------------------------------------------------------
# Prediction validity
# ----------------------------------------------------------------------

prediction_missing = analysis[prediction_column].isna().sum()

prediction_min = analysis[prediction_column].min()
prediction_max = analysis[prediction_column].max()

print(f"Missing predictions: {prediction_missing:,}")
print(f"Prediction minimum:  {prediction_min:.8f}")
print(f"Prediction maximum:  {prediction_max:.8f}")

if prediction_missing > 0:
    raise ValueError("Missing prediction probabilities detected.")

if prediction_min < 0 or prediction_max > 1:
    raise ValueError(
        "Prediction probabilities must be within [0, 1]."
    )


# ======================================================================
# [3] OVERALL DISCRIMINATION
# ======================================================================

print_header("[3] OVERALL DISCRIMINATION")

y_true = analysis[TARGET_COLUMN]
y_score = analysis[prediction_column]

positive_count = int(y_true.sum())
negative_count = int((y_true == 0).sum())

positive_rate = positive_count / len(y_true)

print(f"Observations:        {len(y_true):,}")
print(f"TARGET=1:            {positive_count:,}")
print(f"TARGET=0:            {negative_count:,}")
print(f"Default rate:        {positive_rate:.4%}")


# ----------------------------------------------------------------------
# ROC-AUC
# ----------------------------------------------------------------------

roc_auc = safe_auc(y_true, y_score)

print()
print(f"ROC-AUC:             {roc_auc:.6f}")


# ----------------------------------------------------------------------
# PR-AUC
# ----------------------------------------------------------------------

pr_auc = safe_pr_auc(y_true, y_score)

print(f"PR-AUC:              {pr_auc:.6f}")


# ----------------------------------------------------------------------
# Gini
# ----------------------------------------------------------------------

gini = gini_from_auc(roc_auc)

print(f"Gini:                {gini:.6f}")


# ----------------------------------------------------------------------
# KS
# ----------------------------------------------------------------------

ks_result = ks_statistic(y_true, y_score)

ks = ks_result["ks"]
ks_threshold = ks_result["threshold"]

print(f"KS statistic:        {ks:.6f}")
print(f"KS threshold:        {ks_threshold:.6f}")


# ======================================================================
# [4] RISK-BAND ANALYSIS
# ======================================================================

print_header("[4] RISK-BAND ANALYSIS")

# Higher prediction = higher risk.
# qcut creates approximately equal-sized population bands.

analysis["RISK_BAND"] = pd.qcut(
    analysis[prediction_column],
    q=N_RISK_BANDS,
    labels=False,
    duplicates="drop",
)

analysis["RISK_BAND"] = analysis["RISK_BAND"] + 1


risk_bands = (
    analysis
    .groupby("RISK_BAND", observed=True)
    .agg(
        observations=(ID_COLUMN, "count"),
        predicted_pd=(prediction_column, "mean"),
        observed_default_rate=(TARGET_COLUMN, "mean"),
        defaults=(TARGET_COLUMN, "sum"),
    )
    .reset_index()
)

risk_bands["population_share"] = (
    risk_bands["observations"] / len(analysis)
)

risk_bands["default_capture"] = (
    risk_bands["defaults"] / positive_count
)

risk_bands["cumulative_default_capture"] = (
    risk_bands["defaults"].cumsum() / positive_count
)


# ----------------------------------------------------------------------
# Monotonicity
# ----------------------------------------------------------------------

observed_rates = risk_bands["observed_default_rate"].to_numpy()

monotonic_pairs = np.sum(
    np.diff(observed_rates) >= 0
)

total_pairs = max(len(observed_rates) - 1, 0)

monotonicity_rate = (
    monotonic_pairs / total_pairs
    if total_pairs > 0
    else np.nan
)

risk_bands["monotonic_with_previous"] = (
    risk_bands["observed_default_rate"]
    .diff()
    >= 0
)

risk_bands.loc[
    risk_bands["RISK_BAND"] == risk_bands["RISK_BAND"].min(),
    "monotonic_with_previous"
] = True


print()
print(risk_bands.to_string(index=False))

print()
print(
    f"Monotonicity:       "
    f"{monotonic_pairs}/{total_pairs} adjacent pairs "
    f"non-decreasing ({monotonicity_rate:.2%})"
)


# ----------------------------------------------------------------------
# Top risk bands
# ----------------------------------------------------------------------

top_band = risk_bands.iloc[-1]

bottom_band = risk_bands.iloc[0]

print()
print(
    f"Lowest-risk band default rate:  "
    f"{bottom_band['observed_default_rate']:.4%}"
)

print(
    f"Highest-risk band default rate: "
    f"{top_band['observed_default_rate']:.4%}"
)

print(
    f"Risk separation ratio:           "
    f"{top_band['observed_default_rate'] / bottom_band['observed_default_rate']:.2f}x"
)


# ======================================================================
# [5] THRESHOLD ANALYSIS
# ======================================================================

print_header("[5] THRESHOLD ANALYSIS")

threshold_rows = []

for threshold in THRESHOLDS:

    predicted_positive = y_score >= threshold

    tp = int(((predicted_positive) & (y_true == 1)).sum())
    fp = int(((predicted_positive) & (y_true == 0)).sum())
    tn = int(((~predicted_positive) & (y_true == 0)).sum())
    fn = int(((~predicted_positive) & (y_true == 1)).sum())

    predicted_positive_count = tp + fp

    approval_rate = 1.0 - (
        predicted_positive_count / len(y_true)
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    default_rate_flagged = (
        y_true[predicted_positive].mean()
        if predicted_positive_count > 0
        else np.nan
    )

    threshold_rows.append(
        {
            "threshold": threshold,
            "flagged_count": predicted_positive_count,
            "flagged_rate": predicted_positive_count / len(y_true),
            "approval_rate": approval_rate,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "specificity": specificity,
            "default_capture": recall,
            "default_rate_flagged": default_rate_flagged,
        }
    )


threshold_analysis = pd.DataFrame(threshold_rows)

print(threshold_analysis.to_string(index=False))


# ======================================================================
# [6] SEGMENT DISCRIMINATION
# ======================================================================

print_header("[6] SEGMENT DISCRIMINATION")

segment_rows = []

for segment_name, segment_column in SEGMENTS.items():

    if segment_column not in analysis.columns:
        print(
            f"Skipping {segment_name}: "
            f"column {segment_column} not available."
        )
        continue

    for segment_value, segment_df in analysis.groupby(
        segment_column,
        dropna=False
    ):

        n = len(segment_df)

        if n < MIN_SEGMENT_SIZE:
            continue

        segment_y = segment_df[TARGET_COLUMN]
        segment_score = segment_df[prediction_column]

        if segment_y.nunique() < 2:
            continue

        segment_auc = safe_auc(
            segment_y,
            segment_score
        )

        segment_pr_auc = safe_pr_auc(
            segment_y,
            segment_score
        )

        segment_ks_result = ks_statistic(
            segment_y,
            segment_score
        )

        segment_rows.append(
            {
                "segment": segment_name,
                "segment_value": segment_value,
                "observations": n,
                "default_rate": segment_y.mean(),
                "roc_auc": segment_auc,
                "pr_auc": segment_pr_auc,
                "gini": gini_from_auc(segment_auc),
                "ks": segment_ks_result["ks"],
            }
        )


segment_analysis = pd.DataFrame(segment_rows)

if len(segment_analysis) > 0:

    print(
        segment_analysis
        .sort_values(["segment", "roc_auc"])
        .to_string(index=False)
    )

else:

    print("No eligible segments for discrimination analysis.")


# ======================================================================
# [7] STABILITY / CONSISTENCY CHECKS
# ======================================================================

print_header("[7] STABILITY / CONSISTENCY CHECKS")

print(
    f"Overall ROC-AUC:       {roc_auc:.6f}"
)

print(
    f"Overall PR-AUC:        {pr_auc:.6f}"
)

print(
    f"Overall KS:            {ks:.6f}"
)

print(
    f"Risk-band monotonicity:{monotonicity_rate:.2%}"
)


# ----------------------------------------------------------------------
# Segment degradation
# ----------------------------------------------------------------------

if len(segment_analysis) > 0:

    segment_auc_min = segment_analysis["roc_auc"].min()
    segment_auc_max = segment_analysis["roc_auc"].max()

    segment_auc_mean = segment_analysis["roc_auc"].mean()

    print()
    print(f"Segment ROC-AUC min:   {segment_auc_min:.6f}")
    print(f"Segment ROC-AUC mean:  {segment_auc_mean:.6f}")
    print(f"Segment ROC-AUC max:   {segment_auc_max:.6f}")

    weakest_segment = segment_analysis.loc[
        segment_analysis["roc_auc"].idxmin()
    ]

    strongest_segment = segment_analysis.loc[
        segment_analysis["roc_auc"].idxmax()
    ]

    print()
    print(
        "Weakest segment:      "
        f"{weakest_segment['segment']} = "
        f"{weakest_segment['segment_value']} "
        f"(AUC={weakest_segment['roc_auc']:.6f})"
    )

    print(
        "Strongest segment:    "
        f"{strongest_segment['segment']} = "
        f"{strongest_segment['segment_value']} "
        f"(AUC={strongest_segment['roc_auc']:.6f})"
    )


# ======================================================================
# [8] VALIDATION FINDINGS
# ======================================================================

print_header("[8] VALIDATION FINDINGS")

print()
print("DISCRIMINATION")
print("-" * 70)

print(f"ROC-AUC: {roc_auc:.6f}")
print(f"PR-AUC:  {pr_auc:.6f}")
print(f"Gini:    {gini:.6f}")
print(f"KS:      {ks:.6f}")

print()
print("RISK RANKING")
print("-" * 70)

print(
    f"Risk-band monotonicity: "
    f"{monotonicity_rate:.2%}"
)

print(
    f"Lowest risk band default rate:  "
    f"{bottom_band['observed_default_rate']:.4%}"
)

print(
    f"Highest risk band default rate: "
    f"{top_band['observed_default_rate']:.4%}"
)

print()
print("THRESHOLD BEHAVIOUR")
print("-" * 70)

# Show a few representative operating points.
representative_thresholds = threshold_analysis[
    threshold_analysis["threshold"].isin(
        [0.05, 0.10, 0.20, 0.50]
    )
]

print(
    representative_thresholds.to_string(index=False)
)


print()
print("VALIDATION INTERPRETATION")
print("-" * 70)

# ----------------------------------------------------------------------
# Finding 1: ranking
# ----------------------------------------------------------------------

if monotonicity_rate >= 0.90:
    print(
        "[OBSERVATION] Risk-band default rates show strong "
        "monotonicity."
    )
elif monotonicity_rate >= 0.70:
    print(
        "[OBSERVATION] Risk-band default rates show generally "
        "monotonic behaviour with some deviations."
    )
else:
    print(
        "[OBSERVATION] Risk-band default rates show material "
        "non-monotonicity."
    )


# ----------------------------------------------------------------------
# Finding 2: separation
# ----------------------------------------------------------------------

if (
    bottom_band["observed_default_rate"] > 0
    and top_band["observed_default_rate"]
    / bottom_band["observed_default_rate"] >= 2
):
    print(
        "[OBSERVATION] The highest-risk band exhibits materially "
        "higher observed default frequency than the lowest-risk band."
    )
else:
    print(
        "[OBSERVATION] Separation between the lowest- and "
        "highest-risk bands is limited."
    )


# ----------------------------------------------------------------------
# Finding 3: segment behaviour
# ----------------------------------------------------------------------

if len(segment_analysis) > 0:

    weak_segment_count = int(
        (
            segment_analysis["roc_auc"]
            < roc_auc - 0.10
        ).sum()
    )

    print(
        f"[OBSERVATION] Segments analysed: "
        f"{len(segment_analysis):,}"
    )

    print(
        f"[OBSERVATION] Segments with AUC more than "
        f"0.10 below overall AUC: {weak_segment_count:,}"
    )


# ======================================================================
# [9] OUTPUTS
# ======================================================================

print_header("[9] OUTPUTS")

os.makedirs(OUTPUT_DIR, exist_ok=True)

risk_bands_path = os.path.join(
    OUTPUT_DIR,
    "risk_bands.csv"
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "threshold_analysis.csv"
)

segment_path = os.path.join(
    OUTPUT_DIR,
    "segment_discrimination.csv"
)

risk_bands.to_csv(
    risk_bands_path,
    index=False
)

threshold_analysis.to_csv(
    threshold_path,
    index=False
)

if len(segment_analysis) > 0:
    segment_analysis.to_csv(
        segment_path,
        index=False
    )

print(f"Risk bands:          {risk_bands_path}")
print(f"Threshold analysis:  {threshold_path}")

if len(segment_analysis) > 0:
    print(f"Segment analysis:    {segment_path}")


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print_header("NORTHSTAR DISCRIMINATION ANALYSIS COMPLETE")

print(f"Observations:         {len(analysis):,}")
print(f"ROC-AUC:              {roc_auc:.6f}")
print(f"PR-AUC:               {pr_auc:.6f}")
print(f"Gini:                 {gini:.6f}")
print(f"KS:                   {ks:.6f}")
print(f"Risk-band monotonicity:{monotonicity_rate:.2%}")

print()
print("Output directory:")
print(f"  {OUTPUT_DIR}")

print("=" * 70)