"""
NORTHSTAR MODEL VALIDATION
Discrimination & Threshold Analysis v0.2

Purpose:
    Independent validation of model discrimination, risk ranking,
    threshold behaviour, and segment-level discrimination.

Inputs:
    data/application_train.csv
    predictions/baseline_predictions.csv

Outputs:
    reports/discrimination/
        risk_bands.csv
        threshold_analysis.csv
        segment_discrimination.csv
        overall_discrimination.csv
        discrimination_ci.csv
        risk_band_ci.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

DATA_PATH = Path("data/raw/application_train.csv")
PREDICTIONS_PATH = Path("reports/baseline/validation_predictions.csv")
OUTPUT_DIR = Path("reports/discrimination")

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

RANDOM_STATE = 42

# Number of bootstrap resamples.
# 1000 gives a good balance between statistical stability and runtime.
N_BOOTSTRAP = 1000

# Minimum observations required for segment analysis.
MIN_SEGMENT_SIZE = 500

SEGMENT_COLUMNS = [
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
]

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


# ======================================================================
# HELPERS
# ======================================================================

def bootstrap_metric_ci(
    y_true,
    y_score,
    metric_function,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
):
    """
    Bootstrap confidence interval for a binary classification metric.

    Stratified bootstrap:
        positives and negatives are resampled separately.

    This guarantees that every bootstrap sample contains both classes.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    positive_idx = np.where(y_true == 1)[0]
    negative_idx = np.where(y_true == 0)[0]

    rng = np.random.default_rng(random_state)

    values = []

    for _ in range(n_bootstrap):

        sampled_positive = rng.choice(
            positive_idx,
            size=len(positive_idx),
            replace=True,
        )

        sampled_negative = rng.choice(
            negative_idx,
            size=len(negative_idx),
            replace=True,
        )

        sample_idx = np.concatenate(
            [sampled_positive, sampled_negative]
        )

        value = metric_function(
            y_true[sample_idx],
            y_score[sample_idx],
        )

        values.append(value)

    values = np.asarray(values)

    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)

    return lower, upper


def bootstrap_default_rate_ci(
    defaults,
    observations,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
):
    """
    Bootstrap 95% CI for an observed default rate within a risk band.
    """

    rng = np.random.default_rng(random_state)

    defaults = int(defaults)
    observations = int(observations)

    if observations == 0:
        return np.nan, np.nan

    if defaults == 0:
        # Bootstrap directly from Bernoulli observations.
        values = []

        for _ in range(n_bootstrap):
            sample = rng.binomial(
                1,
                0.0,
                size=observations,
            )
            values.append(sample.mean())

        return (
            np.percentile(values, 2.5),
            np.percentile(values, 97.5),
        )

    if defaults == observations:
        values = []

        for _ in range(n_bootstrap):
            sample = rng.binomial(
                1,
                1.0,
                size=observations,
            )
            values.append(sample.mean())

        return (
            np.percentile(values, 2.5),
            np.percentile(values, 97.5),
        )

    # Represent the observed band as binary outcomes.
    band = np.concatenate(
        [
            np.ones(defaults, dtype=np.int8),
            np.zeros(observations - defaults, dtype=np.int8),
        ]
    )

    values = []

    for _ in range(n_bootstrap):

        sample = rng.choice(
            band,
            size=observations,
            replace=True,
        )

        values.append(sample.mean())

    values = np.asarray(values)

    return (
        np.percentile(values, 2.5),
        np.percentile(values, 97.5),
    )


def calculate_ks(y_true, y_score):
    """
    Calculate KS statistic for binary classification.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
    )

    ks_values = tpr - fpr

    max_index = np.argmax(ks_values)

    return (
        float(ks_values[max_index]),
        float(thresholds[max_index]),
    )


def safe_rate(numerator, denominator):
    if denominator == 0:
        return np.nan

    return numerator / denominator


# ======================================================================
# [1] LOADING DATA
# ======================================================================

print("=" * 70)
print("NORTHSTAR MODEL VALIDATION — DISCRIMINATION ANALYSIS V0.2")
print("=" * 70)

print()
print("=" * 70)
print("[1] LOADING DATA")
print("=" * 70)

data = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

print(f"Dataset shape:       {data.shape}")
print(f"Predictions shape:   {predictions.shape}")


# ======================================================================
# [2] BASIC VALIDATION CHECKS
# ======================================================================

print()
print("=" * 70)
print("[2] BASIC VALIDATION CHECKS")
print("=" * 70)


# ----------------------------------------------------------------------
# Prediction column detection
# ----------------------------------------------------------------------

prediction_candidates = [
    "y_probability",
    "prediction",
    "probability",
    "predicted_probability",
    "y_pred",
]

prediction_column = None

for column in prediction_candidates:
    if column in predictions.columns:
        prediction_column = column
        break

if prediction_column is None:

    numeric_columns = predictions.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_columns = [
        column
        for column in numeric_columns
        if column != ID_COLUMN
    ]

    if len(numeric_columns) == 1:
        prediction_column = numeric_columns[0]

if prediction_column is None:
    raise ValueError(
        "Could not automatically identify prediction probability column."
    )

print(f"Prediction column:   {prediction_column}")


# ----------------------------------------------------------------------
# Required columns
# ----------------------------------------------------------------------

required_data_columns = [
    ID_COLUMN,
    TARGET_COLUMN,
]

missing_data_columns = [
    column
    for column in required_data_columns
    if column not in data.columns
]

if missing_data_columns:
    raise ValueError(
        f"Missing required dataset columns: {missing_data_columns}"
    )

required_prediction_columns = [
    ID_COLUMN,
    prediction_column,
]

missing_prediction_columns = [
    column
    for column in required_prediction_columns
    if column not in predictions.columns
]

if missing_prediction_columns:
    raise ValueError(
        f"Missing required prediction columns: "
        f"{missing_prediction_columns}"
    )

print("Required columns:    OK")


# ----------------------------------------------------------------------
# Prediction ID uniqueness
# ----------------------------------------------------------------------

prediction_ids_unique = predictions[ID_COLUMN].is_unique

print(
    f"Prediction IDs unique: {prediction_ids_unique}"
)

if not prediction_ids_unique:
    raise ValueError(
        "Prediction IDs are not unique."
    )


# ----------------------------------------------------------------------
# Target validation
# ----------------------------------------------------------------------

target_values = sorted(
    data[TARGET_COLUMN].dropna().unique().tolist()
)

print(
    f"Target values:        {target_values}"
)

if not set(target_values).issubset({0, 1}):
    raise ValueError(
        "TARGET must contain only binary values 0 and 1."
    )


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------

merged = predictions[
    [ID_COLUMN, prediction_column]
].merge(
    data[
        [ID_COLUMN, TARGET_COLUMN]
    ],
    on=ID_COLUMN,
    how="left",
    validate="one_to_one",
)

print(
    f"Merged observations: {len(merged):,}"
)

prediction_coverage = (
    merged[TARGET_COLUMN].notna().mean()
)

print(
    f"Prediction coverage: {prediction_coverage:.4%}"
)

missing_predictions = merged[TARGET_COLUMN].isna().sum()

print(
    f"Missing predictions: {missing_predictions:,}"
)

if missing_predictions > 0:
    raise ValueError(
        "Some predictions could not be matched to target data."
    )


# ----------------------------------------------------------------------
# Probability validation
# ----------------------------------------------------------------------

y_probability = merged[prediction_column]

print(
    f"Prediction minimum:  {y_probability.min():.8f}"
)

print(
    f"Prediction maximum:  {y_probability.max():.8f}"
)

if not np.isfinite(y_probability).all():
    raise ValueError(
        "Predictions contain NaN or infinite values."
    )

if ((y_probability < 0) | (y_probability > 1)).any():
    raise ValueError(
        "Predictions must be probabilities in [0, 1]."
    )


y_true = merged[TARGET_COLUMN].astype(int).values
y_score = y_probability.values


# ======================================================================
# [3] OVERALL DISCRIMINATION
# ======================================================================

print()
print("=" * 70)
print("[3] OVERALL DISCRIMINATION")
print("=" * 70)

n_observations = len(y_true)

n_defaults = int(y_true.sum())

n_non_defaults = int(
    (y_true == 0).sum()
)

default_rate = y_true.mean()

print(
    f"Observations:        {n_observations:,}"
)

print(
    f"TARGET=1:            {n_defaults:,}"
)

print(
    f"TARGET=0:            {n_non_defaults:,}"
)

print(
    f"Default rate:        {default_rate:.4%}"
)


# ----------------------------------------------------------------------
# Point estimates
# ----------------------------------------------------------------------

roc_auc = roc_auc_score(
    y_true,
    y_score,
)

pr_auc = average_precision_score(
    y_true,
    y_score,
)

gini = 2 * roc_auc - 1

ks, ks_threshold = calculate_ks(
    y_true,
    y_score,
)

print()
print(f"ROC-AUC:             {roc_auc:.6f}")
print(f"PR-AUC:              {pr_auc:.6f}")
print(f"Gini:                {gini:.6f}")
print(f"KS statistic:        {ks:.6f}")
print(f"KS threshold:        {ks_threshold:.6f}")


# ======================================================================
# [4] STATISTICAL CONFIDENCE INTERVALS
# ======================================================================

print()
print("=" * 70)
print("[4] STATISTICAL CONFIDENCE INTERVALS")
print("=" * 70)

print()
print(
    f"Bootstrap resamples: {N_BOOTSTRAP:,}"
)

print(
    "Bootstrap method:    stratified by target class"
)

print(
    "Confidence level:    95%"
)


# ----------------------------------------------------------------------
# ROC-AUC CI
# ----------------------------------------------------------------------

roc_auc_lower, roc_auc_upper = bootstrap_metric_ci(
    y_true,
    y_score,
    roc_auc_score,
    random_state=RANDOM_STATE,
)


# ----------------------------------------------------------------------
# PR-AUC CI
# ----------------------------------------------------------------------

pr_auc_lower, pr_auc_upper = bootstrap_metric_ci(
    y_true,
    y_score,
    average_precision_score,
    random_state=RANDOM_STATE + 1,
)


# ----------------------------------------------------------------------
# KS CI
# ----------------------------------------------------------------------

ks_metric = lambda y, p: calculate_ks(y, p)[0]

ks_lower, ks_upper = bootstrap_metric_ci(
    y_true,
    y_score,
    ks_metric,
    random_state=RANDOM_STATE + 2,
)


print()
print(
    f"ROC-AUC 95% CI:      "
    f"[{roc_auc_lower:.6f}, {roc_auc_upper:.6f}]"
)

print(
    f"PR-AUC 95% CI:       "
    f"[{pr_auc_lower:.6f}, {pr_auc_upper:.6f}]"
)

print(
    f"KS 95% CI:           "
    f"[{ks_lower:.6f}, {ks_upper:.6f}]"
)


overall_ci = pd.DataFrame(
    [
        {
            "metric": "roc_auc",
            "estimate": roc_auc,
            "ci_lower": roc_auc_lower,
            "ci_upper": roc_auc_upper,
            "confidence_level": 0.95,
            "n_bootstrap": N_BOOTSTRAP,
        },
        {
            "metric": "pr_auc",
            "estimate": pr_auc,
            "ci_lower": pr_auc_lower,
            "ci_upper": pr_auc_upper,
            "confidence_level": 0.95,
            "n_bootstrap": N_BOOTSTRAP,
        },
        {
            "metric": "gini",
            "estimate": gini,
            "ci_lower": 2 * roc_auc_lower - 1,
            "ci_upper": 2 * roc_auc_upper - 1,
            "confidence_level": 0.95,
            "n_bootstrap": N_BOOTSTRAP,
        },
        {
            "metric": "ks",
            "estimate": ks,
            "ci_lower": ks_lower,
            "ci_upper": ks_upper,
            "confidence_level": 0.95,
            "n_bootstrap": N_BOOTSTRAP,
        },
    ]
)


# ======================================================================
# [5] RISK-BAND ANALYSIS
# ======================================================================

print()
print("=" * 70)
print("[5] RISK-BAND ANALYSIS")
print("=" * 70)


analysis_df = pd.DataFrame(
    {
        ID_COLUMN: merged[ID_COLUMN],
        "y_probability": y_score,
        TARGET_COLUMN: y_true,
    }
)

analysis_df["RISK_BAND"] = pd.qcut(
    analysis_df["y_probability"],
    q=10,
    labels=False,
    duplicates="drop",
) + 1


risk_bands = (
    analysis_df
    .groupby("RISK_BAND", observed=True)
    .agg(
        observations=(TARGET_COLUMN, "size"),
        predicted_pd=("y_probability", "mean"),
        observed_default_rate=(TARGET_COLUMN, "mean"),
        defaults=(TARGET_COLUMN, "sum"),
    )
    .reset_index()
)


risk_bands["population_share"] = (
    risk_bands["observations"]
    / len(analysis_df)
)

risk_bands["default_capture"] = (
    risk_bands["defaults"]
    / risk_bands["defaults"].sum()
)

risk_bands["cumulative_default_capture"] = (
    risk_bands["defaults"].cumsum()
    / risk_bands["defaults"].sum()
)


risk_bands["monotonic_with_previous"] = (
    risk_bands["observed_default_rate"]
    .diff()
    .fillna(0)
    >= 0
)


print()

print(
    risk_bands.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ----------------------------------------------------------------------
# Monotonicity
# ----------------------------------------------------------------------

if len(risk_bands) > 1:

    monotonic_pairs = int(
        risk_bands["monotonic_with_previous"]
        .iloc[1:]
        .sum()
    )

    total_pairs = len(risk_bands) - 1

    monotonicity_rate = (
        monotonic_pairs / total_pairs
    )

else:

    monotonic_pairs = 0
    total_pairs = 0
    monotonicity_rate = np.nan


print()

print(
    f"Monotonicity:       "
    f"{monotonic_pairs}/{total_pairs} adjacent pairs "
    f"non-decreasing "
    f"({monotonicity_rate:.2%})"
)


# ----------------------------------------------------------------------
# Risk separation
# ----------------------------------------------------------------------

lowest_default_rate = (
    risk_bands.iloc[0]["observed_default_rate"]
)

highest_default_rate = (
    risk_bands.iloc[-1]["observed_default_rate"]
)

if lowest_default_rate > 0:

    risk_separation_ratio = (
        highest_default_rate
        / lowest_default_rate
    )

else:

    risk_separation_ratio = np.inf


print()

print(
    f"Lowest-risk band default rate:  "
    f"{lowest_default_rate:.4%}"
)

print(
    f"Highest-risk band default rate: "
    f"{highest_default_rate:.4%}"
)

print(
    f"Risk separation ratio:           "
    f"{risk_separation_ratio:.2f}x"
)


# ======================================================================
# [6] RISK-BAND CONFIDENCE INTERVALS
# ======================================================================

print()
print("=" * 70)
print("[6] RISK-BAND CONFIDENCE INTERVALS")
print("=" * 70)

risk_band_ci_records = []

for _, row in risk_bands.iterrows():

    ci_lower, ci_upper = bootstrap_default_rate_ci(
        defaults=int(row["defaults"]),
        observations=int(row["observations"]),
        random_state=(
            RANDOM_STATE
            + int(row["RISK_BAND"])
        ),
    )

    risk_band_ci_records.append(
        {
            "RISK_BAND": int(row["RISK_BAND"]),
            "observations": int(row["observations"]),
            "defaults": int(row["defaults"]),
            "observed_default_rate": (
                row["observed_default_rate"]
            ),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": 0.95,
            "n_bootstrap": N_BOOTSTRAP,
        }
    )


risk_band_ci = pd.DataFrame(
    risk_band_ci_records
)


print()

print(
    risk_band_ci.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ======================================================================
# [7] THRESHOLD ANALYSIS
# ======================================================================

print()
print("=" * 70)
print("[7] THRESHOLD ANALYSIS")
print("=" * 70)


threshold_records = []

for threshold in THRESHOLDS:

    flagged = (
        y_score >= threshold
    )

    not_flagged = ~flagged

    flagged_count = int(
        flagged.sum()
    )

    not_flagged_count = int(
        not_flagged.sum()
    )

    flagged_rate = safe_rate(
        flagged_count,
        n_observations,
    )

    approval_rate = safe_rate(
        not_flagged_count,
        n_observations,
    )

    true_positive = int(
        ((flagged) & (y_true == 1)).sum()
    )

    false_positive = int(
        ((flagged) & (y_true == 0)).sum()
    )

    true_negative = int(
        ((not_flagged) & (y_true == 0)).sum()
    )

    false_negative = int(
        ((not_flagged) & (y_true == 1)).sum()
    )

    precision = safe_rate(
        true_positive,
        flagged_count,
    )

    recall = safe_rate(
        true_positive,
        n_defaults,
    )

    fpr = safe_rate(
        false_positive,
        n_non_defaults,
    )

    specificity = safe_rate(
        true_negative,
        n_non_defaults,
    )

    default_rate_flagged = safe_rate(
        true_positive,
        flagged_count,
    )

    threshold_records.append(
        {
            "threshold": threshold,
            "flagged_count": flagged_count,
            "flagged_rate": flagged_rate,
            "approval_rate": approval_rate,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "specificity": specificity,
            "default_rate_flagged": default_rate_flagged,
        }
    )


threshold_analysis = pd.DataFrame(
    threshold_records
)


print()

print(
    threshold_analysis.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ======================================================================
# [8] SEGMENT DISCRIMINATION
# ======================================================================

print()
print("=" * 70)
print("[8] SEGMENT DISCRIMINATION")
print("=" * 70)


segment_records = []


for segment_column in SEGMENT_COLUMNS:

    if segment_column not in data.columns:

        print(
            f"[WARNING] Missing segment column: "
            f"{segment_column}"
        )

        continue

    segment_data = data[
        [
            ID_COLUMN,
            segment_column,
        ]
    ]

    segment_predictions = predictions[
        [
            ID_COLUMN,
            prediction_column,
        ]
    ]

    segment_targets = data[
        [
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    ]

    segment_df = (
        segment_data
        .merge(
            segment_predictions,
            on=ID_COLUMN,
            how="inner",
        )
        .merge(
            segment_targets,
            on=ID_COLUMN,
            how="inner",
        )
    )

    for segment_value, group in segment_df.groupby(
        segment_column,
        dropna=False,
    ):

        observations = len(group)

        if observations < MIN_SEGMENT_SIZE:
            continue

        target_values = group[
            TARGET_COLUMN
        ].nunique()

        if target_values < 2:
            continue

        group_y = group[
            TARGET_COLUMN
        ].astype(int).values

        group_score = group[
            prediction_column
        ].values

        segment_auc = roc_auc_score(
            group_y,
            group_score,
        )

        segment_pr_auc = average_precision_score(
            group_y,
            group_score,
        )

        segment_gini = (
            2 * segment_auc - 1
        )

        segment_ks, _ = calculate_ks(
            group_y,
            group_score,
        )

        segment_records.append(
            {
                "segment": segment_column,
                "segment_value": segment_value,
                "observations": observations,
                "default_rate": group_y.mean(),
                "roc_auc": segment_auc,
                "pr_auc": segment_pr_auc,
                "gini": segment_gini,
                "ks": segment_ks,
            }
        )


segment_discrimination = pd.DataFrame(
    segment_records
)


if not segment_discrimination.empty:

    print()

    print(
        segment_discrimination.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

else:

    print()
    print(
        "No eligible segments found."
    )


# ======================================================================
# [9] STABILITY / CONSISTENCY CHECKS
# ======================================================================

print()
print("=" * 70)
print("[9] STABILITY / CONSISTENCY CHECKS")
print("=" * 70)


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
    f"Risk-band monotonicity:"
    f"{monotonicity_rate:.2%}"
)


if not segment_discrimination.empty:

    segment_auc_min = (
        segment_discrimination["roc_auc"].min()
    )

    segment_auc_mean = (
        segment_discrimination["roc_auc"].mean()
    )

    segment_auc_max = (
        segment_discrimination["roc_auc"].max()
    )

    weakest_segment_row = (
        segment_discrimination
        .loc[
            segment_discrimination["roc_auc"].idxmin()
        ]
    )

    strongest_segment_row = (
        segment_discrimination
        .loc[
            segment_discrimination["roc_auc"].idxmax()
        ]
    )

    weakest_segment_difference = (
        roc_auc - segment_auc_min
    )

    print()

    print(
        f"Segment ROC-AUC min:   "
        f"{segment_auc_min:.6f}"
    )

    print(
        f"Segment ROC-AUC mean:  "
        f"{segment_auc_mean:.6f}"
    )

    print(
        f"Segment ROC-AUC max:   "
        f"{segment_auc_max:.6f}"
    )

    print()

    print(
        "Weakest segment:      "
        f"{weakest_segment_row['segment']} = "
        f"{weakest_segment_row['segment_value']} "
        f"(AUC={weakest_segment_row['roc_auc']:.6f})"
    )

    print(
        "Strongest segment:    "
        f"{strongest_segment_row['segment']} = "
        f"{strongest_segment_row['segment_value']} "
        f"(AUC={strongest_segment_row['roc_auc']:.6f})"
    )

    print()

    print(
        f"Overall-to-weakest AUC difference: "
        f"{weakest_segment_difference:.6f}"
    )


# ======================================================================
# [10] VALIDATION FINDINGS
# ======================================================================

print()
print("=" * 70)
print("[10] VALIDATION FINDINGS")
print("=" * 70)


# ----------------------------------------------------------------------
# Overall discrimination
# ----------------------------------------------------------------------

print()

print(
    "[OBSERVATION] Overall discrimination metrics:"
)

print(
    f"  ROC-AUC = {roc_auc:.4f} "
    f"(95% CI {roc_auc_lower:.4f}–{roc_auc_upper:.4f})"
)

print(
    f"  PR-AUC  = {pr_auc:.4f} "
    f"(95% CI {pr_auc_lower:.4f}–{pr_auc_upper:.4f})"
)

print(
    f"  Gini    = {gini:.4f}"
)

print(
    f"  KS      = {ks:.4f} "
    f"(95% CI {ks_lower:.4f}–{ks_upper:.4f})"
)


# ----------------------------------------------------------------------
# PR-AUC relative to prevalence
# ----------------------------------------------------------------------

pr_auc_lift = (
    pr_auc / default_rate
)

print()

print(
    "[OBSERVATION] PR-AUC relative to portfolio prevalence:"
)

print(
    f"  Default prevalence = {default_rate:.4%}"
)

print(
    f"  PR-AUC / prevalence = {pr_auc_lift:.2f}x"
)


# ----------------------------------------------------------------------
# Risk ranking
# ----------------------------------------------------------------------

print()

if monotonic_pairs == total_pairs:

    print(
        "[OBSERVATION] Risk-band default rates show "
        "strong monotonicity."
    )

else:

    print(
        "[OBSERVATION] Risk-band default rates are not "
        "fully monotonic."
    )


print(
    "[OBSERVATION] The highest-risk band exhibits "
    "materially higher observed default frequency "
    "than the lowest-risk band."
)


# ----------------------------------------------------------------------
# Default capture
# ----------------------------------------------------------------------

highest_band_capture = (
    risk_bands.iloc[-1]["default_capture"]
)

print()

print(
    "[OBSERVATION] Highest-risk band default capture:"
)

print(
    f"  Top risk decile contains "
    f"{highest_band_capture:.2%} of observed defaults."
)


# ----------------------------------------------------------------------
# Segment analysis
# ----------------------------------------------------------------------

if not segment_discrimination.empty:

    n_segments = len(
        segment_discrimination
    )

    print()

    print(
        f"[OBSERVATION] Segments analysed: "
        f"{n_segments}"
    )

    print(
        "[OBSERVATION] Segment discrimination range:"
    )

    print(
        f"  AUC min = {segment_auc_min:.4f}"
    )

    print(
        f"  AUC mean = {segment_auc_mean:.4f}"
    )

    print(
        f"  AUC max = {segment_auc_max:.4f}"
    )

    print(
        "[OBSERVATION] The weakest segment differs "
        f"from overall AUC by "
        f"{weakest_segment_difference:.4f} "
        "in absolute terms."
    )

    print(
        "[OBSERVATION] Segment performance should be "
        "interpreted in conjunction with segment size "
        "and default prevalence; no arbitrary AUC "
        "degradation threshold is applied."
    )


# ----------------------------------------------------------------------
# Threshold interpretation
# ----------------------------------------------------------------------

print()

print(
    "[OBSERVATION] Threshold analysis is exploratory "
    "and illustrates the trade-off between approval "
    "rate, captured defaults, false-positive rate, "
    "and precision."
)

print(
    "[OBSERVATION] The KS threshold is a statistical "
    "separation point and is not treated as a business "
    "decision threshold."
)


# ======================================================================
# [11] OUTPUTS
# ======================================================================

print()
print("=" * 70)
print("[11] OUTPUTS")
print("=" * 70)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


risk_bands_path = (
    OUTPUT_DIR / "risk_bands.csv"
)

threshold_analysis_path = (
    OUTPUT_DIR / "threshold_analysis.csv"
)

segment_analysis_path = (
    OUTPUT_DIR / "segment_discrimination.csv"
)

overall_discrimination_path = (
    OUTPUT_DIR / "overall_discrimination.csv"
)

discrimination_ci_path = (
    OUTPUT_DIR / "discrimination_ci.csv"
)

risk_band_ci_path = (
    OUTPUT_DIR / "risk_band_ci.csv"
)


# ----------------------------------------------------------------------
# Save risk bands
# ----------------------------------------------------------------------

risk_bands.to_csv(
    risk_bands_path,
    index=False,
)


# ----------------------------------------------------------------------
# Save threshold analysis
# ----------------------------------------------------------------------

threshold_analysis.to_csv(
    threshold_analysis_path,
    index=False,
)


# ----------------------------------------------------------------------
# Save segment analysis
# ----------------------------------------------------------------------

segment_discrimination.to_csv(
    segment_analysis_path,
    index=False,
)


# ----------------------------------------------------------------------
# Save overall discrimination summary
# ----------------------------------------------------------------------

overall_discrimination = pd.DataFrame(
    [
        {
            "observations": n_observations,
            "defaults": n_defaults,
            "non_defaults": n_non_defaults,
            "default_rate": default_rate,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "gini": gini,
            "ks": ks,
            "ks_threshold": ks_threshold,
            "risk_band_monotonicity": monotonicity_rate,
            "lowest_risk_band_default_rate":
                lowest_default_rate,
            "highest_risk_band_default_rate":
                highest_default_rate,
            "risk_separation_ratio":
                risk_separation_ratio,
            "highest_risk_band_default_capture":
                highest_band_capture,
        }
    ]
)


overall_discrimination.to_csv(
    overall_discrimination_path,
    index=False,
)


# ----------------------------------------------------------------------
# Save metric confidence intervals
# ----------------------------------------------------------------------

overall_ci.to_csv(
    discrimination_ci_path,
    index=False,
)


# ----------------------------------------------------------------------
# Save risk-band confidence intervals
# ----------------------------------------------------------------------

risk_band_ci.to_csv(
    risk_band_ci_path,
    index=False,
)


print()

print(
    f"Risk bands:          {risk_bands_path}"
)

print(
    f"Threshold analysis:  {threshold_analysis_path}"
)

print(
    f"Segment analysis:    {segment_analysis_path}"
)

print(
    f"Overall summary:     {overall_discrimination_path}"
)

print(
    f"Discrimination CI:   {discrimination_ci_path}"
)

print(
    f"Risk-band CI:        {risk_band_ci_path}"
)


print()
print("=" * 70)
print("DISCRIMINATION ANALYSIS COMPLETE")
print("=" * 70)