from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ======================================================================
# NORTHSTAR MODEL VALIDATION — TEMPORAL CHECK V0.2
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"

BASELINE_DIR = PROJECT_ROOT / "reports" / "baseline"
PREDICTIONS_PATH = BASELINE_DIR / "validation_predictions.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "temporal"

TEMPORAL_FEATURES_PATH = OUTPUT_DIR / "temporal_features.csv"
TEMPORAL_RESULTS_PATH = OUTPUT_DIR / "temporal_analysis.csv"
COMPARISON_RESULTS_PATH = OUTPUT_DIR / "oot_comparison.csv"

N_BINS = 5

# DAYS_BIRTH is used only as a diagnostic segmentation axis.
# It is NOT an application / observation date.
TEMPORAL_AXIS = "DAYS_BIRTH"

TEMPORAL_FEATURES = [
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
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
    return 2 * roc_auc - 1


def calculate_ks(y_true, y_score):
    data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_score": y_score,
        }
    ).sort_values("y_score", ascending=False)

    total_bad = data["y_true"].sum()
    total_good = len(data) - total_bad

    if total_bad == 0 or total_good == 0:
        return np.nan

    data["cum_bad"] = data["y_true"].cumsum() / total_bad
    data["cum_good"] = (
        (1 - data["y_true"]).cumsum() / total_good
    )

    return (data["cum_bad"] - data["cum_good"]).abs().max()


def calculate_psi(reference_scores, comparison_scores, n_bins=10):
    """
    Calculate PSI for two score distributions.

    Bins are defined using reference quantiles.
    Small epsilon prevents log(0).
    """

    reference_scores = pd.Series(reference_scores).dropna()
    comparison_scores = pd.Series(comparison_scores).dropna()

    if len(reference_scores) == 0 or len(comparison_scores) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, n_bins + 1)

    breakpoints = reference_scores.quantile(quantiles).values
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return np.nan

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_bins = pd.cut(
        reference_scores,
        bins=breakpoints,
        include_lowest=True,
    )

    comparison_bins = pd.cut(
        comparison_scores,
        bins=breakpoints,
        include_lowest=True,
    )

    reference_distribution = (
        reference_bins.value_counts(normalize=True, sort=False)
    )

    comparison_distribution = (
        comparison_bins.value_counts(normalize=True, sort=False)
    )

    epsilon = 1e-6

    reference_distribution = reference_distribution.clip(
        lower=epsilon
    )

    comparison_distribution = comparison_distribution.clip(
        lower=epsilon
    )

    psi = (
        (comparison_distribution - reference_distribution)
        * np.log(
            comparison_distribution
            / reference_distribution
        )
    ).sum()

    return psi


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("NORTHSTAR MODEL VALIDATION — TEMPORAL CHECK V0.2")
print("=" * 70)


# ======================================================================
# [1] LOADING DATA
# ======================================================================

print_section("[1] LOADING DATA")

df = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)

print(f"Dataset shape:       {df.shape}")
print(f"Predictions shape:   {predictions.shape}")


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

print("Required columns: OK")

prediction_ids_unique = predictions[ID_COLUMN].is_unique

print(
    f"Prediction IDs unique: "
    f"{prediction_ids_unique}"
)

if not prediction_ids_unique:
    raise ValueError(
        "Prediction IDs are not unique."
    )

missing_y_true = predictions["y_true"].isna().sum()
missing_y_probability = predictions["y_probability"].isna().sum()

print(f"Missing y_true: {missing_y_true}")
print(
    f"Missing y_probability: "
    f"{missing_y_probability}"
)

if missing_y_true > 0 or missing_y_probability > 0:
    raise ValueError(
        "Missing values detected in predictions."
    )

probability_outside_range = (
    (predictions["y_probability"] < 0)
    | (predictions["y_probability"] > 1)
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

merge_columns = [
    ID_COLUMN,
    TARGET,
] + TEMPORAL_FEATURES

original_subset = df[merge_columns].copy()

merged = predictions.merge(
    original_subset,
    on=ID_COLUMN,
    how="left",
    validate="one_to_one",
)

print(f"Merged shape:       {merged.shape}")

if merged[TARGET].isna().any():
    raise ValueError(
        "Some prediction IDs could not be matched "
        "to the original dataset."
    )

target_mismatches = (
    merged[TARGET] != merged["y_true"]
).sum()

print(f"Target mismatches:   {target_mismatches}")

if target_mismatches > 0:
    raise ValueError(
        "Target mismatch between predictions "
        "and original dataset."
    )

print("Prediction/data merge: OK")


# ======================================================================
# [4] TEMPORAL FEATURES
# ======================================================================

print_section("[4] TEMPORAL FEATURES")

temporal_feature_rows = []

for feature in TEMPORAL_FEATURES:

    series = merged[feature]

    row = {
        "feature": feature,
        "dtype": str(series.dtype),
        "missing_pct": series.isna().mean() * 100,
        "min": series.min(),
        "median": series.median(),
        "max": series.max(),
        "unique": series.nunique(),
    }

    temporal_feature_rows.append(row)

    print()
    print(feature)
    print(f"  dtype:       {row['dtype']}")
    print(
        f"  missing:     "
        f"{row['missing_pct']:.2f}%"
    )
    print(f"  min:         {row['min']}")
    print(f"  median:      {row['median']}")
    print(f"  max:         {row['max']}")
    print(f"  unique:      {row['unique']:,}")


temporal_features_df = pd.DataFrame(
    temporal_feature_rows
)


# ======================================================================
# [5] TEMPORAL FEATURE / TARGET RELATIONSHIP
# ======================================================================

print_section(
    "[5] TEMPORAL FEATURE / TARGET RELATIONSHIP"
)

feature_target_rows = []

overall_target_rate = merged[TARGET].mean()

for feature in TEMPORAL_FEATURES:

    data = merged[
        [feature, TARGET]
    ].dropna()

    feature_auc = roc_auc_score(
        data[TARGET],
        data[feature],
    )

    row = {
        "feature": feature,
        "target_rate": overall_target_rate,
        "feature_auc": feature_auc,
    }

    feature_target_rows.append(row)

    print()
    print(feature)
    print(
        f"  Target rate: "
        f"{overall_target_rate:.4%}"
    )
    print(
        f"  Feature AUC: "
        f"{feature_auc:.4f}"
    )


feature_target_df = pd.DataFrame(
    feature_target_rows
)

temporal_features_df = temporal_features_df.merge(
    feature_target_df,
    on="feature",
    how="left",
)


# ======================================================================
# [6] TEMPORAL SEGMENTATION
# ======================================================================

print_section("[6] TEMPORAL SEGMENTATION")

print(f"Temporal axis: {TEMPORAL_AXIS}")
print(f"Number of bins: {N_BINS}")

axis_data = merged[
    [TEMPORAL_AXIS]
].dropna()

merged["temporal_bin"] = pd.qcut(
    merged[TEMPORAL_AXIS],
    q=N_BINS,
    duplicates="drop",
)

actual_n_bins = merged["temporal_bin"].nunique()

if actual_n_bins < 2:
    raise ValueError(
        "Temporal segmentation produced fewer "
        "than two groups."
    )


# ======================================================================
# [7] TEMPORAL GROUP METRICS
# ======================================================================

print_section("[7] TEMPORAL GROUP METRICS")

group_rows = []

for bin_number, (bin_value, group) in enumerate(
    merged.groupby(
        "temporal_bin",
        observed=True,
    ),
    start=1,
):

    y_true = group["y_true"]
    y_probability = group["y_probability"]

    target_rate = y_true.mean()
    mean_probability = y_probability.mean()

    if y_true.nunique() == 2:
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
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        ks = np.nan

    gini = (
        calculate_gini(roc_auc)
        if not np.isnan(roc_auc)
        else np.nan
    )

    brier = brier_score_loss(
        y_true,
        y_probability,
    )

    calibration_gap = (
        mean_probability - target_rate
    )

    row = {
        "bin_number": bin_number,
        "temporal_bin": str(bin_value),
        "n": len(group),
        "target_rate": target_rate,
        "mean_probability": mean_probability,
        "calibration_gap": calibration_gap,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "gini": gini,
        "ks": ks,
        "brier_score": brier,
    }

    group_rows.append(row)

    print()
    print(
        f"Bin {bin_number}: {bin_value}"
    )
    print(
        f"  N:                {len(group):,}"
    )
    print(
        f"  Target rate:      "
        f"{target_rate:.4%}"
    )
    print(
        f"  Mean probability: "
        f"{mean_probability:.4%}"
    )
    print(
        f"  Calibration gap:  "
        f"{calibration_gap:+.4%}"
    )
    print(
        f"  ROC-AUC:          "
        f"{roc_auc:.4f}"
    )
    print(
        f"  PR-AUC:           "
        f"{pr_auc:.4f}"
    )
    print(
        f"  Gini:             "
        f"{gini:.4f}"
    )
    print(
        f"  KS:               "
        f"{ks:.4f}"
    )
    print(
        f"  Brier score:      "
        f"{brier:.6f}"
    )


temporal_results = pd.DataFrame(group_rows)


# ======================================================================
# [8] REFERENCE vs LATEST GROUP
# ======================================================================

print_section(
    "[8] REFERENCE vs LATEST GROUP"
)

reference_group = merged[
    merged["temporal_bin"]
    == merged["temporal_bin"].cat.categories[0]
]

latest_group = merged[
    merged["temporal_bin"]
    == merged["temporal_bin"].cat.categories[-1]
]

reference_row = temporal_results.iloc[0]
latest_row = temporal_results.iloc[-1]

reference_target_rate = reference_row["target_rate"]
latest_target_rate = latest_row["target_rate"]

reference_probability = (
    reference_row["mean_probability"]
)

latest_probability = (
    latest_row["mean_probability"]
)

reference_calibration_gap = (
    reference_row["calibration_gap"]
)

latest_calibration_gap = (
    latest_row["calibration_gap"]
)

reference_auc = reference_row["roc_auc"]
latest_auc = latest_row["roc_auc"]

reference_pr_auc = reference_row["pr_auc"]
latest_pr_auc = latest_row["pr_auc"]

reference_brier = reference_row["brier_score"]
latest_brier = latest_row["brier_score"]

print()
print("Reference group:")
print(
    f"  N:                "
    f"{len(reference_group):,}"
)
print(
    f"  Target rate:      "
    f"{reference_target_rate:.4%}"
)
print(
    f"  Mean probability: "
    f"{reference_probability:.4%}"
)
print(
    f"  Calibration gap:  "
    f"{reference_calibration_gap:+.4%}"
)
print(
    f"  ROC-AUC:          "
    f"{reference_auc:.4f}"
)
print(
    f"  PR-AUC:           "
    f"{reference_pr_auc:.4f}"
)
print(
    f"  Brier:            "
    f"{reference_brier:.6f}"
)

print()
print("Latest group:")
print(
    f"  N:                "
    f"{len(latest_group):,}"
)
print(
    f"  Target rate:      "
    f"{latest_target_rate:.4%}"
)
print(
    f"  Mean probability: "
    f"{latest_probability:.4%}"
)
print(
    f"  Calibration gap:  "
    f"{latest_calibration_gap:+.4%}"
)
print(
    f"  ROC-AUC:          "
    f"{latest_auc:.4f}"
)
print(
    f"  PR-AUC:           "
    f"{latest_pr_auc:.4f}"
)
print(
    f"  Brier:            "
    f"{latest_brier:.6f}"
)


# ======================================================================
# [9] CHANGES
# ======================================================================

print_section("[9] CHANGES")

target_rate_change = (
    latest_target_rate
    - reference_target_rate
)

mean_probability_change = (
    latest_probability
    - reference_probability
)

calibration_gap_change = (
    latest_calibration_gap
    - reference_calibration_gap
)

roc_auc_change = latest_auc - reference_auc
pr_auc_change = latest_pr_auc - reference_pr_auc
brier_change = latest_brier - reference_brier

segment_psi = calculate_psi(
    reference_group["y_probability"],
    latest_group["y_probability"],
)

auc_min = temporal_results["roc_auc"].min()
auc_max = temporal_results["roc_auc"].max()
auc_range = auc_max - auc_min

print(
    f"Target rate change:       "
    f"{target_rate_change:+.4%}"
    f" ({target_rate_change * 100:+.2f} pp)"
)

print(
    f"Mean probability change:  "
    f"{mean_probability_change:+.4%}"
)

print(
    f"Calibration gap change:   "
    f"{calibration_gap_change:+.4%}"
)

print(
    f"ROC-AUC change:            "
    f"{roc_auc_change:+.4f}"
)

print(
    f"PR-AUC change:             "
    f"{pr_auc_change:+.4f}"
)

print(
    f"Brier change:              "
    f"{brier_change:+.6f}"
)

print(
    f"Segment score PSI:         "
    f"{segment_psi:.6f}"
)

print()
print(
    f"ROC-AUC range across bins: "
    f"{auc_min:.4f} - {auc_max:.4f}"
)

print(
    f"ROC-AUC range width:       "
    f"{auc_range:.4f}"
)


# ======================================================================
# [10] SAVING RESULTS
# ======================================================================

print_section("[10] SAVING RESULTS")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

temporal_features_df.to_csv(
    TEMPORAL_FEATURES_PATH,
    index=False,
)

temporal_results.to_csv(
    TEMPORAL_RESULTS_PATH,
    index=False,
)

comparison = pd.DataFrame(
    [
        {
            "temporal_axis": TEMPORAL_AXIS,
            "reference_bin": str(
                temporal_results.iloc[0]["temporal_bin"]
            ),
            "latest_bin": str(
                temporal_results.iloc[-1]["temporal_bin"]
            ),
            "reference_n": len(reference_group),
            "latest_n": len(latest_group),
            "reference_target_rate": reference_target_rate,
            "latest_target_rate": latest_target_rate,
            "target_rate_change": target_rate_change,
            "reference_mean_probability": reference_probability,
            "latest_mean_probability": latest_probability,
            "mean_probability_change": mean_probability_change,
            "reference_calibration_gap": reference_calibration_gap,
            "latest_calibration_gap": latest_calibration_gap,
            "calibration_gap_change": calibration_gap_change,
            "reference_roc_auc": reference_auc,
            "latest_roc_auc": latest_auc,
            "roc_auc_change": roc_auc_change,
            "reference_pr_auc": reference_pr_auc,
            "latest_pr_auc": latest_pr_auc,
            "pr_auc_change": pr_auc_change,
            "reference_brier": reference_brier,
            "latest_brier": latest_brier,
            "brier_change": brier_change,
            "segment_score_psi": segment_psi,
            "auc_min": auc_min,
            "auc_max": auc_max,
            "auc_range": auc_range,
        }
    ]
)

comparison.to_csv(
    COMPARISON_RESULTS_PATH,
    index=False,
)

print()
print("Temporal feature analysis saved to:")
print(f"  {TEMPORAL_FEATURES_PATH}")

print()
print("Temporal analysis saved to:")
print(f"  {TEMPORAL_RESULTS_PATH}")

print()
print("Reference/latest comparison saved to:")
print(f"  {COMPARISON_RESULTS_PATH}")


# ======================================================================
# [11] VALIDATION INTERPRETATION
# ======================================================================

print_section("[11] VALIDATION INTERPRETATION")

print()
print(
    "This is a diagnostic temporal stability analysis."
)

print()
print("IMPORTANT:")
print(
    "DAYS_BIRTH is not an application date and therefore"
)
print(
    "cannot be treated as a formal production OOT axis."
)

print()
print(
    "The current analysis evaluates whether model behaviour"
)
print(
    "changes systematically across segments of a"
)
print(
    "temporal-related feature."
)

print()
print(
    "The observed differences should be interpreted as:"
)

print()
print(
    "    population heterogeneity"
)
print(
    "    + model performance heterogeneity"
)
print(
    "    + calibration behaviour across segments"
)

print()
print(
    "and NOT as evidence of production-time model drift."
)

print()
print(
    "Key observations from the current diagnostic:"
)

print()
print(
    f"    Target rate range: "
    f"{temporal_results['target_rate'].min():.4%}"
    f" - "
    f"{temporal_results['target_rate'].max():.4%}"
)

print(
    f"    ROC-AUC range: "
    f"{auc_min:.4f}"
    f" - "
    f"{auc_max:.4f}"
)

print(
    f"    Segment score PSI: "
    f"{segment_psi:.6f}"
)

print()
print(
    "A high segment score PSI should be interpreted as"
)
print(
    "evidence of score-distribution heterogeneity between"
)
print(
    "the reference and latest segments, not as formal"
)
print(
    "production population drift."
)

print()
print(
    "A formal OOT validation requires a genuine observation /"
)
print(
    "application date or another defensible time-ordering"
)
print(
    "variable."
)

print()
print("=" * 70)
print("TEMPORAL CHECK COMPLETE")
print("=" * 70)