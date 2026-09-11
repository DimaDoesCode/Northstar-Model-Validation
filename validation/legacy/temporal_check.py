"""
NORTHSTAR MODEL VALIDATION
Temporal Check V0.1

Purpose:
    Perform an initial temporal stability analysis of the
    baseline model validation sample.

Important:
    The Home Credit application_train.csv dataset does not
    contain a genuine application / observation date.

    Therefore this script does NOT perform a formal production
    OOT validation.

    Instead, it:
        1. identifies available temporal-related features;
        2. examines their distributions;
        3. evaluates target prevalence across temporal bins;
        4. evaluates baseline model performance across bins;
        5. compares the earliest and latest groups;
        6. calculates prediction-distribution PSI.

    DAYS_BIRTH is used only as a diagnostic temporal axis.
    It must NOT be interpreted as application time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TARGET = "TARGET"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    / "temporal"
)

TEMPORAL_RESULTS_PATH = (
    OUTPUT_DIR
    / "temporal_analysis.csv"
)

OOT_RESULTS_PATH = (
    OUTPUT_DIR
    / "oot_comparison.csv"
)


# Number of quantile groups
N_BINS = 5


# Candidate temporal-related features
TEMPORAL_FEATURES = [
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
]


# Initial diagnostic axis.
#
# IMPORTANT:
# This is not a true application-time variable.
TEMPORAL_AXIS = "DAYS_BIRTH"


# ============================================================
# HELPERS
# ============================================================

def calculate_psi(
    reference,
    comparison,
    bins=10,
):
    """
    Calculate Population Stability Index.

    Reference:
        Baseline/reference score distribution.

    Comparison:
        Distribution being compared against reference.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    comparison = np.asarray(
        comparison,
        dtype=float,
    )

    reference = reference[
        np.isfinite(reference)
    ]

    comparison = comparison[
        np.isfinite(comparison)
    ]

    if len(reference) == 0 or len(comparison) == 0:
        return np.nan

    # Use reference quantiles as bin boundaries.
    edges = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1),
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        return np.nan

    reference_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    comparison_counts, _ = np.histogram(
        comparison,
        bins=edges,
    )

    reference_pct = (
        reference_counts
        / len(reference)
    )

    comparison_pct = (
        comparison_counts
        / len(comparison)
    )

    # Avoid log(0) and division by zero.
    eps = 1e-6

    reference_pct = np.clip(
        reference_pct,
        eps,
        None,
    )

    comparison_pct = np.clip(
        comparison_pct,
        eps,
        None,
    )

    psi = np.sum(
        (
            comparison_pct
            - reference_pct
        )
        * np.log(
            comparison_pct
            / reference_pct
        )
    )

    return float(psi)


def safe_roc_auc(
    y_true,
    y_probability,
):
    """Calculate ROC-AUC safely."""

    if y_true.nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            y_probability,
        )
    )


def safe_pr_auc(
    y_true,
    y_probability,
):
    """Calculate PR-AUC safely."""

    if y_true.nunique() < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            y_probability,
        )
    )


def calculate_group_metrics(group):
    """
    Calculate model validation metrics
    for one temporal group.
    """

    y_true = group["y_true"]
    y_probability = group["y_probability"]

    return {
        "n": len(group),
        "target_rate": y_true.mean(),
        "mean_probability": y_probability.mean(),
        "roc_auc": safe_roc_auc(
            y_true,
            y_probability,
        ),
        "pr_auc": safe_pr_auc(
            y_true,
            y_probability,
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_probability,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NORTHSTAR MODEL VALIDATION — "
        "TEMPORAL CHECK V0.1"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1] LOADING DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Validation predictions not found:\n"
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

    # --------------------------------------------------------
    # 2. BASIC VALIDATION CHECKS
    # --------------------------------------------------------

    print("\n[2] BASIC VALIDATION CHECKS")

    required_data_columns = [
        "SK_ID_CURR",
        TARGET,
    ]

    required_prediction_columns = [
        "SK_ID_CURR",
        "y_true",
        "y_probability",
        "y_prediction",
    ]

    for column in required_data_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required dataset column missing: "
                f"{column}"
            )

    for column in required_prediction_columns:

        if column not in predictions.columns:
            raise ValueError(
                f"Required prediction column missing: "
                f"{column}"
            )

    if predictions["SK_ID_CURR"].duplicated().any():

        raise ValueError(
            "Duplicate SK_ID_CURR found in "
            "validation predictions."
        )

    if len(predictions) == 0:

        raise ValueError(
            "Validation predictions are empty."
        )

    print("Required columns: OK")

    print(
        f"Prediction IDs unique: "
        f"{predictions['SK_ID_CURR'].is_unique}"
    )

    print(
        f"Missing y_true: "
        f"{predictions['y_true'].isna().sum()}"
    )

    print(
        f"Missing y_probability: "
        f"{predictions['y_probability'].isna().sum()}"
    )

    # --------------------------------------------------------
    # 3. MERGE PREDICTIONS WITH ORIGINAL DATA
    # --------------------------------------------------------

    print(
        "\n[3] MERGING PREDICTIONS "
        "WITH ORIGINAL DATA"
    )

    analysis = df.merge(
        predictions,
        on="SK_ID_CURR",
        how="inner",
        validate="one_to_one",
        suffixes=("_data", "_prediction"),
    )

    print(
        f"Merged shape:       {analysis.shape}"
    )

    if len(analysis) != len(predictions):

        raise ValueError(
            "Not all validation predictions "
            "could be matched to the source dataset."
        )

    # Verify target consistency.
    target_mismatch = (
        analysis[TARGET]
        != analysis["y_true"]
    ).sum()

    print(
        f"Target mismatches:   "
        f"{target_mismatch}"
    )

    if target_mismatch > 0:

        raise ValueError(
            "TARGET and y_true do not match."
        )

    print("Prediction/data merge: OK")

    # --------------------------------------------------------
    # 4. IDENTIFY TEMPORAL FEATURES
    # --------------------------------------------------------

    print("\n[4] TEMPORAL FEATURES")

    available_features = [
        feature
        for feature in TEMPORAL_FEATURES
        if feature in analysis.columns
    ]

    if not available_features:

        raise ValueError(
            "No temporal-related features "
            "found in dataset."
        )

    for feature in available_features:

        series = analysis[feature]

        print(f"\n{feature}")

        print(
            f"  dtype:       {series.dtype}"
        )

        print(
            f"  missing:     "
            f"{series.isna().mean():.2%}"
        )

        print(
            f"  min:         "
            f"{series.min()}"
        )

        print(
            f"  median:      "
            f"{series.median()}"
        )

        print(
            f"  max:         "
            f"{series.max()}"
        )

        print(
            f"  unique:      "
            f"{series.nunique():,}"
        )

    # --------------------------------------------------------
    # 5. TEMPORAL FEATURE / TARGET RELATIONSHIP
    # --------------------------------------------------------

    print(
        "\n[5] TEMPORAL FEATURE / TARGET RELATIONSHIP"
    )

    for feature in available_features:

        temp = analysis[
            [feature, TARGET]
        ].dropna()

        if temp.empty:
            continue

        feature_auc = safe_roc_auc(
            temp[TARGET],
            temp[feature],
        )

        print(
            f"\n{feature}"
        )

        print(
            f"  Target rate: "
            f"{temp[TARGET].mean():.4%}"
        )

        print(
            f"  Feature AUC: "
            f"{feature_auc:.4f}"
        )

    # --------------------------------------------------------
    # 6. TEMPORAL SEGMENTATION
    # --------------------------------------------------------

    print(
        "\n[6] TEMPORAL SEGMENTATION"
    )

    if TEMPORAL_AXIS not in analysis.columns:

        raise ValueError(
            f"Temporal axis not found: "
            f"{TEMPORAL_AXIS}"
        )

    temporal = analysis[
        [
            "SK_ID_CURR",
            "y_true",
            "y_probability",
            "y_prediction",
            TEMPORAL_AXIS,
        ]
    ].copy()

    temporal = temporal.dropna(
        subset=[TEMPORAL_AXIS]
    )

    if temporal.empty:

        raise ValueError(
            "No observations available after "
            "dropping missing temporal values."
        )

    # Quantile segmentation.
    temporal["temporal_bin"] = pd.qcut(
        temporal[TEMPORAL_AXIS],
        q=N_BINS,
        duplicates="drop",
    )

    categories = (
        temporal["temporal_bin"]
        .cat.categories
    )

    print(
        f"Temporal axis: "
        f"{TEMPORAL_AXIS}"
    )

    print(
        f"Number of bins: "
        f"{len(categories)}"
    )

    # --------------------------------------------------------
    # 7. GROUP METRICS
    # --------------------------------------------------------

    print(
        "\n[7] TEMPORAL GROUP METRICS"
    )

    temporal_results = []

    for bin_number, (
        temporal_bin,
        group,
    ) in enumerate(
        temporal.groupby(
            "temporal_bin",
            observed=True,
        ),
        start=1,
    ):

        metrics = calculate_group_metrics(
            group
        )

        row = {
            "temporal_feature": TEMPORAL_AXIS,
            "bin_number": bin_number,
            "temporal_bin": str(
                temporal_bin
            ),
            "temporal_min": group[
                TEMPORAL_AXIS
            ].min(),
            "temporal_median": group[
                TEMPORAL_AXIS
            ].median(),
            "temporal_max": group[
                TEMPORAL_AXIS
            ].max(),
            **metrics,
        }

        temporal_results.append(row)

        print(
            f"\nBin {bin_number}: "
            f"{temporal_bin}"
        )

        print(
            f"  N:                "
            f"{metrics['n']:,}"
        )

        print(
            f"  Target rate:      "
            f"{metrics['target_rate']:.4%}"
        )

        print(
            f"  Mean probability: "
            f"{metrics['mean_probability']:.4%}"
        )

        print(
            f"  ROC-AUC:          "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC:           "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"  Brier score:      "
            f"{metrics['brier_score']:.6f}"
        )

    temporal_results_df = pd.DataFrame(
        temporal_results
    )

    # --------------------------------------------------------
    # 8. REFERENCE vs LATEST GROUP
    # --------------------------------------------------------

    print(
        "\n[8] REFERENCE vs LATEST GROUP"
    )

    first_bin = categories[0]
    last_bin = categories[-1]

    reference = temporal[
        temporal["temporal_bin"]
        == first_bin
    ]

    latest = temporal[
        temporal["temporal_bin"]
        == last_bin
    ]

    reference_metrics = (
        calculate_group_metrics(
            reference
        )
    )

    latest_metrics = (
        calculate_group_metrics(
            latest
        )
    )

    print("\nReference group:")

    print(
        f"  N:           "
        f"{reference_metrics['n']:,}"
    )

    print(
        f"  Target rate: "
        f"{reference_metrics['target_rate']:.4%}"
    )

    print(
        f"  ROC-AUC:     "
        f"{reference_metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC:      "
        f"{reference_metrics['pr_auc']:.4f}"
    )

    print(
        f"  Brier:       "
        f"{reference_metrics['brier_score']:.6f}"
    )

    print("\nLatest group:")

    print(
        f"  N:           "
        f"{latest_metrics['n']:,}"
    )

    print(
        f"  Target rate: "
        f"{latest_metrics['target_rate']:.4%}"
    )

    print(
        f"  ROC-AUC:     "
        f"{latest_metrics['roc_auc']:.4f}"
    )

    print(
        f"  PR-AUC:      "
        f"{latest_metrics['pr_auc']:.4f}"
    )

    print(
        f"  Brier:       "
        f"{latest_metrics['brier_score']:.6f}"
    )

    # --------------------------------------------------------
    # 9. CHANGES
    # --------------------------------------------------------

    print(
        "\n[9] CHANGES"
    )

    target_rate_change = (
        latest_metrics["target_rate"]
        - reference_metrics["target_rate"]
    )

    roc_auc_change = (
        latest_metrics["roc_auc"]
        - reference_metrics["roc_auc"]
    )

    pr_auc_change = (
        latest_metrics["pr_auc"]
        - reference_metrics["pr_auc"]
    )

    brier_change = (
        latest_metrics["brier_score"]
        - reference_metrics["brier_score"]
    )

    score_psi = calculate_psi(
        reference["y_probability"],
        latest["y_probability"],
    )

    print(
        f"  Target rate change: "
        f"{target_rate_change:+.4%}"
    )

    print(
        f"  ROC-AUC change:     "
        f"{roc_auc_change:+.4f}"
    )

    print(
        f"  PR-AUC change:      "
        f"{pr_auc_change:+.4f}"
    )

    print(
        f"  Brier change:       "
        f"{brier_change:+.6f}"
    )

    print(
        f"  Score PSI:          "
        f"{score_psi:.6f}"
    )

    # --------------------------------------------------------
    # 10. SAVE TEMPORAL RESULTS
    # --------------------------------------------------------

    print(
        "\n[10] SAVING RESULTS"
    )

    temporal_results_df.to_csv(
        TEMPORAL_RESULTS_PATH,
        index=False,
    )

    oot_comparison = pd.DataFrame(
        [
            {
                "temporal_feature": TEMPORAL_AXIS,
                "reference_bin": str(
                    first_bin
                ),
                "latest_bin": str(
                    last_bin
                ),
                "reference_n": (
                    reference_metrics["n"]
                ),
                "latest_n": (
                    latest_metrics["n"]
                ),
                "reference_target_rate": (
                    reference_metrics[
                        "target_rate"
                    ]
                ),
                "latest_target_rate": (
                    latest_metrics[
                        "target_rate"
                    ]
                ),
                "target_rate_change": (
                    target_rate_change
                ),
                "reference_roc_auc": (
                    reference_metrics[
                        "roc_auc"
                    ]
                ),
                "latest_roc_auc": (
                    latest_metrics[
                        "roc_auc"
                    ]
                ),
                "roc_auc_change": (
                    roc_auc_change
                ),
                "reference_pr_auc": (
                    reference_metrics[
                        "pr_auc"
                    ]
                ),
                "latest_pr_auc": (
                    latest_metrics[
                        "pr_auc"
                    ]
                ),
                "pr_auc_change": (
                    pr_auc_change
                ),
                "reference_brier": (
                    reference_metrics[
                        "brier_score"
                    ]
                ),
                "latest_brier": (
                    latest_metrics[
                        "brier_score"
                    ]
                ),
                "brier_change": (
                    brier_change
                ),
                "score_psi": score_psi,
            }
        ]
    )

    oot_comparison.to_csv(
        OOT_RESULTS_PATH,
        index=False,
    )

    print(
        f"\nTemporal analysis saved to:"
        f"\n  {TEMPORAL_RESULTS_PATH}"
    )

    print(
        f"\nReference/latest comparison saved to:"
        f"\n  {OOT_RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # 11. VALIDATION INTERPRETATION
    # --------------------------------------------------------

    print(
        "\n[11] VALIDATION INTERPRETATION"
    )

    print(
        """
This is a diagnostic temporal stability analysis.

IMPORTANT:
DAYS_BIRTH is not an application date and therefore
cannot be treated as a formal production OOT axis.

The current analysis evaluates whether model behaviour
changes systematically across segments of a temporal-related
feature.

Observed changes should therefore be interpreted as:

    temporal feature stability
    + population heterogeneity
    + model performance heterogeneity

and NOT as evidence of production-time model drift.

A formal OOT validation requires a genuine observation /
application date or another defensible time-ordering variable.
"""
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TEMPORAL CHECK COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()