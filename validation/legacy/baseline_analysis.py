"""
NORTHSTAR MODEL VALIDATION
Baseline Analysis V0.1

Purpose:
    Analyze validation predictions produced by the baseline model.

The analysis focuses on:
    - discrimination
    - prediction distribution
    - calibration
    - risk ranking
    - concentration of bad cases

This script does NOT retrain the model.
It uses validation_predictions.csv produced by baseline_model.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline"
    / "validation_predictions.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline"
    / "metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "baseline"
)

N_BINS = 10


# ============================================================
# HELPERS
# ============================================================

def calculate_gini(roc_auc: float) -> float:
    """Convert ROC-AUC to Gini coefficient."""
    return 2 * roc_auc - 1


def calculate_ks(y_true, y_score) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.

    KS = max absolute difference between cumulative
    bad and good distributions.
    """

    data = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "score": np.asarray(y_score),
        }
    ).sort_values(
        "score",
        ascending=False,
    )

    total_good = (data["target"] == 0).sum()
    total_bad = (data["target"] == 1).sum()

    data["cum_bad"] = (
        (data["target"] == 1).cumsum()
        / total_bad
    )

    data["cum_good"] = (
        (data["target"] == 0).cumsum()
        / total_good
    )

    ks = (
        data["cum_bad"] - data["cum_good"]
    ).abs().max()

    return float(ks)


def print_section(title: str):
    """Print section header."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_discrimination(
    y_true,
    y_probability,
):
    """Calculate primary discrimination metrics."""

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability,
    )

    gini = calculate_gini(roc_auc)

    ks = calculate_ks(
        y_true,
        y_probability,
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "gini": gini,
        "ks": ks,
    }


def analyze_prediction_distribution(
    y_probability,
):
    """Return descriptive statistics for predicted probabilities."""

    return pd.Series(
        y_probability
    ).describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )


def create_calibration_table(
    y_true,
    y_probability,
    n_bins=10,
):
    """
    Create equal-frequency probability bins.

    For each bin:
        - number of observations
        - mean predicted probability
        - actual bad rate
        - calibration error
    """

    data = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "probability": np.asarray(y_probability),
        }
    )

    data["bin"] = pd.qcut(
        data["probability"],
        q=n_bins,
        duplicates="drop",
    )

    calibration = (
        data.groupby(
            "bin",
            observed=True,
        )
        .agg(
            count=("target", "size"),
            mean_predicted=("probability", "mean"),
            actual_bad_rate=("target", "mean"),
        )
        .reset_index()
    )

    calibration["calibration_error"] = (
        calibration["actual_bad_rate"]
        - calibration["mean_predicted"]
    )

    calibration["absolute_calibration_error"] = (
        calibration["calibration_error"].abs()
    )

    calibration["bin"] = (
        calibration["bin"].astype(str)
    )

    return calibration


def create_decile_table(
    y_true,
    y_probability,
    n_bins=10,
):
    """
    Create risk-ranking deciles.

    Decile 1 = lowest predicted risk.
    Decile 10 = highest predicted risk.
    """

    data = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "probability": np.asarray(y_probability),
        }
    )

    data = data.sort_values(
        "probability",
        ascending=True,
    ).reset_index(drop=True)

    data["decile"] = (
        pd.qcut(
            data.index,
            q=n_bins,
            labels=False,
        )
        + 1
    )

    deciles = (
        data.groupby("decile")
        .agg(
            count=("target", "size"),
            mean_probability=("probability", "mean"),
            bad_count=("target", "sum"),
            bad_rate=("target", "mean"),
        )
        .reset_index()
    )

    total_bad = data["target"].sum()

    deciles["bad_share"] = (
        deciles["bad_count"]
        / total_bad
    )

    deciles["cumulative_bad_share"] = (
        deciles["bad_count"].cumsum()
        / total_bad
    )

    return deciles


def create_concentration_table(
    y_true,
    y_probability,
):
    """
    Measure how many bad cases are captured
    in the highest-risk population segments.
    """

    data = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "probability": np.asarray(y_probability),
        }
    )

    data = data.sort_values(
        "probability",
        ascending=False,
    ).reset_index(drop=True)

    total_observations = len(data)
    total_bad = data["target"].sum()

    percentages = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
    ]

    rows = []

    for percentage in percentages:

        n = max(
            1,
            int(
                np.ceil(
                    total_observations
                    * percentage
                )
            ),
        )

        segment = data.iloc[:n]

        bad_count = segment["target"].sum()

        rows.append(
            {
                "top_population_pct": percentage * 100,
                "population_count": n,
                "bad_count": int(bad_count),
                "bad_capture_pct": (
                    bad_count
                    / total_bad
                    * 100
                ),
                "segment_bad_rate": (
                    segment["target"].mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "NORTHSTAR MODEL VALIDATION — "
        "BASELINE ANALYSIS V0.1"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD PREDICTIONS
    # --------------------------------------------------------

    print_section("[1] LOADING VALIDATION PREDICTIONS")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Validation predictions not found:\n"
            f"{INPUT_PATH}\n\n"
            f"Run baseline_model.py first."
        )

    predictions = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "y_true",
        "y_probability",
        "y_prediction",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    print(
        f"Predictions shape: "
        f"{predictions.shape}"
    )

    print(
        f"Target prevalence: "
        f"{predictions['y_true'].mean():.4%}"
    )

    # --------------------------------------------------------
    # 2. BASIC DATA CHECKS
    # --------------------------------------------------------

    print_section("[2] BASIC VALIDATION CHECKS")

    y_true = predictions["y_true"]
    y_probability = predictions["y_probability"]

    print(
        f"Missing y_true: "
        f"{y_true.isna().sum()}"
    )

    print(
        f"Missing probabilities: "
        f"{y_probability.isna().sum()}"
    )

    print(
        f"Probability range: "
        f"{y_probability.min():.6f} "
        f"— "
        f"{y_probability.max():.6f}"
    )

    invalid_probability = (
        (y_probability < 0)
        | (y_probability > 1)
    ).sum()

    print(
        f"Invalid probabilities: "
        f"{invalid_probability}"
    )

    if (
        y_true.isna().any()
        or y_probability.isna().any()
    ):
        raise ValueError(
            "Missing values detected in "
            "validation predictions."
        )

    if invalid_probability > 0:
        raise ValueError(
            "Probability values outside "
            "[0, 1] detected."
        )

    # --------------------------------------------------------
    # 3. DISCRIMINATION
    # --------------------------------------------------------

    print_section("[3] DISCRIMINATION")

    discrimination = analyze_discrimination(
        y_true,
        y_probability,
    )

    print(
        f"ROC-AUC : "
        f"{discrimination['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC  : "
        f"{discrimination['pr_auc']:.4f}"
    )

    print(
        f"Gini    : "
        f"{discrimination['gini']:.4f}"
    )

    print(
        f"KS      : "
        f"{discrimination['ks']:.4f}"
    )

    # --------------------------------------------------------
    # 4. PROBABILITY DISTRIBUTION
    # --------------------------------------------------------

    print_section("[4] PREDICTION DISTRIBUTION")

    distribution = analyze_prediction_distribution(
        y_probability
    )

    print(
        distribution.to_string()
    )

    # --------------------------------------------------------
    # 5. CALIBRATION
    # --------------------------------------------------------

    print_section("[5] CALIBRATION")

    calibration = create_calibration_table(
        y_true,
        y_probability,
        n_bins=N_BINS,
    )

    print(
        calibration.to_string(
            index=False,
            formatters={
                "mean_predicted": "{:.6f}".format,
                "actual_bad_rate": "{:.6f}".format,
                "calibration_error": "{:.6f}".format,
                "absolute_calibration_error":
                    "{:.6f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # 6. GLOBAL CALIBRATION METRICS
    # --------------------------------------------------------

    print_section("[6] GLOBAL CALIBRATION METRICS")

    brier = brier_score_loss(
        y_true,
        y_probability,
    )

    logloss = log_loss(
        y_true,
        y_probability,
    )

    mean_absolute_calibration_error = (
        calibration[
            "absolute_calibration_error"
        ].mean()
    )

    print(
        f"Brier score: "
        f"{brier:.6f}"
    )

    print(
        f"Log loss: "
        f"{logloss:.6f}"
    )

    print(
        f"Mean absolute calibration error: "
        f"{mean_absolute_calibration_error:.6f}"
    )

    # --------------------------------------------------------
    # 7. RISK RANKING
    # --------------------------------------------------------

    print_section("[7] RISK RANKING — DECILES")

    deciles = create_decile_table(
        y_true,
        y_probability,
        n_bins=N_BINS,
    )

    print(
        deciles.to_string(
            index=False,
            formatters={
                "mean_probability":
                    "{:.6f}".format,
                "bad_rate":
                    "{:.6f}".format,
                "bad_share":
                    "{:.6f}".format,
                "cumulative_bad_share":
                    "{:.6f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # 8. HIGH-RISK CONCENTRATION
    # --------------------------------------------------------

    print_section(
        "[8] HIGH-RISK CONCENTRATION"
    )

    concentration = create_concentration_table(
        y_true,
        y_probability,
    )

    print(
        concentration.to_string(
            index=False,
            formatters={
                "top_population_pct":
                    "{:.0f}".format,
                "bad_capture_pct":
                    "{:.2f}%".format,
                "segment_bad_rate":
                    "{:.2f}%".format,
            },
        )
    )

    # --------------------------------------------------------
    # 9. VALIDATION FLAGS
    # --------------------------------------------------------

    print_section("[9] INITIAL VALIDATION FLAGS")

    flags = []

    # Discrimination
    if discrimination["roc_auc"] >= 0.70:
        flags.append(
            "PASS: ROC-AUC >= 0.70"
        )
    else:
        flags.append(
            "REVIEW: ROC-AUC < 0.70"
        )

    # Risk ordering
    first_bad_rate = deciles.iloc[0]["bad_rate"]
    last_bad_rate = deciles.iloc[-1]["bad_rate"]

    if last_bad_rate > first_bad_rate:
        flags.append(
            "PASS: Bad rate increases "
            "from low-risk to high-risk deciles"
        )
    else:
        flags.append(
            "REVIEW: Risk ordering is not monotonic"
        )

    # Calibration
    if mean_absolute_calibration_error < 0.01:
        flags.append(
            "PASS: Mean absolute calibration "
            "error < 1%"
        )
    elif mean_absolute_calibration_error < 0.02:
        flags.append(
            "REVIEW: Mean absolute calibration "
            "error between 1% and 2%"
        )
    else:
        flags.append(
            "REVIEW: Mean absolute calibration "
            "error >= 2%"
        )

    # Probability range
    if (
        y_probability.min() >= 0
        and y_probability.max() <= 1
    ):
        flags.append(
            "PASS: All probabilities are "
            "within [0, 1]"
        )

    for flag in flags:
        print(f"  {flag}")

    # --------------------------------------------------------
    # 10. SAVE RESULTS
    # --------------------------------------------------------

    print_section("[10] SAVING ANALYSIS RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    discrimination_df = pd.DataFrame(
        [discrimination]
    )

    discrimination_df.to_csv(
        OUTPUT_DIR
        / "analysis_discrimination.csv",
        index=False,
    )

    distribution.to_csv(
        OUTPUT_DIR
        / "analysis_prediction_distribution.csv"
    )

    calibration.to_csv(
        OUTPUT_DIR
        / "analysis_calibration.csv",
        index=False,
    )

    deciles.to_csv(
        OUTPUT_DIR
        / "analysis_deciles.csv",
        index=False,
    )

    concentration.to_csv(
        OUTPUT_DIR
        / "analysis_concentration.csv",
        index=False,
    )

    calibration_metrics = pd.DataFrame(
        [
            {
                "brier_score": brier,
                "log_loss": logloss,
                "mean_absolute_calibration_error":
                    mean_absolute_calibration_error,
            }
        ]
    )

    calibration_metrics.to_csv(
        OUTPUT_DIR
        / "analysis_calibration_metrics.csv",
        index=False,
    )

    print(
        "\nAnalysis reports saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # 11. SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"ROC-AUC : "
        f"{discrimination['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC  : "
        f"{discrimination['pr_auc']:.4f}"
    )

    print(
        f"Gini    : "
        f"{discrimination['gini']:.4f}"
    )

    print(
        f"KS      : "
        f"{discrimination['ks']:.4f}"
    )

    print(
        f"Brier   : "
        f"{brier:.6f}"
    )

    print(
        f"Log Loss: "
        f"{logloss:.6f}"
    )

    print(
        "\nBaseline analysis completed successfully."
    )


if __name__ == "__main__":
    main()