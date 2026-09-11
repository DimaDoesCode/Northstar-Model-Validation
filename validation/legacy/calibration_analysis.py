"""
Northstar Model Validation — Calibration Analysis V0.1

Purpose
-------
Independent validation of probability calibration for a binary
classification model.

The analysis evaluates whether predicted probabilities correspond
to observed default rates.

Outputs
-------
reports/calibration/
    calibration_metrics.csv
    calibration_by_decile.csv
    calibration_curve.png

Usage
-----
python src/validation/calibration_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss, log_loss


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "baseline" / "validation_predictions.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "calibration"

METRICS_PATH = REPORT_DIR / "calibration_metrics.csv"
DECILE_PATH = REPORT_DIR / "calibration_by_decile.csv"
PLOT_PATH = REPORT_DIR / "calibration_curve.png"

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
PREDICTION_COLUMN = "y_prediction"

N_BINS = 10


# ======================================================================
# HELPERS
# ======================================================================

def print_section(title):
    """Print a formatted section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_required_columns(df, required_columns, df_name):
    """Validate that required columns exist."""
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}"
        )

    print(
        f"Required {df_name.lower()} columns: OK"
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("NORTHSTAR MODEL VALIDATION — CALIBRATION ANALYSIS V0.1")
    print("=" * 70)

    # ------------------------------------------------------------------
    # [1] LOADING DATA
    # ------------------------------------------------------------------

    print_section("[1] LOADING DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    dataset = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    print(f"Dataset shape:       {dataset.shape}")
    print(f"Predictions shape:   {predictions.shape}")

    # ------------------------------------------------------------------
    # [2] BASIC VALIDATION CHECKS
    # ------------------------------------------------------------------

    print_section("[2] BASIC VALIDATION CHECKS")

    validate_required_columns(
        dataset,
        [ID_COLUMN, TARGET_COLUMN],
        "dataset"
    )

    validate_required_columns(
        predictions,
        [ID_COLUMN, PREDICTION_COLUMN],
        "prediction"
    )

    prediction_ids_unique = predictions[ID_COLUMN].is_unique

    print(
        f"Prediction IDs unique: {prediction_ids_unique}"
    )

    if not prediction_ids_unique:
        raise ValueError(
            "Prediction IDs are not unique."
        )

    dataset_ids_unique = dataset[ID_COLUMN].is_unique

    print(
        f"Dataset IDs unique:    {dataset_ids_unique}"
    )

    if not dataset_ids_unique:
        raise ValueError(
            "Dataset IDs are not unique."
        )

    # ------------------------------------------------------------------
    # [3] MERGING TARGET AND PREDICTIONS
    # ------------------------------------------------------------------

    print_section("[3] MERGING TARGET AND PREDICTIONS")

    validation = predictions.merge(
        dataset[[ID_COLUMN, TARGET_COLUMN]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one"
    )

    print(f"Merged validation shape: {validation.shape}")

    missing_target = validation[TARGET_COLUMN].isna().sum()

    print(
        f"Missing target values:   {missing_target}"
    )

    if missing_target > 0:
        raise ValueError(
            "Some prediction IDs could not be matched to TARGET."
        )

    # ------------------------------------------------------------------
    # [4] PREDICTION VALIDATION
    # ------------------------------------------------------------------

    print_section("[4] PREDICTION VALIDATION")

    missing_predictions = (
        validation[PREDICTION_COLUMN].isna().sum()
    )

    print(
        f"Missing predictions:      {missing_predictions}"
    )

    if missing_predictions > 0:
        raise ValueError(
            "Missing predicted probabilities detected."
        )

    prediction_min = validation[PREDICTION_COLUMN].min()
    prediction_max = validation[PREDICTION_COLUMN].max()

    print(
        f"Prediction minimum:       {prediction_min:.6f}"
    )
    print(
        f"Prediction maximum:       {prediction_max:.6f}"
    )

    if (
        prediction_min < 0
        or prediction_max > 1
    ):
        raise ValueError(
            "Predicted probabilities must be in [0, 1]."
        )

    target_values = set(
        validation[TARGET_COLUMN].unique()
    )

    print(
        f"Target values:             {sorted(target_values)}"
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "TARGET must contain only 0 and 1."
        )

    # ------------------------------------------------------------------
    # [5] BASIC DATASET STATISTICS
    # ------------------------------------------------------------------

    print_section("[5] BASIC DATASET STATISTICS")

    y_true = validation[TARGET_COLUMN].astype(int)
    y_prob = validation[PREDICTION_COLUMN].astype(float)

    n_observations = len(validation)

    observed_bad_rate = y_true.mean()
    mean_predicted_probability = y_prob.mean()

    print(
        f"Observations:              {n_observations:,}"
    )
    print(
        f"Observed bad rate:         {observed_bad_rate:.6f}"
    )
    print(
        f"Mean predicted probability:{mean_predicted_probability:.6f}"
    )

    # ------------------------------------------------------------------
    # [6] OVERALL CALIBRATION METRICS
    # ------------------------------------------------------------------

    print_section("[6] OVERALL CALIBRATION METRICS")

    # Calibration-in-the-large:
    #
    # Difference between average predicted probability
    # and observed event rate.
    #
    # Positive value:
    #     model predicts too much risk on average.
    #
    # Negative value:
    #     model predicts too little risk on average.

    calibration_in_the_large = (
        mean_predicted_probability
        - observed_bad_rate
    )

    brier_score = brier_score_loss(
        y_true,
        y_prob
    )

    logloss = log_loss(
        y_true,
        y_prob
    )

    print(
        f"Calibration-in-the-large: {calibration_in_the_large:+.6f}"
    )
    print(
        f"Brier score:              {brier_score:.6f}"
    )
    print(
        f"Log loss:                  {logloss:.6f}"
    )

    # ------------------------------------------------------------------
    # [7] CALIBRATION BY DECILE
    # ------------------------------------------------------------------

    print_section("[7] CALIBRATION BY DECILE")

    calibration_data = validation[
        [ID_COLUMN, TARGET_COLUMN, PREDICTION_COLUMN]
    ].copy()

    # Rank first to make qcut robust to duplicated probabilities.
    calibration_data["_rank"] = (
        calibration_data[PREDICTION_COLUMN]
        .rank(method="first")
    )

    calibration_data["risk_decile"] = pd.qcut(
        calibration_data["_rank"],
        q=N_BINS,
        labels=False
    ) + 1

    decile_rows = []

    for decile in range(1, N_BINS + 1):

        group = calibration_data[
            calibration_data["risk_decile"] == decile
        ]

        n = len(group)

        mean_predicted = (
            group[PREDICTION_COLUMN].mean()
        )

        observed_rate = (
            group[TARGET_COLUMN].mean()
        )

        calibration_error = (
            observed_rate - mean_predicted
        )

        absolute_error = abs(calibration_error)

        decile_rows.append(
            {
                "risk_decile": decile,
                "n_observations": n,
                "mean_predicted_probability": mean_predicted,
                "observed_bad_rate": observed_rate,
                "calibration_error": calibration_error,
                "absolute_calibration_error": absolute_error,
                "min_predicted_probability":
                    group[PREDICTION_COLUMN].min(),
                "max_predicted_probability":
                    group[PREDICTION_COLUMN].max(),
                "bad_count":
                    int(group[TARGET_COLUMN].sum()),
            }
        )

    calibration_by_decile = pd.DataFrame(
        decile_rows
    )

    print(
        calibration_by_decile.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # ------------------------------------------------------------------
    # [8] AGGREGATE CALIBRATION ERRORS
    # ------------------------------------------------------------------

    print_section("[8] AGGREGATE CALIBRATION ERRORS")

    mae_calibration_error = (
        calibration_by_decile[
            "absolute_calibration_error"
        ].mean()
    )

    max_calibration_error = (
        calibration_by_decile[
            "absolute_calibration_error"
        ].max()
    )

    weighted_calibration_error = (
        np.average(
            calibration_by_decile[
                "absolute_calibration_error"
            ],
            weights=calibration_by_decile[
                "n_observations"
            ]
        )
    )

    print(
        f"Mean absolute calibration error: "
        f"{mae_calibration_error:.6f}"
    )

    print(
        f"Maximum absolute calibration error: "
        f"{max_calibration_error:.6f}"
    )

    print(
        f"Weighted calibration error: "
        f"{weighted_calibration_error:.6f}"
    )

    # ------------------------------------------------------------------
    # [9] CALIBRATION METRICS TABLE
    # ------------------------------------------------------------------

    print_section("[9] SAVING CALIBRATION METRICS")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics = pd.DataFrame(
        [
            {
                "metric": "n_observations",
                "value": n_observations,
            },
            {
                "metric": "observed_bad_rate",
                "value": observed_bad_rate,
            },
            {
                "metric": "mean_predicted_probability",
                "value": mean_predicted_probability,
            },
            {
                "metric": "calibration_in_the_large",
                "value": calibration_in_the_large,
            },
            {
                "metric": "brier_score",
                "value": brier_score,
            },
            {
                "metric": "log_loss",
                "value": logloss,
            },
            {
                "metric": "mean_absolute_calibration_error",
                "value": mae_calibration_error,
            },
            {
                "metric": "maximum_absolute_calibration_error",
                "value": max_calibration_error,
            },
            {
                "metric": "weighted_calibration_error",
                "value": weighted_calibration_error,
            },
        ]
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False
    )

    calibration_by_decile.to_csv(
        DECILE_PATH,
        index=False
    )

    print(
        f"Metrics saved:            {METRICS_PATH}"
    )

    print(
        f"Decile table saved:       {DECILE_PATH}"
    )

    # ------------------------------------------------------------------
    # [10] CALIBRATION CURVE
    # ------------------------------------------------------------------

    print_section("[10] CALIBRATION CURVE")

    x = calibration_by_decile[
        "mean_predicted_probability"
    ]

    y = calibration_by_decile[
        "observed_bad_rate"
    ]

    plt.figure(
        figsize=(8, 8)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        x,
        y,
        marker="o",
        label="Model calibration"
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed bad rate"
    )

    plt.title(
        "Calibration Curve — Validation Sample"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150
    )

    plt.close()

    print(
        f"Calibration curve saved: {PLOT_PATH}"
    )

    # ------------------------------------------------------------------
    # [11] VALIDATION SUMMARY
    # ------------------------------------------------------------------

    print_section("[11] VALIDATION SUMMARY")

    print(
        f"Overall observed bad rate: "
        f"{observed_bad_rate:.4f}"
    )

    print(
        f"Mean predicted probability: "
        f"{mean_predicted_probability:.4f}"
    )

    print(
        f"Calibration-in-the-large: "
        f"{calibration_in_the_large:+.4f}"
    )

    print(
        f"Mean absolute calibration error: "
        f"{mae_calibration_error:.4f}"
    )

    print(
        f"Maximum absolute calibration error: "
        f"{max_calibration_error:.4f}"
    )

    print(
        f"Brier score: "
        f"{brier_score:.4f}"
    )

    print(
        f"Log loss: "
        f"{logloss:.4f}"
    )

    print()
    print("=" * 70)
    print("CALIBRATION ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()