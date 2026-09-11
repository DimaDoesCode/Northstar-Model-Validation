from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# NORTHSTAR MODEL VALIDATION — CALIBRATION ANALYSIS V0.3
# Statistical Confidence Intervals
# ======================================================================


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline"
    / "validation_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "calibration"

METRICS_PATH = OUTPUT_DIR / "calibration_metrics_v03.csv"
DECILE_PATH = OUTPUT_DIR / "calibration_by_decile_v03.csv"
PLOT_PATH = OUTPUT_DIR / "calibration_curve_ci_v03.png"


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "y_true"
PROBABILITY_COLUMN = "y_probability"
PREDICTION_COLUMN = "y_prediction"

N_BINS = 10

# 95% confidence interval
Z_95 = 1.959963984540054


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------

def wilson_confidence_interval(
    successes: int,
    n: int,
    z: float = Z_95,
):
    """
    Calculate Wilson score confidence interval
    for a binomial proportion.
    """

    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1 + (z ** 2) / n

    centre = (
        p
        + (z ** 2) / (2 * n)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1 - p) / n
                + (z ** 2) / (4 * n ** 2)
            )
        )
        / denominator
    )

    lower = centre - margin
    upper = centre + margin

    return lower, upper


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------

print("=" * 70)
print("NORTHSTAR MODEL VALIDATION — CALIBRATION ANALYSIS V0.3")
print("Statistical Confidence Intervals")
print("=" * 70)


# ----------------------------------------------------------------------
# [1] LOADING BASELINE PREDICTIONS
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[1] LOADING BASELINE PREDICTIONS")
print("=" * 70)

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Predictions file not found:\n{PREDICTIONS_PATH}"
    )

df = pd.read_csv(PREDICTIONS_PATH)

print(f"Predictions shape:   {df.shape}")


# ----------------------------------------------------------------------
# [2] BASIC VALIDATION CHECKS
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[2] BASIC VALIDATION CHECKS")
print("=" * 70)

required_columns = [
    ID_COLUMN,
    TARGET_COLUMN,
    PROBABILITY_COLUMN,
    PREDICTION_COLUMN,
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required prediction columns: OK")

ids_unique = df[ID_COLUMN].is_unique

print(f"Prediction IDs unique: {ids_unique}")

if not ids_unique:
    raise ValueError("Prediction IDs are not unique.")


# ----------------------------------------------------------------------
# [3] TARGET VALIDATION
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[3] TARGET VALIDATION")
print("=" * 70)

missing_target = df[TARGET_COLUMN].isna().sum()

print(f"Missing target values:   {missing_target}")

if missing_target > 0:
    raise ValueError("Target contains missing values.")

target_values = sorted(
    df[TARGET_COLUMN].dropna().unique().tolist()
)

print(f"Target values:             {target_values}")

if not set(target_values).issubset({0, 1}):
    raise ValueError(
        "Target contains values other than 0 and 1."
    )


# ----------------------------------------------------------------------
# [4] PROBABILITY VALIDATION
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[4] PROBABILITY VALIDATION")
print("=" * 70)

missing_probabilities = df[PROBABILITY_COLUMN].isna().sum()

print(f"Missing probabilities:    {missing_probabilities}")

if missing_probabilities > 0:
    raise ValueError(
        "Predicted probabilities contain missing values."
    )

probability_min = df[PROBABILITY_COLUMN].min()
probability_max = df[PROBABILITY_COLUMN].max()

print(
    f"Probability minimum:      {probability_min:.10f}"
)

print(
    f"Probability maximum:      {probability_max:.10f}"
)

if probability_min < 0 or probability_max > 1:
    raise ValueError(
        "Predicted probabilities must be within [0, 1]."
    )

unique_probabilities = df[PROBABILITY_COLUMN].nunique()

print(
    f"Unique probability values:{unique_probabilities:,}"
)

if unique_probabilities <= 2:
    raise ValueError(
        "Probability column appears to contain hard predictions "
        "rather than continuous probabilities."
    )

print("Continuous probability check: OK")


# ----------------------------------------------------------------------
# [5] PREDICTION CONSISTENCY CHECK
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[5] PREDICTION CONSISTENCY CHECK")
print("=" * 70)

expected_predictions = (
    df[PROBABILITY_COLUMN] >= 0.50
).astype(int)

predictions_consistent = (
    df[PREDICTION_COLUMN].astype(int)
    == expected_predictions
).all()

print(
    "Stored predictions consistent with 0.50 threshold: "
    f"{predictions_consistent}"
)

if not predictions_consistent:
    raise ValueError(
        "Stored predictions are inconsistent with "
        "y_probability >= 0.50."
    )


# ----------------------------------------------------------------------
# [6] BASIC DATASET STATISTICS
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[6] BASIC DATASET STATISTICS")
print("=" * 70)

n_observations = len(df)

observed_bad_rate = (
    df[TARGET_COLUMN].mean()
)

mean_predicted_probability = (
    df[PROBABILITY_COLUMN].mean()
)

calibration_in_the_large = (
    mean_predicted_probability
    - observed_bad_rate
)

print(
    f"Observations:              {n_observations:,}"
)

print(
    f"Observed bad rate:         {observed_bad_rate:.6f}"
)

print(
    f"Mean predicted probability:{mean_predicted_probability:.6f}"
)


# ----------------------------------------------------------------------
# [7] OVERALL CALIBRATION METRICS
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[7] OVERALL CALIBRATION METRICS")
print("=" * 70)

brier_score = np.mean(
    (
        df[PROBABILITY_COLUMN]
        - df[TARGET_COLUMN]
    ) ** 2
)

# Avoid log(0)
eps = 1e-15

p = np.clip(
    df[PROBABILITY_COLUMN].to_numpy(),
    eps,
    1 - eps,
)

y = df[TARGET_COLUMN].to_numpy()

log_loss = -np.mean(
    y * np.log(p)
    + (1 - y) * np.log(1 - p)
)

print(
    f"Calibration-in-the-large: "
    f"{calibration_in_the_large:.6f}"
)

print(
    f"Brier score:              "
    f"{brier_score:.6f}"
)

print(
    f"Log loss:                  "
    f"{log_loss:.6f}"
)


# ----------------------------------------------------------------------
# [8] CALIBRATION BY DECILE
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[8] CALIBRATION BY DECILE")
print("=" * 70)


# Rank first guarantees that qcut can create
# approximately equal-sized bins even when
# probability values are tied.

df["_probability_rank"] = (
    df[PROBABILITY_COLUMN]
    .rank(method="first")
)

df["risk_decile"] = pd.qcut(
    df["_probability_rank"],
    q=N_BINS,
    labels=False,
) + 1


decile_rows = []


for decile in range(1, N_BINS + 1):

    segment = df[
        df["risk_decile"] == decile
    ]

    n = len(segment)

    bad_count = int(
        segment[TARGET_COLUMN].sum()
    )

    mean_probability = (
        segment[PROBABILITY_COLUMN].mean()
    )

    observed_rate = (
        segment[TARGET_COLUMN].mean()
    )

    calibration_error = (
        observed_rate
        - mean_probability
    )

    absolute_calibration_error = abs(
        calibration_error
    )

    min_probability = (
        segment[PROBABILITY_COLUMN].min()
    )

    max_probability = (
        segment[PROBABILITY_COLUMN].max()
    )

    ci_lower, ci_upper = (
        wilson_confidence_interval(
            successes=bad_count,
            n=n,
        )
    )

    prediction_inside_ci = (
        ci_lower
        <= mean_probability
        <= ci_upper
    )

    ci_width = ci_upper - ci_lower

    decile_rows.append(
        {
            "risk_decile": decile,
            "n_observations": n,
            "mean_predicted_probability": mean_probability,
            "observed_bad_rate": observed_rate,
            "calibration_error": calibration_error,
            "absolute_calibration_error": (
                absolute_calibration_error
            ),
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "ci_width_95": ci_width,
            "predicted_probability_inside_95_ci": (
                prediction_inside_ci
            ),
            "min_predicted_probability": min_probability,
            "max_predicted_probability": max_probability,
            "bad_count": bad_count,
        }
    )


decile_df = pd.DataFrame(decile_rows)


print(
    decile_df[
        [
            "risk_decile",
            "n_observations",
            "mean_predicted_probability",
            "observed_bad_rate",
            "ci_lower_95",
            "ci_upper_95",
            "calibration_error",
            "predicted_probability_inside_95_ci",
        ]
    ].to_string(index=False)
)


# ----------------------------------------------------------------------
# [9] CONFIDENCE INTERVAL SUMMARY
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[9] CONFIDENCE INTERVAL SUMMARY")
print("=" * 70)

outside_ci_count = int(
    (
        ~decile_df[
            "predicted_probability_inside_95_ci"
        ]
    ).sum()
)

inside_ci_count = (
    N_BINS - outside_ci_count
)

print(
    f"Deciles inside 95% CI:     "
    f"{inside_ci_count}/{N_BINS}"
)

print(
    f"Deciles outside 95% CI:    "
    f"{outside_ci_count}/{N_BINS}"
)

if outside_ci_count > 0:

    outside_deciles = decile_df.loc[
        ~decile_df[
            "predicted_probability_inside_95_ci"
        ],
        "risk_decile",
    ].tolist()

    print(
        "Deciles outside CI:        "
        f"{outside_deciles}"
    )

else:

    print(
        "Deciles outside CI:        None"
    )


# ----------------------------------------------------------------------
# [10] AGGREGATE CALIBRATION ERRORS
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[10] AGGREGATE CALIBRATION ERRORS")
print("=" * 70)

mean_absolute_calibration_error = (
    decile_df[
        "absolute_calibration_error"
    ].mean()
)

maximum_absolute_calibration_error = (
    decile_df[
        "absolute_calibration_error"
    ].max()
)

weighted_calibration_error = (
    np.average(
        decile_df["calibration_error"],
        weights=decile_df["n_observations"],
    )
)

print(
    f"Mean absolute calibration error: "
    f"{mean_absolute_calibration_error:.6f}"
)

print(
    f"Maximum absolute calibration error: "
    f"{maximum_absolute_calibration_error:.6f}"
)

print(
    f"Weighted calibration error: "
    f"{weighted_calibration_error:.6f}"
)


# ----------------------------------------------------------------------
# [11] SAVING CALIBRATION METRICS
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[11] SAVING CALIBRATION METRICS")
print("=" * 70)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


metrics_df = pd.DataFrame(
    [
        {
            "metric": "observations",
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
            "metric": "mean_absolute_calibration_error",
            "value": mean_absolute_calibration_error,
        },
        {
            "metric": "maximum_absolute_calibration_error",
            "value": maximum_absolute_calibration_error,
        },
        {
            "metric": "weighted_calibration_error",
            "value": weighted_calibration_error,
        },
        {
            "metric": "brier_score",
            "value": brier_score,
        },
        {
            "metric": "log_loss",
            "value": log_loss,
        },
        {
            "metric": "deciles_inside_95_ci",
            "value": inside_ci_count,
        },
        {
            "metric": "deciles_outside_95_ci",
            "value": outside_ci_count,
        },
    ]
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


decile_df.to_csv(
    DECILE_PATH,
    index=False,
)


print(
    f"Metrics saved:            {METRICS_PATH}"
)

print(
    f"Decile table saved:       {DECILE_PATH}"
)


# ----------------------------------------------------------------------
# [12] CALIBRATION CURVE WITH CONFIDENCE INTERVALS
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[12] CALIBRATION CURVE WITH 95% CI")
print("=" * 70)


plt.figure(figsize=(9, 7))


# Perfect calibration line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)


# Calibration points
plt.errorbar(
    decile_df[
        "mean_predicted_probability"
    ],
    decile_df[
        "observed_bad_rate"
    ],
    yerr=[
        decile_df[
            "observed_bad_rate"
        ]
        - decile_df[
            "ci_lower_95"
        ],

        decile_df[
            "ci_upper_95"
        ]
        - decile_df[
            "observed_bad_rate"
        ],
    ],
    fmt="o",
    capsize=4,
    label="Risk deciles",
)


plt.xlabel(
    "Mean predicted probability"
)

plt.ylabel(
    "Observed bad rate"
)

plt.title(
    "Calibration Curve with 95% Confidence Intervals"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150,
)

plt.close()


print(
    f"Calibration curve saved: {PLOT_PATH}"
)


# ----------------------------------------------------------------------
# [13] VALIDATION SUMMARY
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[13] VALIDATION SUMMARY")
print("=" * 70)

print(
    f"Overall observed bad rate: "
    f"{observed_bad_rate:.4f}"
)

print(
    f"Mean predicted probability:"
    f"{mean_predicted_probability:.4f}"
)

print(
    f"Calibration-in-the-large: "
    f"{calibration_in_the_large:.4f}"
)

print(
    f"Mean absolute calibration error: "
    f"{mean_absolute_calibration_error:.4f}"
)

print(
    f"Maximum absolute calibration error: "
    f"{maximum_absolute_calibration_error:.4f}"
)

print(
    f"Brier score: "
    f"{brier_score:.4f}"
)

print(
    f"Log loss: "
    f"{log_loss:.4f}"
)

print(
    f"Deciles inside 95% CI: "
    f"{inside_ci_count}/{N_BINS}"
)

print(
    f"Deciles outside 95% CI: "
    f"{outside_ci_count}/{N_BINS}"
)


# ----------------------------------------------------------------------
# [14] FINAL INTERPRETATION
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("[14] FINAL INTERPRETATION")
print("=" * 70)

if outside_ci_count == 0:

    print(
        "No risk decile shows a predicted probability "
        "outside the 95% confidence interval of the "
        "observed bad rate."
    )

elif outside_ci_count <= 2:

    print(
        "A small number of risk deciles show predicted "
        "probabilities outside the 95% confidence interval "
        "of the observed bad rate."
    )

else:

    print(
        "Multiple risk deciles show predicted probabilities "
        "outside the 95% confidence interval of the "
        "observed bad rate."
    )


print()
print(
    "Note: Confidence intervals are used as a calibration "
    "diagnostic and should not be interpreted as a formal "
    "hypothesis test of model calibration."
)


# ----------------------------------------------------------------------
# CLEANUP
# ----------------------------------------------------------------------

df.drop(
    columns=["_probability_rank", "risk_decile"],
    inplace=True,
)


print()
print("=" * 70)
print("CALIBRATION ANALYSIS V0.3 COMPLETED")
print("=" * 70)