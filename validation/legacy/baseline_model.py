"""
NORTHSTAR MODEL VALIDATION
Baseline Model V0.1

Purpose:
    Build a simple, transparent Logistic Regression baseline
    for subsequent model validation.

The baseline is intentionally simple and reproducible.
Its purpose is not to maximize predictive performance,
but to establish a transparent reference model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TARGET = "TARGET"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "baseline"
MODEL_PATH = OUTPUT_DIR / "baseline_logistic_regression.joblib"

VALIDATION_SIZE = 0.20
THRESHOLD = 0.50


# ============================================================
# HELPERS
# ============================================================

def calculate_gini(roc_auc: float) -> float:
    """Convert ROC-AUC to Gini coefficient."""
    return 2 * roc_auc - 1


def calculate_ks(y_true, y_score) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.

    KS = max(TPR - FPR)
    """
    data = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "score": np.asarray(y_score),
        }
    ).sort_values("score", ascending=False)

    total_good = (data["target"] == 0).sum()
    total_bad = (data["target"] == 1).sum()

    data["cum_bad"] = (
        (data["target"] == 1).cumsum() / total_bad
    )
    data["cum_good"] = (
        (data["target"] == 0).cumsum() / total_good
    )

    ks = (data["cum_bad"] - data["cum_good"]).abs().max()

    return float(ks)


def print_class_distribution(y, name):
    """Print target distribution."""
    counts = y.value_counts().sort_index()
    percentages = y.value_counts(normalize=True).sort_index() * 100

    print(f"\n{name} target distribution:")

    for cls in counts.index:
        print(
            f"  TARGET={cls}: "
            f"{counts[cls]:,} "
            f"({percentages[cls]:.2f}%)"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NORTHSTAR MODEL VALIDATION — BASELINE MODEL V0.1")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print("\n[1] LOADING DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    # --------------------------------------------------------
    # 2. TARGET / FEATURES
    # --------------------------------------------------------

    print("\n[2] PREPARING FEATURES")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # Explicitly remove identifier.
    # It has no legitimate predictive meaning.
    id_columns = [
        column
        for column in ["SK_ID_CURR"]
        if column in X.columns
    ]

    if id_columns:
        print(f"Removing ID columns: {id_columns}")
        X = X.drop(columns=id_columns)

    print(f"Features after ID removal: {X.shape[1]}")

    print_class_distribution(y, "Full dataset")

    # --------------------------------------------------------
    # 3. TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    print("\n[3] TRAIN / VALIDATION SPLIT")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train shape:      {X_train.shape}")
    print(f"Validation shape: {X_valid.shape}")

    print_class_distribution(y_train, "Train")
    print_class_distribution(y_valid, "Validation")

    # --------------------------------------------------------
    # 4. IDENTIFY FEATURE TYPES
    # --------------------------------------------------------

    print("\n[4] FEATURE TYPES")

    numeric_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"Numeric features:      {len(numeric_features)}")
    print(f"Categorical features:  {len(categorical_features)}")

    # --------------------------------------------------------
    # 5. PREPROCESSING
    # --------------------------------------------------------

    print("\n[5] BUILDING PREPROCESSING PIPELINE")

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # --------------------------------------------------------
    # 6. MODEL
    # --------------------------------------------------------

    print("\n[6] BUILDING LOGISTIC REGRESSION")

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # --------------------------------------------------------
    # 7. FIT
    # --------------------------------------------------------

    print("\n[7] TRAINING")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------------
    # 8. PREDICTIONS
    # --------------------------------------------------------

    print("\n[8] GENERATING VALIDATION PREDICTIONS")

    y_valid_probability = pipeline.predict_proba(
        X_valid
    )[:, 1]

    y_valid_prediction = (
        y_valid_probability >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # 9. METRICS
    # --------------------------------------------------------

    print("\n[9] VALIDATION METRICS")

    roc_auc = roc_auc_score(
        y_valid,
        y_valid_probability,
    )

    pr_auc = average_precision_score(
        y_valid,
        y_valid_probability,
    )

    gini = calculate_gini(roc_auc)

    ks = calculate_ks(
        y_valid,
        y_valid_probability,
    )

    cm = confusion_matrix(
        y_valid,
        y_valid_prediction,
    )

    tn, fp, fn, tp = cm.ravel()

    print(f"\n  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  Gini    : {gini:.4f}")
    print(f"  KS      : {ks:.4f}")

    print(f"\n  Threshold: {THRESHOLD:.2f}")

    print("\n  Confusion matrix:")
    print(cm)

    print("\n  Classification counts:")
    print(f"    True Negatives : {tn:,}")
    print(f"    False Positives: {fp:,}")
    print(f"    False Negatives: {fn:,}")
    print(f"    True Positives : {tp:,}")

    # --------------------------------------------------------
    # 10. MODEL COEFFICIENTS
    # --------------------------------------------------------

    print("\n[10] EXTRACTING MODEL COEFFICIENTS")

    fitted_preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    fitted_model = pipeline.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    coefficients = fitted_model.coef_[0]

    coefficients_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        )
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )
    )

    print("\nTop 20 coefficients by absolute magnitude:")

    print(
        coefficients_df[
            ["feature", "coefficient"]
        ].head(20).to_string(index=False)
    )

    # --------------------------------------------------------
    # 11. SAVE RESULTS
    # --------------------------------------------------------

    print("\n[11] SAVING RESULTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    coefficients_df.to_csv(
        OUTPUT_DIR / "coefficients.csv",
        index=False,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "gini": gini,
                "ks": ks,
                "threshold": THRESHOLD,
                "random_state": RANDOM_STATE,
                "validation_size": VALIDATION_SIZE,
            }
        ]
    )

    metrics.to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False,
    )

    predictions = pd.DataFrame(
        {
            "SK_ID_CURR": df.loc[
                y_valid.index,
                "SK_ID_CURR",
            ].to_numpy(),
            "y_true": y_valid.to_numpy(),
            "y_probability": y_valid_probability,
            "y_prediction": y_valid_prediction,
        }
    )

    predictions.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    print(f"\nModel saved to:")
    print(f"  {MODEL_PATH}")

    print("\nReports saved to:")
    print(f"  {OUTPUT_DIR}")

    # --------------------------------------------------------
    # 12. SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE MODEL SUMMARY")
    print("=" * 70)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"Gini    : {gini:.4f}")
    print(f"KS      : {ks:.4f}")

    print("\nBaseline model completed successfully.")


if __name__ == "__main__":
    main()