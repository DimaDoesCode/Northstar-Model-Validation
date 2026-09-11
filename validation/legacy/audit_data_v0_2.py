"""
Northstar Model Validation
Data Audit v0.2

Purpose:
    Identify potential data-quality and validation risks
    before feature engineering and model training.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"

TRAIN_FILE = DATA_DIR / "application_train.csv"
TEST_FILE = DATA_DIR / "application_test.csv"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

MISSINGNESS_THRESHOLD = 0.30
RARE_CATEGORY_THRESHOLD = 20


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print_header("NORTHSTAR MODEL VALIDATION — DATA AUDIT V0.2")

print("[0] LOADING DATA")

train = pd.read_csv(TRAIN_FILE)

print(f"Train shape: {train.shape}")

test = None

if TEST_FILE.exists():
    test = pd.read_csv(TEST_FILE)
    print(f"Test shape:  {test.shape}")
else:
    print("Test file not found — train/test checks skipped.")


# ---------------------------------------------------------------------
# 1. MISSINGNESS RISK
# ---------------------------------------------------------------------

print_header("[1] MISSINGNESS RISK")

missing_pct = train.isna().mean() * 100

missing_risk = (
    missing_pct[missing_pct >= MISSINGNESS_THRESHOLD * 100]
    .sort_values(ascending=False)
)

print(
    f"Columns with >= {MISSINGNESS_THRESHOLD:.0%} missing values: "
    f"{len(missing_risk)}"
)

if len(missing_risk) > 0:
    result = pd.DataFrame({
        "missing_pct": missing_risk.round(2)
    })

    print(result.to_string())


# ---------------------------------------------------------------------
# 2. MISSINGNESS VS TARGET
# ---------------------------------------------------------------------

print_header("[2] MISSINGNESS VS TARGET")

target_rate = train[TARGET].mean()

print(f"Overall TARGET=1 rate: {target_rate:.4%}")

rows = []

for col in missing_risk.index:
    missing = train[col].isna()

    rate_missing = train.loc[missing, TARGET].mean()
    rate_present = train.loc[~missing, TARGET].mean()

    rows.append({
        "feature": col,
        "missing_pct": missing.mean() * 100,
        "target_missing": rate_missing * 100,
        "target_present": rate_present * 100,
        "difference": (rate_missing - rate_present) * 100,
    })

missing_target = (
    pd.DataFrame(rows)
    .sort_values("difference", key=abs, ascending=False)
)

if len(missing_target) > 0:
    print(
        missing_target.to_string(
            index=False,
            formatters={
                "missing_pct": "{:.2f}".format,
                "target_missing": "{:.2f}".format,
                "target_present": "{:.2f}".format,
                "difference": "{:+.2f}".format,
            },
        )
    )


# ---------------------------------------------------------------------
# 3. SUSPICIOUS VALUES
# ---------------------------------------------------------------------

print_header("[3] SUSPICIOUS VALUES")

# Known sentinel in Home Credit dataset
if "DAYS_EMPLOYED" in train.columns:
    sentinel = train["DAYS_EMPLOYED"].eq(365243)

    print(
        f"DAYS_EMPLOYED == 365243: "
        f"{sentinel.sum():,} rows "
        f"({sentinel.mean() * 100:.2f}%)"
    )

    if sentinel.sum() > 0:
        print(
            f"TARGET rate among sentinel rows: "
            f"{train.loc[sentinel, TARGET].mean() * 100:.2f}%"
        )

        print(
            f"TARGET rate among other rows:   "
            f"{train.loc[~sentinel, TARGET].mean() * 100:.2f}%"
        )


# Extreme income values
if "AMT_INCOME_TOTAL" in train.columns:
    income = train["AMT_INCOME_TOTAL"]

    q99 = income.quantile(0.99)
    q999 = income.quantile(0.999)

    print("\nAMT_INCOME_TOTAL:")
    print(f"  99th percentile:  {q99:,.0f}")
    print(f"  99.9th percentile:{q999:,.0f}")
    print(f"  Maximum:          {income.max():,.0f}")

    extreme = income > q999

    print(
        f"  Above 99.9th percentile: "
        f"{extreme.sum():,} rows"
    )

    if extreme.sum() > 0:
        print(
            f"  TARGET rate: "
            f"{train.loc[extreme, TARGET].mean() * 100:.2f}%"
        )


# ---------------------------------------------------------------------
# 4. RARE CATEGORIES
# ---------------------------------------------------------------------

print_header("[4] RARE CATEGORIES")

categorical_cols = train.select_dtypes(
    include=["object", "category"]
).columns

rare_found = False

for col in categorical_cols:
    counts = train[col].value_counts(dropna=False)

    rare = counts[counts < RARE_CATEGORY_THRESHOLD]

    if len(rare) > 0:
        rare_found = True

        print(f"\n{col}:")
        print(rare.to_string())

if not rare_found:
    print("No rare categories found.")


# ---------------------------------------------------------------------
# 5. TRAIN / TEST ID OVERLAP
# ---------------------------------------------------------------------

print_header("[5] TRAIN / TEST OVERLAP")

if test is not None and ID_COL in train.columns and ID_COL in test.columns:

    train_ids = set(train[ID_COL])
    test_ids = set(test[ID_COL])

    overlap = train_ids.intersection(test_ids)

    print(f"Train IDs: {len(train_ids):,}")
    print(f"Test IDs:  {len(test_ids):,}")
    print(f"Overlap:   {len(overlap):,}")

    if len(overlap) == 0:
        print("STATUS: OK — no ID overlap.")
    else:
        print("STATUS: WARNING — train/test ID overlap detected.")

else:
    print("ID overlap check skipped.")


# ---------------------------------------------------------------------
# 6. TARGET RATE BY IMPORTANT CATEGORICAL FEATURES
# ---------------------------------------------------------------------

print_header("[6] TARGET RATE BY CATEGORICAL FEATURES")

important_categories = [
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "NAME_CONTRACT_TYPE",
]

for col in important_categories:

    if col not in train.columns:
        continue

    summary = (
        train.groupby(col, dropna=False)[TARGET]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    summary["target_rate_%"] = summary["mean"] * 100
    summary = summary.drop(columns="mean")

    print(f"\n{col}")
    print(
        summary.to_string(
            formatters={
                "target_rate_%": "{:.2f}".format
            }
        )
    )


# ---------------------------------------------------------------------
# 7. SUMMARY
# ---------------------------------------------------------------------

print_header("[7] AUDIT RISK SUMMARY")

print(f"Missingness >= 30%:       {len(missing_risk)} columns")

if "DAYS_EMPLOYED" in train.columns:
    print(
        "DAYS_EMPLOYED sentinel:   "
        f"{train['DAYS_EMPLOYED'].eq(365243).sum():,} rows"
    )

print(f"Categorical columns:       {len(categorical_cols)}")

if test is not None and ID_COL in train.columns and ID_COL in test.columns:
    print(f"Train/Test ID overlap:     {len(overlap):,}")
else:
    print("Train/Test ID overlap:     not checked")

print("\nKey risks to investigate:")
print("- Missingness may carry predictive information.")
print("- DAYS_EMPLOYED contains a sentinel value (365243).")
print("- Some categorical values are extremely rare.")
print("- TARGET is strongly imbalanced.")
print("- Extreme monetary values require distribution checks.")

print("\n" + "=" * 70)
print("AUDIT V0.2 COMPLETE")
print("=" * 70)