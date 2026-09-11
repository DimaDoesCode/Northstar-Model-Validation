"""
Northstar Model Validation
Data Investigation v0.1

Purpose:
    Investigate the main risks identified by audit_data.py:
    1. Missingness patterns
    2. DAYS_EMPLOYED sentinel value
    3. Train / test distribution shift

This script does not modify the data.
It produces investigation results only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports"

TRAIN_FILE = DATA_DIR / "application_train.csv"
TEST_FILE = DATA_DIR / "application_test.csv"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

MISSINGNESS_THRESHOLD = 0.30
SHIFT_THRESHOLD = 0.10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def target_rate(df):
    return df[TARGET].mean() * 100


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print_header("NORTHSTAR MODEL VALIDATION — DATA INVESTIGATION V0.1")

print("[0] LOADING DATA")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

REPORT_DIR.mkdir(exist_ok=True)


# =====================================================================
# 1. MISSINGNESS INVESTIGATION
# =====================================================================

print_header("[1] MISSINGNESS INVESTIGATION")

print("Checking whether missing values occur in groups.")


# ---------------------------------------------------------------------
# 1.1 Missingness patterns
# ---------------------------------------------------------------------

missing = train.isna()

missing_counts = missing.sum(axis=1)

print("\nNumber of missing values per row:")

missing_summary = pd.Series({
    "count": missing_counts.count(),
    "mean": missing_counts.mean(),
    "50%": missing_counts.quantile(0.50),
    "75%": missing_counts.quantile(0.75),
    "90%": missing_counts.quantile(0.90),
    "95%": missing_counts.quantile(0.95),
    "99%": missing_counts.quantile(0.99),
    "max": missing_counts.max(),
})

print(missing_summary.to_string())

# ---------------------------------------------------------------------
# 1.2 Correlation between missingness indicators
# ---------------------------------------------------------------------

missing_cols = [
    col
    for col in train.columns
    if train[col].isna().mean() >= MISSINGNESS_THRESHOLD
]

if len(missing_cols) > 1:

    missing_corr = (
        train[missing_cols]
        .isna()
        .astype(int)
        .corr()
    )

    pairs = []

    for i, col1 in enumerate(missing_cols):
        for col2 in missing_cols[i + 1:]:
            corr = missing_corr.loc[col1, col2]

            if corr >= 0.90:
                pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "missingness_corr": corr,
                })

    pairs_df = pd.DataFrame(pairs)

    print(
        f"\nHighly correlated missingness pairs (corr >= 0.90): "
        f"{len(pairs_df)}"
    )

    if not pairs_df.empty:
        print(
            pairs_df.sort_values(
                "missingness_corr",
                ascending=False,
            ).head(30).to_string(index=False)
        )

        pairs_df.to_csv(
            REPORT_DIR / "missingness_correlations.csv",
            index=False,
        )

else:
    print("Not enough columns for missingness correlation analysis.")


# ---------------------------------------------------------------------
# 1.3 Missingness burden vs TARGET
# ---------------------------------------------------------------------

train["_MISSING_COUNT"] = missing_counts

missing_bins = pd.cut(
    train["_MISSING_COUNT"],
    bins=[-1, 0, 5, 10, 20, 40, 60, 100, np.inf],
    labels=[
        "0",
        "1-5",
        "6-10",
        "11-20",
        "21-40",
        "41-60",
        "61-100",
        "100+",
    ],
)

missing_target = (
    train.groupby(
        missing_bins,
        observed=False,
    )[TARGET]
    .agg(["count", "mean"])
)

missing_target["target_rate_%"] = missing_target["mean"] * 100
missing_target = missing_target.drop(columns="mean")

print("\nTARGET rate by number of missing values per row:")

print(
    missing_target.to_string(
        formatters={
            "target_rate_%": "{:.2f}".format
        }
    )
)

missing_target.to_csv(
    REPORT_DIR / "missingness_vs_target.csv"
)


# ---------------------------------------------------------------------
# 1.4 Most informative missingness indicators
# ---------------------------------------------------------------------

overall_target = train[TARGET].mean()

rows = []

for col in missing_cols:

    is_missing = train[col].isna()

    missing_rate = train.loc[is_missing, TARGET].mean()
    present_rate = train.loc[~is_missing, TARGET].mean()

    rows.append({
        "feature": col,
        "missing_pct": is_missing.mean() * 100,
        "target_missing_%": missing_rate * 100,
        "target_present_%": present_rate * 100,
        "difference_pp": (
            missing_rate - present_rate
        ) * 100,
    })

missing_effect = (
    pd.DataFrame(rows)
    .sort_values(
        "difference_pp",
        key=abs,
        ascending=False,
    )
)

print("\nLargest TARGET differences associated with missingness:")

print(
    missing_effect.to_string(
        index=False,
        formatters={
            "missing_pct": "{:.2f}".format,
            "target_missing_%": "{:.2f}".format,
            "target_present_%": "{:.2f}".format,
            "difference_pp": "{:+.2f}".format,
        },
    )
)

missing_effect.to_csv(
    REPORT_DIR / "missingness_target_effect.csv",
    index=False,
)


# =====================================================================
# 2. DAYS_EMPLOYED INVESTIGATION
# =====================================================================

print_header("[2] DAYS_EMPLOYED INVESTIGATION")

if "DAYS_EMPLOYED" in train.columns:

    sentinel = train["DAYS_EMPLOYED"].eq(365243)

    train["DAYS_EMPLOYED_SENTINEL"] = sentinel.astype(int)

    print(
        f"Sentinel rows: {sentinel.sum():,} "
        f"({sentinel.mean() * 100:.2f}%)"
    )

    print("\nTARGET:")

    print(
        f"Sentinel: {target_rate(train.loc[sentinel]):.2f}%"
    )

    print(
        f"Normal:   {target_rate(train.loc[~sentinel]):.2f}%"
    )

    # -------------------------------------------------------------
    # Sentinel vs income type
    # -------------------------------------------------------------

    if "NAME_INCOME_TYPE" in train.columns:

        sentinel_income = (
            train.groupby(
                ["NAME_INCOME_TYPE", "DAYS_EMPLOYED_SENTINEL"]
            )[TARGET]
            .agg(["count", "mean"])
        )

        sentinel_income["target_rate_%"] = (
            sentinel_income["mean"] * 100
        )

        sentinel_income = sentinel_income.drop(
            columns="mean"
        )

        print(
            "\nDAYS_EMPLOYED sentinel by income type:"
        )

        print(
            sentinel_income.to_string(
                formatters={
                    "target_rate_%": "{:.2f}".format
                }
            )
        )

        sentinel_income.to_csv(
            REPORT_DIR / "days_employed_by_income_type.csv"
        )

    # -------------------------------------------------------------
    # Sentinel vs age
    # -------------------------------------------------------------

    if "DAYS_BIRTH" in train.columns:

        train["_AGE_YEARS"] = (
            -train["DAYS_BIRTH"] / 365.25
        )

        age_bins = pd.cut(
            train["_AGE_YEARS"],
            bins=[0, 25, 30, 35, 40, 50, 60, 100],
            right=False,
        )

        age_summary = (
            train.groupby(
                [age_bins, "DAYS_EMPLOYED_SENTINEL"],
                observed=False,
            )[TARGET]
            .agg(["count", "mean"])
        )

        age_summary["target_rate_%"] = (
            age_summary["mean"] * 100
        )

        age_summary = age_summary.drop(columns="mean")

        print("\nDAYS_EMPLOYED sentinel by age group:")

        print(
            age_summary.to_string(
                formatters={
                    "target_rate_%": "{:.2f}".format
                }
            )
        )

        age_summary.to_csv(
            REPORT_DIR / "days_employed_by_age.csv"
        )

else:
    print("DAYS_EMPLOYED not found.")


# =====================================================================
# 3. TRAIN / TEST DISTRIBUTION SHIFT
# =====================================================================

print_header("[3] TRAIN / TEST DISTRIBUTION SHIFT")

print(
    "Comparing numeric distributions using standardized mean difference."
)


numeric_cols = train.select_dtypes(
    include=[np.number]
).columns.tolist()

numeric_cols = [
    col
    for col in numeric_cols
    if col not in [
        TARGET,
        ID_COL,
        "_MISSING_COUNT",
        "DAYS_EMPLOYED_SENTINEL",
        "_AGE_YEARS",
    ]
]


shift_rows = []

for col in numeric_cols:

    train_mean = train[col].mean()
    test_mean = test[col].mean()

    train_std = train[col].std()
    test_std = test[col].std()

    pooled_std = np.sqrt(
        (train_std ** 2 + test_std ** 2) / 2
    )

    if pooled_std == 0 or pd.isna(pooled_std):
        smd = 0.0
    else:
        smd = abs(train_mean - test_mean) / pooled_std

    train_missing = train[col].isna().mean()
    test_missing = test[col].isna().mean()

    shift_rows.append({
        "feature": col,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "train_missing_pct": train_missing * 100,
        "test_missing_pct": test_missing * 100,
        "missing_difference_pp": (
            test_missing - train_missing
        ) * 100,
        "SMD": smd,
    })


shift_df = (
    pd.DataFrame(shift_rows)
    .sort_values("SMD", ascending=False)
)

print(
    f"\nFeatures with SMD >= {SHIFT_THRESHOLD}: "
    f"{(shift_df['SMD'] >= SHIFT_THRESHOLD).sum()}"
)

print("\nLargest train/test shifts:")

print(
    shift_df.head(30).to_string(
        index=False,
        formatters={
            "train_mean": "{:.4f}".format,
            "test_mean": "{:.4f}".format,
            "train_missing_pct": "{:.2f}".format,
            "test_missing_pct": "{:.2f}".format,
            "missing_difference_pp": "{:+.2f}".format,
            "SMD": "{:.3f}".format,
        },
    )
)

shift_df.to_csv(
    REPORT_DIR / "train_test_numeric_shift.csv",
    index=False,
)


# =====================================================================
# 4. TRAIN / TEST MISSINGNESS SHIFT
# =====================================================================

print_header("[4] TRAIN / TEST MISSINGNESS SHIFT")

common_cols = [
    col for col in train.columns
    if col in test.columns
    and col != TARGET
    and not col.startswith("_")
]

missing_shift = []

for col in common_cols:
    train_missing = train[col].isna().mean()
    test_missing = test[col].isna().mean()

    missing_shift.append({
        "feature": col,
        "train_missing_pct": train_missing * 100,
        "test_missing_pct": test_missing * 100,
        "difference_pp": (test_missing - train_missing) * 100,
    })

missing_shift_df = pd.DataFrame(missing_shift)

missing_shift_df["abs_difference_pp"] = (
    missing_shift_df["difference_pp"].abs()
)

missing_shift_df = missing_shift_df.sort_values(
    "abs_difference_pp",
    ascending=False
)

missing_shift_df.to_csv(
    REPORT_DIR / "train_test_missingness_shift.csv",
    index=False
)

print(
    missing_shift_df.head(20).to_string(index=False)
)


# =====================================================================
# 5. FINAL SUMMARY
# =====================================================================

print_header("[5] INVESTIGATION SUMMARY")

print(
    f"Columns with >= 30% missing: "
    f"{len(missing_cols)}"
)

if not pairs_df.empty:
    print(
        f"Highly correlated missingness pairs: "
        f"{len(pairs_df)}"
    )

if "DAYS_EMPLOYED" in train.columns:
    print(
        f"DAYS_EMPLOYED sentinel rows: "
        f"{sentinel.sum():,}"
    )

print(
    f"Numeric features investigated for shift: "
    f"{len(numeric_cols)}"
)

print(
    f"Numeric features with SMD >= {SHIFT_THRESHOLD}: "
    f"{(shift_df['SMD'] >= SHIFT_THRESHOLD).sum()}"
)

print("\nReports saved to:")
print(REPORT_DIR)

print("\n" + "=" * 70)
print("INVESTIGATION V0.1 COMPLETE")
print("=" * 70)