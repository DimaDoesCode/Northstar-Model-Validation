from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports"

TRAIN_FILE = DATA_DIR / "application_train.csv"
TEST_FILE = DATA_DIR / "application_test.csv"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"


# ======================================================================
# HELPERS
# ======================================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_distribution(series, name):
    """Print robust distribution statistics."""
    print(f"\n{name}")
    print(f"count:  {series.count():,}")
    print(f"missing: {series.isna().sum():,} "
          f"({series.isna().mean() * 100:.2f}%)")

    if series.notna().sum() == 0:
        return

    print(f"mean:   {series.mean():,.4f}")
    print(f"median: {series.median():,.4f}")
    print(f"std:    {series.std():,.4f}")

    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999]:
        print(f"q{q * 100:g}:    {series.quantile(q):,.4f}")

    print(f"min:    {series.min():,.4f}")
    print(f"max:    {series.max():,.4f}")


# ======================================================================
# [0] LOAD DATA
# ======================================================================

section("[0] LOADING DATA")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")


# ======================================================================
# [1] AMT_REQ_CREDIT_BUREAU_MON / QRT
# ======================================================================

section("[1] CREDIT BUREAU FEATURES")

print(
    "Investigating the large train/test distribution shift "
    "in AMT_REQ_CREDIT_BUREAU_MON and QRT."
)

bureau_features = [
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
]

bureau_summary = []

for col in bureau_features:

    print()
    print("-" * 70)
    print(col)

    train_col = train[col]
    test_col = test[col]

    print_distribution(train_col, "TRAIN")
    print_distribution(test_col, "TEST")

    # Value frequencies
    train_counts = train_col.value_counts(dropna=False).head(10)
    test_counts = test_col.value_counts(dropna=False).head(10)

    print("\nTop TRAIN values:")
    print(train_counts.to_string())

    print("\nTop TEST values:")
    print(test_counts.to_string())

    # Key statistics for zero/non-zero
    train_nonzero = (train_col > 0).mean()
    test_nonzero = (test_col > 0).mean()

    train_zero = (train_col == 0).mean()
    test_zero = (test_col == 0).mean()

    print(
        f"\nZero rate:     train={train_zero * 100:.2f}% "
        f"test={test_zero * 100:.2f}%"
    )

    print(
        f"Non-zero rate: train={train_nonzero * 100:.2f}% "
        f"test={test_nonzero * 100:.2f}%"
    )

    # Target rate by value category in TRAIN
    tmp = train[[col, TARGET]].copy()

    tmp["VALUE_GROUP"] = np.select(
        [
            tmp[col].isna(),
            tmp[col] == 0,
            tmp[col] == 1,
            tmp[col] == 2,
            tmp[col] >= 3,
        ],
        [
            "missing",
            "0",
            "1",
            "2",
            "3+",
        ],
        default="other",
    )

    target_by_value = (
        tmp.groupby("VALUE_GROUP", observed=True)[TARGET]
        .agg(["count", "mean"])
        .reset_index()
    )

    target_by_value["target_rate_%"] = (
        target_by_value["mean"] * 100
    )

    target_by_value = target_by_value.drop(columns="mean")

    print("\nTRAIN TARGET rate by value group:")
    print(target_by_value.to_string(index=False))

    target_by_value.to_csv(
        REPORT_DIR / f"{col.lower()}_target_by_value.csv",
        index=False,
    )

    bureau_summary.append({
        "feature": col,
        "train_missing_pct": train_col.isna().mean() * 100,
        "test_missing_pct": test_col.isna().mean() * 100,
        "train_zero_pct": train_zero * 100,
        "test_zero_pct": test_zero * 100,
        "train_nonzero_pct": train_nonzero * 100,
        "test_nonzero_pct": test_nonzero * 100,
        "train_mean": train_col.mean(),
        "test_mean": test_col.mean(),
        "train_median": train_col.median(),
        "test_median": test_col.median(),
    })

bureau_summary_df = pd.DataFrame(bureau_summary)

bureau_summary_df.to_csv(
    REPORT_DIR / "bureau_train_test_investigation.csv",
    index=False,
)

print("\nSummary:")
print(bureau_summary_df.to_string(index=False))


# ======================================================================
# [2] EXT_SOURCE_1 MISSINGNESS
# ======================================================================

section("[2] EXT_SOURCE_1 MISSINGNESS")

print(
    "Investigating why EXT_SOURCE_1 missingness differs "
    "substantially between train and test."
)

col = "EXT_SOURCE_1"

train_missing = train[col].isna()
test_missing = test[col].isna()

print(
    f"\nTRAIN missing: {train_missing.sum():,} "
    f"({train_missing.mean() * 100:.2f}%)"
)

print(
    f"TEST missing:  {test_missing.sum():,} "
    f"({test_missing.mean() * 100:.2f}%)"
)

print(
    f"Difference:    "
    f"{(test_missing.mean() - train_missing.mean()) * 100:+.2f} pp"
)


# Target rate in train
tmp = train[[col, TARGET]].copy()
tmp["EXT_SOURCE_1_MISSING"] = tmp[col].isna()

missing_target = (
    tmp.groupby("EXT_SOURCE_1_MISSING")[TARGET]
    .agg(["count", "mean"])
    .reset_index()
)

missing_target["target_rate_%"] = missing_target["mean"] * 100
missing_target = missing_target.drop(columns="mean")

print("\nTRAIN TARGET by EXT_SOURCE_1 missingness:")
print(missing_target.to_string(index=False))

missing_target.to_csv(
    REPORT_DIR / "ext_source_1_missingness_target.csv",
    index=False,
)


# Investigate missingness against major categorical variables
categorical_features = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

ext_source_groups = []

for feature in categorical_features:

    if feature not in train.columns:
        continue

    grouped = (
        train.groupby(
            [feature, train[col].isna().rename("EXT_SOURCE_1_MISSING")],
            dropna=False,
            observed=True,
        )[TARGET]
        .agg(["count", "mean"])
        .reset_index()
    )

    grouped["target_rate_%"] = grouped["mean"] * 100
    grouped = grouped.drop(columns="mean")

    grouped.insert(0, "grouping_feature", feature)

    ext_source_groups.append(grouped)

    print(f"\n{feature}:")
    print(grouped.to_string(index=False))

if ext_source_groups:
    ext_source_groups_df = pd.concat(
        ext_source_groups,
        ignore_index=True,
    )

    ext_source_groups_df.to_csv(
        REPORT_DIR / "ext_source_1_missingness_by_category.csv",
        index=False,
    )


# Compare EXT_SOURCE_1 distribution among non-missing values
print("\nNon-missing EXT_SOURCE_1 distribution:")

print_distribution(
    train.loc[~train_missing, col],
    "TRAIN",
)

print_distribution(
    test.loc[~test_missing, col],
    "TEST",
)


# ======================================================================
# [3] EXTREME AMT_INCOME_TOTAL
# ======================================================================

section("[3] AMT_INCOME_TOTAL EXTREMES")

print(
    "Investigating extreme AMT_INCOME_TOTAL values "
    "without removing or modifying them."
)

col = "AMT_INCOME_TOTAL"

print_distribution(train[col], "AMT_INCOME_TOTAL — TRAIN")


# Explicit extreme thresholds
thresholds = [
    ("99th percentile", train[col].quantile(0.99)),
    ("99.9th percentile", train[col].quantile(0.999)),
    ("99.99th percentile", train[col].quantile(0.9999)),
]

income_extremes = []

for label, threshold in thresholds:

    mask = train[col] > threshold

    subset = train.loc[
        mask,
        [
            ID_COL,
            col,
            TARGET,
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "CNT_CHILDREN",
            "AMT_CREDIT",
            "AMT_ANNUITY",
        ],
    ].copy()

    target_rate = subset[TARGET].mean() * 100

    print()
    print(f"{label}: {threshold:,.2f}")
    print(f"Rows above threshold: {len(subset):,}")
    print(f"TARGET rate:          {target_rate:.2f}%")

    income_extremes.append({
        "threshold": label,
        "threshold_value": threshold,
        "rows": len(subset),
        "share_pct": len(subset) / len(train) * 100,
        "target_rate_%": target_rate,
    })

    # Save the actual extreme rows for inspection
    safe_name = (
        label
        .lower()
        .replace(" ", "_")
        .replace(".", "")
    )

    subset.to_csv(
        REPORT_DIR / f"income_extremes_{safe_name}.csv",
        index=False,
    )


income_summary_df = pd.DataFrame(income_extremes)

income_summary_df.to_csv(
    REPORT_DIR / "income_extremes_summary.csv",
    index=False,
)

print("\nExtreme income summary:")
print(income_summary_df.to_string(index=False))


# ======================================================================
# [4] EXTREME INCOME — CONTEXT
# ======================================================================

section("[4] EXTREME INCOME CONTEXT")

# Use the 99.9th percentile as the main investigation threshold
income_threshold = train[col].quantile(0.999)

extreme = train[train[col] > income_threshold].copy()
normal = train[train[col] <= income_threshold].copy()

print(
    f"Using 99.9th percentile threshold: "
    f"{income_threshold:,.2f}"
)

print(
    f"Extreme rows: {len(extreme):,} "
    f"({len(extreme) / len(train) * 100:.3f}%)"
)

print("\nIncome type distribution — extreme:")
print(
    extreme["NAME_INCOME_TYPE"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nIncome type distribution — normal:")
print(
    normal["NAME_INCOME_TYPE"]
    .value_counts(normalize=True, dropna=False)
    .head(15)
    .mul(100)
    .round(2)
    .to_string()
)

print("\nEducation distribution — extreme:")
print(
    extreme["NAME_EDUCATION_TYPE"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nHousing distribution — extreme:")
print(
    extreme["NAME_HOUSING_TYPE"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nExtreme income rows:")
print(
    extreme[
        [
            ID_COL,
            "AMT_INCOME_TOTAL",
            TARGET,
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "CNT_CHILDREN",
            "AMT_CREDIT",
            "AMT_ANNUITY",
        ]
    ]
    .sort_values("AMT_INCOME_TOTAL", ascending=False)
    .head(30)
    .to_string(index=False)
)


# ======================================================================
# [5] SUMMARY
# ======================================================================

section("[5] INVESTIGATION SUMMARY")

print("Three targeted investigations completed:")
print("1. Credit bureau train/test distribution shift")
print("2. EXT_SOURCE_1 train/test missingness shift")
print("3. Extreme AMT_INCOME_TOTAL values")

print("\nReports saved to:")
print(REPORT_DIR)

print("\n" + "=" * 70)
print("INVESTIGATION V0.2 COMPLETE")
print("=" * 70)