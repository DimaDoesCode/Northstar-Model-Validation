from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/application_train.csv")


def main():
    print("=" * 70)
    print("NORTHSTAR MODEL VALIDATION — DATA AUDIT V0.1")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)

    print("\n[1] DATASET")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Memory:  {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # ------------------------------------------------------------------
    # 2. Target
    # ------------------------------------------------------------------
    target = "TARGET"

    print("\n[2] TARGET")

    if target not in df.columns:
        print("ERROR: TARGET column not found!")
    else:
        print(df[target].value_counts(dropna=False).to_string())

        print("\nTarget distribution:")
        print(
            (df[target].value_counts(normalize=True, dropna=False) * 100)
            .round(2)
            .to_string()
        )

    # ------------------------------------------------------------------
    # 3. Data types
    # ------------------------------------------------------------------
    print("\n[3] DATA TYPES")
    print(df.dtypes.value_counts().to_string())

    # ------------------------------------------------------------------
    # 4. Missing values
    # ------------------------------------------------------------------
    print("\n[4] MISSING VALUES")

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_report = pd.DataFrame(
        {
            "missing": missing,
            "missing_pct": missing_pct,
        }
    )

    missing_report = missing_report[
        missing_report["missing"] > 0
    ].sort_values("missing_pct", ascending=False)

    print(f"Columns with missing values: {len(missing_report)}")
    print(missing_report.head(20).to_string())

    # ------------------------------------------------------------------
    # 5. Duplicates
    # ------------------------------------------------------------------
    print("\n[5] DUPLICATES")

    print(f"Duplicate rows: {df.duplicated().sum():,}")

    if "SK_ID_CURR" in df.columns:
        print(
            f"Duplicate SK_ID_CURR: "
            f"{df['SK_ID_CURR'].duplicated().sum():,}"
        )

    # ------------------------------------------------------------------
    # 6. Constant columns
    # ------------------------------------------------------------------
    print("\n[6] CONSTANT COLUMNS")

    constant = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    print(f"Constant columns: {len(constant)}")

    if constant:
        print(constant)

    # ------------------------------------------------------------------
    # 7. Numeric summary
    # ------------------------------------------------------------------
    print("\n[7] NUMERIC VARIABLES")

    numeric = df.select_dtypes(include="number")

    summary = numeric.describe().T

    summary["missing_pct"] = (
        df[numeric.columns].isna().mean() * 100
    ).round(2)

    print(
        summary[
            ["count", "mean", "std", "min", "50%", "max", "missing_pct"]
        ]
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # 8. Categorical variables
    # ------------------------------------------------------------------
    print("\n[8] CATEGORICAL VARIABLES")

    categorical = df.select_dtypes(include=["object", "category"])

    print(f"Categorical columns: {len(categorical.columns)}")

    for col in categorical.columns:
        n_unique = categorical[col].nunique(dropna=False)

        print(
            f"{col}: "
            f"{n_unique:,} unique values"
        )

        # Show values only for relatively small categorical variables
        if n_unique <= 20:
            print(
                categorical[col]
                .value_counts(dropna=False)
                .head(20)
                .to_string()
            )

    # ------------------------------------------------------------------
    # 9. Suspicious numeric values
    # ------------------------------------------------------------------
    print("\n[9] BASIC VALUE CHECKS")

    checks = {
        "AMT_INCOME_TOTAL": "negative values",
        "AMT_CREDIT": "negative values",
        "AMT_ANNUITY": "negative values",
        "AMT_GOODS_PRICE": "negative values",
        "CNT_CHILDREN": "negative values",
        "CNT_FAM_MEMBERS": "negative values",
    }

    for column, description in checks.items():
        if column not in df.columns:
            continue

        count = (df[column] < 0).sum()

        print(
            f"{column:25s} {description:20s}: {count:,}"
        )

    # ------------------------------------------------------------------
    # 10. Potentially suspicious target relationships
    # ------------------------------------------------------------------
    print("\n[10] SIMPLE TARGET RELATIONSHIPS")

    if target in df.columns:
        numeric_features = [
            col for col in numeric.columns
            if col != target
        ]

        correlations = (
            df[numeric_features + [target]]
            .corr(numeric_only=True)[target]
            .drop(target)
            .abs()
            .sort_values(ascending=False)
        )

        print(
            "Top 15 numeric features by absolute correlation "
            "with TARGET:"
        )
        print(correlations.head(15).to_string())

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()