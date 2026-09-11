from pathlib import Path
import pandas as pd

def validate_input_data(path: str | Path, target: str = "TARGET", id_column: str = "SK_ID_CURR") -> dict:
    df = pd.read_csv(path)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "target_present": target in df.columns,
        "id_present": id_column in df.columns,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_target": int(df[target].isna().sum()) if target in df else None,
        "target_values": sorted(df[target].dropna().unique().tolist()) if target in df else [],
    }
