from pathlib import Path
import hashlib
import pandas as pd

def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def prediction_integrity(predictions: pd.DataFrame, id_column="SK_ID_CURR") -> dict:
    required = {id_column, "y_true", "y_probability", "y_prediction"}
    missing = sorted(required - set(predictions.columns))
    return {
        "required_columns_present": not missing,
        "missing_columns": missing,
        "unique_ids": int(predictions[id_column].nunique()) == len(predictions),
        "missing_predictions": int(predictions["y_probability"].isna().sum()),
    }
