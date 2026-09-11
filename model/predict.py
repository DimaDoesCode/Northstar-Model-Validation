from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "model/model_package"
DATA = ROOT / "data/raw/application_train.csv"
OUTPUT = ROOT / "reports/baseline/validation_predictions.csv"

TARGET = "TARGET"
ID = "SK_ID_CURR"

def predict(df: pd.DataFrame) -> pd.DataFrame:
    preprocessor = joblib.load(PACKAGE / "preprocessing.joblib")
    model = joblib.load(PACKAGE / "model.joblib")

    X = df.drop(columns=[TARGET, ID], errors="ignore")
    probabilities = model.predict_proba(preprocessor.transform(X))[:, 1]
    return pd.DataFrame({
        ID: df[ID].values,
        "y_true": df[TARGET].values,
        "y_probability": probabilities,
        "y_prediction": (probabilities >= 0.50).astype(int),
    })

def main() -> None:
    df = pd.read_csv(DATA)
    result = predict(df)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Predictions written to {OUTPUT}")

if __name__ == "__main__":
    main()
