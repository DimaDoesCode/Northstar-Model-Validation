from pathlib import Path
import pandas as pd
from .data_quality import validate_input_data
from .performance import core_metrics, calibration_summary, risk_bands
from .reproducibility import prediction_integrity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/application_train.csv"
PREDICTIONS = ROOT / "reports/baseline/validation_predictions.csv"

def main() -> None:
    print("NORTHSTAR MODEL VALIDATION — V1.0")
    print("=" * 60)

    dq = validate_input_data(DATA)
    print("\nData quality:", dq)

    if not PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS}. Run model/predict.py first."
        )

    pred = pd.read_csv(PREDICTIONS)
    print("\nPrediction integrity:", prediction_integrity(pred))

    metrics = core_metrics(pred["y_true"], pred["y_probability"])
    print("\nCore performance:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")

    print("\nCalibration:")
    for name, value in calibration_summary(pred["y_true"], pred["y_probability"]).items():
        print(f"  {name}: {value:.6f}")

    print("\nRisk bands:")
    print(risk_bands(pred["y_true"], pred["y_probability"]).to_string(index=False))

if __name__ == "__main__":
    main()
