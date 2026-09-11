import pandas as pd
from validation.performance import core_metrics

def test_core_metrics_on_stored_predictions():
    df = pd.read_csv("reports/baseline/validation_predictions.csv")
    m = core_metrics(df["y_true"], df["y_probability"])
    assert 0.0 < m["roc_auc"] < 1.0
    assert 0.0 < m["pr_auc"] < 1.0
    assert 0.0 < m["brier"] < 1.0
