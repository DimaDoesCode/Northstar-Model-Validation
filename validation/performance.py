import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

def gini(roc_auc: float) -> float:
    return 2 * roc_auc - 1

def ks_statistic(y_true, scores) -> float:
    data = pd.DataFrame({"y": y_true, "score": scores}).sort_values("score", ascending=False)
    bad = (data["y"] == 1).sum()
    good = (data["y"] == 0).sum()
    cum_bad = (data["y"] == 1).cumsum() / bad
    cum_good = (data["y"] == 0).cumsum() / good
    return float((cum_bad - cum_good).abs().max())

def core_metrics(y_true, probabilities) -> dict:
    auc = roc_auc_score(y_true, probabilities)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "gini": float(gini(auc)),
        "ks": float(ks_statistic(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }

def calibration_summary(y_true, probabilities) -> dict:
    y_true = np.asarray(y_true)
    p = np.asarray(probabilities)
    return {
        "observed_bad_rate": float(y_true.mean()),
        "mean_predicted_probability": float(p.mean()),
        "calibration_in_the_large": float(p.mean() - y_true.mean()),
        "brier": float(brier_score_loss(y_true, p)),
    }

def risk_bands(y_true, probabilities, n_bands=10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "probability": probabilities})
    df["risk_band"] = pd.qcut(df["probability"], q=n_bands, labels=False, duplicates="drop") + 1
    result = df.groupby("risk_band", observed=True).agg(
        n=("y_true", "size"),
        predicted_probability=("probability", "mean"),
        observed_default_rate=("y_true", "mean"),
        defaults=("y_true", "sum"),
    ).reset_index()
    result["capture"] = result["defaults"] / result["defaults"].sum()
    result["cumulative_capture"] = result["capture"].cumsum()
    return result
