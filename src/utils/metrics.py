# src/utils/metrics.py

from typing import Dict, Tuple

import numpy as np
from sklearn import metrics
import pandas as pd


def is_classification_target(y: pd.Series) -> bool:
    """Heuristic: non-numeric or few unique values -> classification."""
    try:
        _ = y.astype(float)
        numeric = True
    except Exception:
        numeric = False

    unique = y.nunique()

    if not numeric:
        return True
    if unique <= 20:
        return True
    return False


def get_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": metrics.f1_score(y_true, y_pred, average="weighted"),
    }


def get_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "r2": float(metrics.r2_score(y_true, y_pred)),
    }


def pick_main_metric(problem_type: str, metrics_dict: Dict[str, float]) -> Tuple[str, float]:
    """Return (metric_name, value) used for ranking. For regression, value is signed so max() works."""
    if problem_type == "classification":
        metric = "f1_macro" if "f1_macro" in metrics_dict else list(metrics_dict.keys())[0]
        return metric, metrics_dict[metric]
    else:
        metric = "rmse" if "rmse" in metrics_dict else list(metrics_dict.keys())[0]
        # smaller RMSE is better → use negative for easy max()
        return metric, -metrics_dict[metric]
