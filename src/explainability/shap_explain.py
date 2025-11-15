# src/explainability/shap_explain.py

import os
from typing import Optional

import numpy as np
import shap

from src.config import EXPLAIN_DIR


def compute_shap_for_model(model, X_sample, model_name: str) -> Optional[str]:
    """
    Compute SHAP values for a tree-based model (RF, XGB, LGBM).
    Saves a summary plot PNG and returns its path.
    """
    try:
        EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample)

        out_path = EXPLAIN_DIR / f"{model_name}_shap_summary.png"
        shap.plots.beeswarm(shap_values, show=False)
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return str(out_path)
    except Exception as e:
        print(f"[WARN] Failed to compute SHAP explanation: {e}")
        return None
