# src/agents/explainability_agent.py

from dataclasses import dataclass
from typing import Optional

import shap
import matplotlib.pyplot as plt


@dataclass
class ExplainabilityOutput:
    shap_summary_fig: Optional[plt.Figure]


class ExplainabilityAgent:
    def run(self, model, X_test):
        try:
            explainer = shap.TreeExplainer(model.named_steps["model"])
            shap_values = explainer.shap_values(
                model.named_steps["preprocess"].transform(X_test)
            )

            fig = plt.figure(figsize=(7, 5))
            shap.summary_plot(
                shap_values,
                feature_names=model.named_steps["preprocess"]
                .get_feature_names_out()
                .tolist(),
                show=False,
            )
        except Exception:
            fig = None

        return ExplainabilityOutput(shap_summary_fig=fig)
