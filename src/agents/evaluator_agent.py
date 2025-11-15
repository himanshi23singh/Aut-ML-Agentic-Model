# src/agents/evaluator_agent.py

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.metrics import is_classification_target


@dataclass
class EvaluationOutput:
    confusion_matrix_fig: plt.Figure | None


class EvaluatorAgent:
    def run(self, y_test, y_pred):
        fig = None

        if is_classification_target(y_test):
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            plt.title("Confusion Matrix")

        return EvaluationOutput(confusion_matrix_fig=fig)
