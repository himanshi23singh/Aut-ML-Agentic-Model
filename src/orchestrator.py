# src/orchestrator.py

import pandas as pd

from utils.io import load_csv, infer_target_column
from agents.model_training_agent import ModelTrainingAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.report_agent import ReportAgent


class Orchestrator:

    def run(self, csv_path: str):

        df = load_csv(csv_path)
        target_col = infer_target_column(df)

        # Feature engineering
        fe_agent = FeatureEngineeringAgent()
        fe_out = fe_agent.run(df, target_col)

        # MODEL TRAINING (AUTO)
        train_agent = ModelTrainingAgent(task_type="auto")
        train_out = train_agent.run(df, fe_out.pipeline, target_col)

        # Evaluation
        eval_agent = EvaluatorAgent()
        eval_out = eval_agent.run(
            train_out.y_test, train_out.y_pred_test
        )

        # Explainability
        exp_agent = ExplainabilityAgent()
        exp_out = exp_agent.run(train_out.best_model, train_out.X_test)

        # Report
        rep_agent = ReportAgent()
        rep_out = rep_agent.run(
            train_out.best_model_name,
            train_out.metrics[train_out.best_model_name]
        )

        return {
            "best_model": train_out.best_model_name,
            "metrics": train_out.metrics[train_out.best_model_name],
            "confusion_matrix_fig": getattr(eval_out, "confusion_matrix_fig", None),
            "shap_fig": exp_out.shap_summary_fig,
            "report_path": rep_out.report_path,
        }
