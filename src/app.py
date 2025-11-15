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

        # 1. Load CSV
        df = load_csv(csv_path)

        # 2. Detect target column
        target_col = infer_target_column(df)

        # 3. Detect task type (classification vs regression)
        if df[target_col].dtype in ["float64", "int64"] and df[target_col].nunique() > 20:
            task_type = "regression"
        else:
            task_type = "classification"

        # 4. Feature engineering
        fe_agent = FeatureEngineeringAgent()
        fe_out = fe_agent.run(df, target_col)

        # 5. Model training (now passes correct parameters)
        train_agent = ModelTrainingAgent(task_type=task_type)
        train_out = train_agent.run(df, fe_out.pipeline, target_col)

        # 6. Evaluation
        eval_agent = EvaluatorAgent()
        eval_out = eval_agent.run(
            train_out.y_test,
            train_out.y_pred_test,
            task_type=task_type
        )

        # 7. Explainability (works for regression too)
        exp_agent = ExplainabilityAgent()
        exp_out = exp_agent.run(train_out.best_model, train_out.X_test)

        # 8. Report generation
        rep_agent = ReportAgent()
        rep_out = rep_agent.run(
            train_out.best_model_name,
            train_out.metrics[train_out.best_model_name]
        )

        # 9. Final output dictionary
        return {
            "best_model": train_out.best_model_name,
            "metrics": train_out.metrics[train_out.best_model_name],
            "confusion_matrix_fig": getattr(eval_out, "confusion_matrix_fig", None),
            "shap_fig": exp_out.shap_summary_fig,
            "report_path": rep_out.report_path,
        }
