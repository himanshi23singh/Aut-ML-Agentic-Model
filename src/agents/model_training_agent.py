# src/agents/model_training_agent.py

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import numpy as np

# MODELS
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


@dataclass
class TrainingOutput:
    best_model: object
    best_model_name: str
    X_test: np.ndarray
    y_test: np.ndarray
    y_pred_test: np.ndarray
    metrics: dict


class ModelTrainingAgent:

    def __init__(self, task_type="auto"):
        self.task_type = task_type

    def detect_task_type(self, y):
        """Automatically detect classification or regression."""
        if y.dtype in ["float64", "int64"] and y.nunique() > 20:
            return "regression"
        else:
            return "classification"

    def build_models(self, task_type):

        if task_type == "classification":
            return {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Random Forest": RandomForestClassifier(),
                "XGBoost": xgb.XGBClassifier(),
                "LightGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0)
            }

        else:
            return {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBoost Regressor": xgb.XGBRegressor(),
                "LightGBM Regressor": LGBMRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0)
            }

    def run(self, df, pipeline, target_col):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Auto-detect classification or regression
        if self.task_type == "auto":
            task_type = self.detect_task_type(y)
        else:
            task_type = self.task_type

        # Apply preprocessing pipeline
        X_processed = pipeline.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        models = self.build_models(task_type)

        best_score = -999
        best_model = None
        best_model_name = None
        all_metrics = {}

        # Train all models
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                if task_type == "classification":
                    score = accuracy_score(y_test, preds)
                    all_metrics[name] = {
                        "accuracy": score,
                        "f1": f1_score(y_test, preds, average="weighted")
                    }

                else:  # regression
                    mse = mean_squared_error(y_test, preds)
                    score = -mse  # negative MSE for maximizing
                    all_metrics[name] = {
                        "MSE": mse,
                        "RMSE": np.sqrt(mse)
                    }

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name

            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue

        return TrainingOutput(
            best_model=best_model,
            best_model_name=best_model_name,
            X_test=X_test,
            y_test=y_test,
            y_pred_test=best_model.predict(X_test),
            metrics=all_metrics
        )
