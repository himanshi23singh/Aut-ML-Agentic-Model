# src/models/regression.py

from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_regression_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest_reg": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost_reg": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        ),
        "lightgbm_reg": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        ),
    }
