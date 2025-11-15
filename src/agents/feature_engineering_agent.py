# src/agents/feature_engineering_agent.py

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline

from preprocess.tabular import build_preprocess_pipeline



@dataclass
class FeatureEngineeringResult:
    pipeline: Pipeline
    numeric_features: List[str]
    categorical_features: List[str]


class FeatureEngineeringAgent:
    def run(self, df: pd.DataFrame, target_col: str) -> FeatureEngineeringResult:
        pipeline, num_cols, cat_cols = build_preprocess_pipeline(df, target_col)
        return FeatureEngineeringResult(
            pipeline=pipeline,
            numeric_features=num_cols,
            categorical_features=cat_cols,
        )
