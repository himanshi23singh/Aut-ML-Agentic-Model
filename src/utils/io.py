# src/utils/io.py

from pathlib import Path
from typing import Optional, List

import joblib
import pandas as pd

from config import MODELS_DIR, CANDIDATE_TARGET_NAMES


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_model(model, name: str) -> str:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return str(path)


def load_model(path: str):
    return joblib.load(path)


def infer_target_column(df: pd.DataFrame, candidates: Optional[List[str]] = None) -> Optional[str]:
    """Try to detect target column by name; else use last column."""
    if candidates is None:
        candidates = CANDIDATE_TARGET_NAMES

    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    return df.columns[-1] if len(df.columns) > 1 else None
