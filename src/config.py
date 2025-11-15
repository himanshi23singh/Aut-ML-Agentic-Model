# src/config.py

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EXPLAIN_DIR = ARTIFACTS_DIR / "explanations"

for d in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, EXPLAIN_DIR]:
    os.makedirs(d, exist_ok=True)

# Candidate column names for auto target detection
CANDIDATE_TARGET_NAMES = ["target", "label", "class", "y", "outcome"]

# Global config
RANDOM_STATE = 42
CV_FOLDS = 5
