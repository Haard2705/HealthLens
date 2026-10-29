"""One place for every path and constant the ML pipeline uses.

Other modules import from here instead of hard-coding strings, so moving a folder
or changing the seed is a one-line change.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CODEBOOKS_DIR = DATA_DIR / "codebooks"

MODELS_DIR = REPO_ROOT / "ml" / "models"
REPORTS_DIR = REPO_ROOT / "docs" / "figures"

# Every random step (split, models, SHAP sampling) reads this one value.
SEED = 42

# File types load_data.py knows how to read.
RAW_EXTENSIONS = (".csv", ".xpt", ".parquet")


def raw_file_override() -> Path | None:
    """Optional explicit raw file, set with the HEALTHLENS_RAW_FILE environment variable."""
    value = os.environ.get("HEALTHLENS_RAW_FILE")
    return Path(value) if value else None
