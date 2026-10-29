"""One place for paths and constants, so no other module hard-codes them."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

# CDC BRFSS 2025 annual file (SAS transport format). Download steps are in the README.
RAW_BRFSS_FILE = DATA_RAW / "LLCP2025.XPT"

RANDOM_SEED = 42
