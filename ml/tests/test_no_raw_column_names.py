"""H07: only `ml/src/sources/` may know raw survey column names.

Everything downstream speaks canonical names, so swapping BRFSS for CCHS means
writing one adapter file and nothing else.
"""

import re
from pathlib import Path

from src.sources import brfss

SRC = Path(__file__).resolve().parents[1] / "src"


def test_raw_column_names_stay_inside_sources():
    offenders = []
    for path in SRC.rglob("*.py"):
        if "sources" in path.relative_to(SRC).parts:
            continue
        text = path.read_text(encoding="utf-8")
        for name in brfss.RAW_COLUMNS:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text):
                offenders.append(f"{path.relative_to(SRC)} mentions {name}")
    assert not offenders, offenders
