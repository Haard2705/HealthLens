"""Write ``backend/openapi.json`` from the app itself, so the committed schema cannot drift.

    python -m backend.scripts.export_openapi            # rewrite it
    python -m backend.scripts.export_openapi --check    # fail if it is out of date (CI)

The app is built with fixed settings and no artifacts, so the schema never depends on what
happens to be on the machine that exported it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.config import Settings
from backend.app.main import create_app

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "openapi.json"


def schema() -> dict:
    settings = Settings(env="test", model_dir=Path("ml/models"), _env_file=None)
    return create_app(settings).openapi()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the OpenAPI schema.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args(argv)

    text = json.dumps(schema(), indent=2) + "\n"
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current == text:
            return 0
        sys.stderr.write(f"{args.out} is out of date; re-run this script without --check\n")
        return 1
    args.out.write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.write(f"wrote {args.out}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
