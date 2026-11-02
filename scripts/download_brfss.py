"""Download a CDC BRFSS annual file (SAS transport, .XPT) into data/raw/.

The raw file is ~760 MB, too large for git, so everyone fetches the same official copy:

    python scripts/download_brfss.py              # 2025 file -> data/raw/LLCP2025.XPT
    python scripts/download_brfss.py --year 2024
    python scripts/download_brfss.py --force      # re-download even if the file exists

Safe to re-run: skips the download when the .XPT is already there.
"""

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"
CHUNK = 1024 * 1024


def download(url: str, dest: Path) -> None:
    """Stream url to dest, printing progress as it goes."""
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / CHUNK:.0f} / {total / CHUNK:.0f} MB", end="", flush=True)
    print()


def extract_xpt(zip_path: Path, target: Path) -> None:
    """Pull the single .XPT out of the CDC zip and save it as target."""
    with zipfile.ZipFile(zip_path) as zf:
        # CDC names the member with a trailing space ("LLCP2025.XPT "), so match loosely.
        members = [m for m in zf.namelist() if m.strip().upper().endswith(".XPT")]
        if len(members) != 1:
            raise SystemExit(f"Expected one .XPT in the zip, found: {zf.namelist()}")
        partial = target.with_suffix(".XPT.part")
        with zf.open(members[0]) as src, partial.open("wb") as out:
            shutil.copyfileobj(src, out, CHUNK)
        partial.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2025, help="BRFSS survey year (default 2025)")
    parser.add_argument("--force", action="store_true", help="re-download if the file exists")
    args = parser.parse_args()

    target = RAW_DIR / f"LLCP{args.year}.XPT"
    if target.exists() and not args.force:
        print(
            f"{target} already exists ({target.stat().st_size / CHUNK:.0f} MB). "
            "Use --force to re-download."
        )
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = URL.format(year=args.year)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "brfss.zip"
        print(f"Downloading {url}")
        download(url, zip_path)
        print(f"Extracting to {target}")
        extract_xpt(zip_path, target)

    print(f"Done: {target} ({target.stat().st_size / CHUNK:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
