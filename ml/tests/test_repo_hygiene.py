"""Guards the rule that survey microdata and model artifacts never enter git.

These run in CI on every ML change; the `hygiene` CI job enforces the same rule
on tracked files. Here we check the .gitignore itself so the protection can't be
silently deleted.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GITIGNORE = REPO_ROOT / ".gitignore"


@pytest.fixture(scope="module")
def ignore_rules() -> set[str]:
    lines = GITIGNORE.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


@pytest.mark.parametrize("rule", ["data/raw/*", "ml/models/*", ".env", "*.joblib"])
def test_gitignore_protects_data_models_and_secrets(ignore_rules: set[str], rule: str) -> None:
    assert rule in ignore_rules, f".gitignore must contain {rule!r}"


@pytest.mark.parametrize("folder", ["data/raw", "data/processed", "data/codebooks", "ml/models"])
def test_expected_folders_exist(folder: str) -> None:
    assert (REPO_ROOT / folder).is_dir(), f"missing folder {folder}"
