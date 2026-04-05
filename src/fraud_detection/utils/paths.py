from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "FRAUD_DETECTION_ROOT"


def find_project_root() -> Path:
    env_root = os.getenv(PROJECT_ROOT_ENV)
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root from pyproject.toml")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
