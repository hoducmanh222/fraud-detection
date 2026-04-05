from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fraud_detection.utils.paths import find_project_root


def load_yaml(relative_path: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    path = find_project_root() / relative_path
    if not path.exists():
        return default.copy() if default is not None else {}
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def save_yaml(relative_path: str, payload: dict[str, Any]) -> Path:
    path = find_project_root() / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path
