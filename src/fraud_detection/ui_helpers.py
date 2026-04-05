from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

from fraud_detection.data.features import RAW_FEATURE_COLUMNS
from fraud_detection.utils.paths import find_project_root


def default_transaction_payload() -> dict[str, Any]:
    return {
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
    }


def parse_batch_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(uploaded_bytes))
    missing = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing}")
    return frame[RAW_FEATURE_COLUMNS]


def load_local_status() -> dict[str, Any]:
    project_root = find_project_root()
    paths = {
        "candidate": project_root / "models" / "registry" / "candidate.json",
        "champion": project_root / "models" / "registry" / "champion.json",
        "promotion": project_root / "models" / "registry" / "last_promotion.json",
        "drift": project_root / "reports" / "drift" / "drift_report.json",
        "training": project_root / "reports" / "metrics" / "train_metrics.json",
    }
    payload: dict[str, Any] = {}
    for key, path in paths.items():
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                payload[key] = json.load(handle)
        else:
            payload[key] = {}
    return payload
