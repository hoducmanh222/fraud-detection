from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_detection.config import load_yaml
from fraud_detection.data.features import RAW_FEATURE_COLUMNS
from fraud_detection.utils.paths import find_project_root


class ModelLoadError(RuntimeError):
    """Raised when the persisted model bundle cannot be deserialized."""


def _resolve_bundle_path() -> Path:
    project_root = find_project_root()
    serve_cfg = load_yaml("configs/serve.yaml")
    env_path = os.getenv("FRAUD_DETECTION_MODEL_PATH")
    if env_path:
        return Path(env_path)

    active_registry = project_root / str(
        serve_cfg.get("service", {}).get("active_registry_path", "models/registry/champion.json")
    )
    if active_registry.exists():
        with open(active_registry, encoding="utf-8") as handle:
            registry_payload = json.load(handle)
        bundle_path = registry_payload.get("bundle_path")
        registry_bundle = project_root / str(bundle_path) if bundle_path else None
        if registry_bundle is not None and registry_bundle.is_file():
            return registry_bundle

    configured = project_root / str(
        serve_cfg.get("service", {}).get(
            "model_bundle_path", "models/trained/production_model.joblib"
        )
    )
    if configured.is_file():
        return configured

    return project_root / "models" / "trained" / "model_bundle.joblib"


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    bundle_path = _resolve_bundle_path()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle does not exist at {bundle_path}")
    try:
        return joblib.load(bundle_path)
    except Exception as exc:  # pragma: no cover - depends on runtime/model compatibility
        raise ModelLoadError(f"Failed to load model bundle at {bundle_path}: {exc}") from exc


def clear_bundle_cache() -> None:
    load_bundle.cache_clear()


def predict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bundle = load_bundle()
    frame = pd.DataFrame(records)
    missing_columns = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    frame = frame[RAW_FEATURE_COLUMNS]
    probabilities = bundle["pipeline"].predict_proba(frame)[:, 1]
    threshold = float(bundle["threshold"])
    metadata = bundle.get("metadata", {})

    return [
        {
            "fraud_probability": float(probability),
            "fraud_prediction": int(probability >= threshold),
            "threshold": threshold,
            "model_version": str(metadata.get("version", "unknown")),
            "selected_model": str(metadata.get("selected_model", "unknown")),
        }
        for probability in probabilities
    ]


def health_status() -> dict[str, Any]:
    bundle_path = _resolve_bundle_path()
    if not bundle_path.exists():
        return {
            "status": "degraded",
            "model_ready": False,
            "model_path": str(bundle_path),
        }

    try:
        load_bundle()
        return {
            "status": "ok",
            "model_ready": True,
            "model_path": str(bundle_path),
        }
    except ModelLoadError as exc:
        return {
            "status": "degraded",
            "model_ready": False,
            "model_path": str(bundle_path),
            "detail": str(exc),
        }


def model_metadata() -> dict[str, Any]:
    bundle = load_bundle()
    metadata = bundle.get("metadata", {})
    return {
        "version": str(metadata.get("version", "unknown")),
        "selected_model": str(metadata.get("selected_model", "unknown")),
        "threshold": float(bundle.get("threshold", 0.5)),
        "trained_at": metadata.get("trained_at"),
        "validation_metrics": bundle.get("validation_metrics", {}),
        "test_metrics": bundle.get("test_metrics", {}),
    }


def latest_drift_report() -> dict[str, Any]:
    project_root = find_project_root()
    report_path = project_root / "reports" / "drift" / "drift_report.json"
    if not report_path.exists():
        return {"summary": {"status": "missing"}, "numeric": {}, "categorical": {}}
    with open(report_path, encoding="utf-8") as handle:
        return json.load(handle)
