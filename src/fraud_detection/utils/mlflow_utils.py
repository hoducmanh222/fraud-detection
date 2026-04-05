from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def get_mlflow():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        return mlflow, MlflowClient
    except ImportError:
        return None, None


def configure_mlflow(tracking_uri: str, registry_uri: str) -> bool:
    mlflow, _ = get_mlflow()
    if mlflow is None:
        return False
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", tracking_uri))
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", registry_uri))
    return True


def start_run(experiment_name: str, run_name: str):
    mlflow, _ = get_mlflow()
    if mlflow is None:
        return nullcontext()
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any]) -> None:
    mlflow, _ = get_mlflow()
    if mlflow is None:
        return
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float]) -> None:
    mlflow, _ = get_mlflow()
    if mlflow is None:
        return
    mlflow.log_metrics(metrics)


def log_artifact(path: Path, artifact_path: str | None = None) -> None:
    mlflow, _ = get_mlflow()
    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_dict(payload: dict[str, Any], artifact_path: str) -> None:
    mlflow, _ = get_mlflow()
    if mlflow is None:
        return
    mlflow.log_dict(payload, artifact_path)


def register_model_alias(model_name: str, source_uri: str, alias: str) -> dict[str, Any]:
    mlflow, mlflow_client_cls = get_mlflow()
    if mlflow is None or mlflow_client_cls is None:
        return {"registered": False, "reason": "mlflow-unavailable"}

    try:
        result = mlflow.register_model(model_uri=source_uri, name=model_name)
        client = mlflow_client_cls()
        client.set_registered_model_alias(name=model_name, alias=alias, version=result.version)
        return {
            "registered": True,
            "model_name": model_name,
            "alias": alias,
            "version": result.version,
        }
    except Exception as exc:  # pragma: no cover - server dependent
        return {"registered": False, "reason": str(exc)}
