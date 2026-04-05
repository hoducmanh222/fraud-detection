from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fraud_detection.config import load_yaml, save_yaml
from fraud_detection.utils.mlflow_utils import register_model_alias
from fraud_detection.utils.paths import ensure_dirs, find_project_root


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_promotion() -> None:
    project_root = find_project_root()
    train_cfg = load_yaml("configs/train.yaml")
    serve_cfg = load_yaml("configs/serve.yaml")

    registry_dir = project_root / "models" / "registry"
    trained_dir = project_root / "models" / "trained"
    ensure_dirs(registry_dir, trained_dir)

    candidate_path = registry_dir / "candidate.json"
    champion_path = registry_dir / "champion.json"
    promotion_path = registry_dir / "last_promotion.json"
    drift_path = project_root / "reports" / "drift" / "drift_report.json"

    candidate = _load_json(candidate_path)
    champion = _load_json(champion_path)
    drift = _load_json(drift_path)
    if not candidate:
        raise FileNotFoundError("Candidate manifest does not exist. Run training first.")

    promotion_cfg = train_cfg.get("promotion", {})
    candidate_metrics = candidate.get("metrics", {})
    drift_status = drift.get("summary", {}).get("status", "unknown")

    reasons = []
    promote = True

    if float(candidate_metrics.get("recall", 0.0)) < float(promotion_cfg.get("min_recall", 0.75)):
        promote = False
        reasons.append("candidate recall below minimum")
    if float(candidate_metrics.get("fpr", 1.0)) > float(promotion_cfg.get("max_fpr", 0.03)):
        promote = False
        reasons.append("candidate fpr above maximum")
    if drift_status == "alert":
        promote = False
        reasons.append("drift status is alert")

    if champion:
        current_auprc = float(champion.get("metrics", {}).get("auprc", 0.0))
        candidate_auprc = float(candidate_metrics.get("auprc", 0.0))
        required_delta = float(promotion_cfg.get("min_auprc_delta", 0.002))
        if (candidate_auprc - current_auprc) < required_delta:
            promote = False
            reasons.append("candidate does not beat champion by required AUPRC delta")

    candidate_bundle = project_root / str(
        candidate.get("bundle_path", "models/trained/model_bundle.joblib")
    )
    production_bundle = trained_dir / "production_model.joblib"
    mlflow_result = {"registered": False, "reason": "promotion-not-run"}

    if promote:
        shutil.copy2(candidate_bundle, production_bundle)
        promoted_manifest = dict(candidate)
        promoted_manifest["status"] = "champion"
        promoted_manifest["promoted_at"] = datetime.now(timezone.utc).isoformat()
        promoted_manifest["bundle_path"] = str(production_bundle.relative_to(project_root))
        with open(champion_path, "w", encoding="utf-8") as handle:
            json.dump(promoted_manifest, handle, indent=2)

        mlflow = candidate.get("mlflow", {})
        if mlflow.get("enabled") and (mlflow.get("model_uri") or mlflow.get("run_id")):
            source_uri = str(mlflow.get("model_uri") or f"runs:/{mlflow['run_id']}/model")
            mlflow_result = register_model_alias(
                model_name=str(
                    train_cfg.get("experiment", {}).get("registered_model_name", "fraud-detector")
                ),
                source_uri=source_uri,
                alias="champion",
            )

        updated_serve_cfg = dict(serve_cfg)
        updated_serve_cfg.setdefault("service", {})
        updated_serve_cfg.setdefault("model", {})
        updated_serve_cfg["service"]["model_bundle_path"] = str(
            production_bundle.relative_to(project_root)
        )
        updated_serve_cfg["model"]["threshold"] = float(candidate.get("threshold", 0.5))
        updated_serve_cfg["model"]["selected_model"] = str(
            candidate.get("selected_model", "unknown")
        )
        updated_serve_cfg["model"]["version"] = str(candidate.get("version", "unknown"))
        updated_serve_cfg["model"]["trained_at"] = promoted_manifest["promoted_at"]
        save_yaml("configs/serve.yaml", updated_serve_cfg)
    elif not champion_path.exists():
        with open(champion_path, "w", encoding="utf-8") as handle:
            json.dump({"status": "none", "bundle_path": None}, handle, indent=2)

    payload = {
        "promote": promote,
        "reasons": reasons or ["candidate accepted"],
        "candidate_version": candidate.get("version"),
        "champion_version_before": champion.get("version"),
        "drift_status": drift_status,
        "mlflow": mlflow_result,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(promotion_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
