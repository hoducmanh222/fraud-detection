from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.config import load_yaml
from fraud_detection.utils.paths import ensure_dirs, find_project_root

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference.to_numpy(dtype=float), quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_hist, _ = np.histogram(reference, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)

    reference_ratio = np.where(
        reference_hist == 0, 1e-6, reference_hist / max(reference_hist.sum(), 1)
    )
    current_ratio = np.where(current_hist == 0, 1e-6, current_hist / max(current_hist.sum(), 1))
    return float(
        np.sum((current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio))
    )


def _categorical_diff(reference: pd.Series, current: pd.Series) -> float:
    ref_dist = reference.value_counts(normalize=True)
    cur_dist = current.value_counts(normalize=True)
    union = ref_dist.index.union(cur_dist.index)
    ref = ref_dist.reindex(union, fill_value=0.0)
    cur = cur_dist.reindex(union, fill_value=0.0)
    return float((cur - ref).abs().max())


def generate_drift_report() -> None:
    project_root = find_project_root()
    cfg = load_yaml("configs/monitoring.yaml")
    monitoring_cfg = cfg.get("monitoring", {})

    reference_path = project_root / str(
        monitoring_cfg.get("reference_path", "data/processed/reference.parquet")
    )
    current_path = project_root / str(
        monitoring_cfg.get("current_path", "data/processed/current.parquet")
    )
    report_path = project_root / str(
        monitoring_cfg.get("report_path", "reports/drift/drift_report.json")
    )
    ensure_dirs(report_path.parent)

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    report: dict[str, Any] = {
        "summary": {"status": "ok"},
        "numeric": {},
        "categorical": {},
    }

    psi_warn = float(monitoring_cfg.get("psi_warn_threshold", 0.1))
    psi_alert = float(monitoring_cfg.get("psi_alert_threshold", 0.2))
    cat_warn = float(monitoring_cfg.get("categorical_diff_warn_threshold", 0.1))
    cat_alert = float(monitoring_cfg.get("categorical_diff_alert_threshold", 0.2))

    overall_status = "ok"
    for column in NUMERIC_COLUMNS:
        score = _psi(reference_df[column], current_df[column])
        status = "ok"
        if score >= psi_alert:
            status = "alert"
        elif score >= psi_warn:
            status = "warn"
        report["numeric"][column] = {"psi": score, "status": status}
        if status == "alert":
            overall_status = "alert"
        elif status == "warn" and overall_status != "alert":
            overall_status = "warn"

    reference_df = reference_df.assign(
        orig_prefix=reference_df["nameOrig"].astype(str).str[:1],
        dest_prefix=reference_df["nameDest"].astype(str).str[:1],
    )
    current_df = current_df.assign(
        orig_prefix=current_df["nameOrig"].astype(str).str[:1],
        dest_prefix=current_df["nameDest"].astype(str).str[:1],
    )
    for column in ["type", "orig_prefix", "dest_prefix"]:
        score = _categorical_diff(reference_df[column], current_df[column])
        status = "ok"
        if score >= cat_alert:
            status = "alert"
        elif score >= cat_warn:
            status = "warn"
        report["categorical"][column] = {"max_distribution_shift": score, "status": status}
        if status == "alert":
            overall_status = "alert"
        elif status == "warn" and overall_status != "alert":
            overall_status = "warn"

    report["summary"] = {
        "status": overall_status,
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
