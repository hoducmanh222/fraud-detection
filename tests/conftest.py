from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml


def _synthetic_transactions() -> pd.DataFrame:
    rows = []
    for step in range(1, 61):
        for idx in range(18):
            amount = float(100 + (idx * 35) + step)
            rows.append(
                {
                    "step": step,
                    "type": "PAYMENT" if idx % 2 == 0 else "CASH_IN",
                    "amount": amount,
                    "nameOrig": f"C{step:03d}{idx:03d}",
                    "oldbalanceOrg": amount + 1000.0,
                    "newbalanceOrig": 1000.0,
                    "nameDest": f"M{step:03d}{idx:03d}",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": amount,
                    "isFraud": 0,
                    "isFlaggedFraud": 0,
                }
            )

        fraud_amount = 5000.0 + (step * 25.0)
        rows.append(
            {
                "step": step,
                "type": "TRANSFER",
                "amount": fraud_amount,
                "nameOrig": f"CFR{step:03d}A",
                "oldbalanceOrg": fraud_amount,
                "newbalanceOrig": 0.0,
                "nameDest": f"CFR{step:03d}B",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            }
        )
        rows.append(
            {
                "step": step,
                "type": "CASH_OUT",
                "amount": fraud_amount,
                "nameOrig": f"CFR{step:03d}C",
                "oldbalanceOrg": fraud_amount,
                "newbalanceOrig": 0.0,
                "nameDest": f"CFR{step:03d}D",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            }
        )
    return pd.DataFrame(rows)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


@pytest.fixture()
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path
    for relative in [
        "configs",
        "data/raw",
        "data/processed",
        "models/trained",
        "models/registry",
        "reports/metrics",
        "reports/drift",
    ]:
        (root / relative).mkdir(parents=True, exist_ok=True)

    dataset = _synthetic_transactions()
    csv_path = root / "data" / "raw" / "paysim.csv"
    dataset.to_csv(csv_path, index=False)
    zip_path = root / "data" / "raw" / "paysim_fraud.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="PS_20174392719_1491204439457_log.csv")
    csv_path.unlink()

    _write_yaml(
        root / "configs" / "data.yaml",
        {
            "project": {"name": "fraud-detector", "seed": 42},
            "data": {
                "raw_path": "data/raw/paysim_fraud.zip",
                "raw_csv_name": "PS_20174392719_1491204439457_log.csv",
                "processed_dir": "data/processed",
                "reports_dir": "reports/metrics",
                "target_column": "isFraud",
                "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
                "allowed_transaction_types": [
                    "PAYMENT",
                    "TRANSFER",
                    "CASH_OUT",
                    "DEBIT",
                    "CASH_IN",
                ],
            },
        },
    )
    _write_yaml(
        root / "configs" / "train.yaml",
        {
            "experiment": {
                "name": "fraud-detector-tests",
                "tracking_uri": "./mlruns",
                "registry_uri": "./mlruns",
                "registered_model_name": "fraud-detector",
            },
            "models": {
                "candidates": ["logistic_regression"],
                "logistic_regression": {"C": 1.0, "max_iter": 300, "class_weight": "balanced"},
                "lightgbm": {
                    "n_estimators": 50,
                    "learning_rate": 0.1,
                    "num_leaves": 16,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.0,
                    "reg_lambda": 0.0,
                    "min_child_samples": 10,
                },
                "optuna": {"enabled": False, "n_trials": 1, "timeout_seconds": 30},
            },
            "selection": {"primary_metric": "auprc"},
            "threshold": {
                "policy": "recall_first",
                "min_precision": 0.2,
                "max_fpr": 0.5,
                "fallback_threshold": 0.5,
            },
            "promotion": {"min_auprc_delta": 0.0, "min_recall": 0.5, "max_fpr": 0.5},
        },
    )
    _write_yaml(
        root / "configs" / "serve.yaml",
        {
            "service": {
                "model_bundle_path": "models/trained/production_model.joblib",
                "active_registry_path": "models/registry/champion.json",
            },
            "model": {
                "threshold": 0.5,
                "selected_model": "unknown",
                "version": "dev",
                "trained_at": None,
            },
        },
    )
    _write_yaml(
        root / "configs" / "monitoring.yaml",
        {
            "monitoring": {
                "reference_path": "data/processed/reference.parquet",
                "current_path": "data/processed/current.parquet",
                "report_path": "reports/drift/drift_report.json",
                "psi_warn_threshold": 0.1,
                "psi_alert_threshold": 0.2,
                "categorical_diff_warn_threshold": 0.1,
                "categorical_diff_alert_threshold": 0.2,
            }
        },
    )
    _write_json(
        root / "reports" / "drift" / "drift_report.json",
        {"summary": {"status": "ok"}, "numeric": {}, "categorical": {}},
    )

    monkeypatch.setenv("FRAUD_DETECTION_ROOT", str(root))
    monkeypatch.delenv("FRAUD_DETECTION_MODEL_PATH", raising=False)
    return root
