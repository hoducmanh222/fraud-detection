from __future__ import annotations

import json

from fraud_detection.data.pipeline import prepare_datasets
from fraud_detection.modeling.train import train_models
from fraud_detection.monitoring.promotion import evaluate_promotion


def test_promotion_writes_decision(project_root) -> None:
    prepare_datasets()
    train_models()
    evaluate_promotion()

    with open(
        project_root / "models" / "registry" / "last_promotion.json", encoding="utf-8"
    ) as handle:
        payload = json.load(handle)

    assert "promote" in payload
    assert payload["reasons"]
