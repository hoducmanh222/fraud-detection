from __future__ import annotations

from fastapi.testclient import TestClient

from fraud_detection.api.main import app
from fraud_detection.api.service import clear_bundle_cache
from fraud_detection.data.pipeline import prepare_datasets
from fraud_detection.modeling.train import train_models


def test_api_predict_and_model_metadata(project_root) -> None:
    prepare_datasets()
    train_models()
    clear_bundle_cache()

    client = TestClient(app)
    payload = {
        "step": 58,
        "type": "TRANSFER",
        "amount": 7000.0,
        "nameOrig": "CFOO",
        "oldbalanceOrg": 7000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "CBAR",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }

    health = client.get("/api/v1/health")
    predict = client.post("/api/v1/predict", json=payload)
    metadata = client.get("/api/v1/model")

    assert health.status_code == 200
    assert predict.status_code == 200
    assert metadata.status_code == 200
    assert "fraud_probability" in predict.json()
