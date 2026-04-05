from __future__ import annotations

import joblib
import yaml

from fraud_detection.modeling.train import LGBMClassifier, train_models


def test_train_models_writes_bundle(project_root) -> None:
    from fraud_detection.data.pipeline import prepare_datasets

    prepare_datasets()
    train_models()

    bundle = joblib.load(project_root / "models" / "trained" / "model_bundle.joblib")
    assert bundle["metadata"]["selected_model"] in {"logistic_regression", "lightgbm"}
    assert 0.0 <= float(bundle["threshold"]) <= 1.0
    assert bundle["metadata"]["training_environment"]["scikit_learn"]


def test_lightgbm_candidate_training_if_available(project_root) -> None:
    if LGBMClassifier is None:
        return

    from fraud_detection.data.pipeline import prepare_datasets

    prepare_datasets()
    train_cfg_path = project_root / "configs" / "train.yaml"
    with open(train_cfg_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["models"]["candidates"] = ["lightgbm"]
    config["models"]["optuna"]["enabled"] = False
    with open(train_cfg_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train_models()
    bundle = joblib.load(project_root / "models" / "trained" / "model_bundle.joblib")
    assert bundle["metadata"]["selected_model"] == "lightgbm"
