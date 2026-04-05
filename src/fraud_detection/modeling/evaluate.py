from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay

from fraud_detection.config import load_yaml
from fraud_detection.data.features import RAW_FEATURE_COLUMNS
from fraud_detection.modeling.train import _compute_metrics
from fraud_detection.utils.paths import ensure_dirs, find_project_root


def evaluate_model() -> None:
    project_root = find_project_root()
    data_cfg = load_yaml("configs/data.yaml")
    serve_cfg = load_yaml("configs/serve.yaml")

    processed_dir = project_root / str(
        data_cfg.get("data", {}).get("processed_dir", "data/processed")
    )
    bundle_path = project_root / str(
        serve_cfg.get("service", {}).get(
            "candidate_bundle_path", "models/trained/model_bundle.joblib"
        )
    )
    if not bundle_path.exists():
        bundle_path = project_root / "models" / "trained" / "model_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle does not exist at {bundle_path}")

    bundle = joblib.load(bundle_path)
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    X_test = test_df[RAW_FEATURE_COLUMNS]
    y_test = test_df[str(data_cfg.get("data", {}).get("target_column", "isFraud"))]
    y_score = pipeline.predict_proba(X_test)[:, 1]
    metrics = _compute_metrics(y_test, y_score, threshold)

    figures_dir = project_root / "reports" / "figures"
    metrics_dir = project_root / "reports" / "metrics"
    ensure_dirs(figures_dir, metrics_dir)

    figure_path = figures_dir / "pr_curve.png"
    if y_test.nunique() < 2:
        figure = plt.figure(figsize=(6, 4))
        axis = figure.add_subplot(1, 1, 1)
        axis.text(
            0.5, 0.5, "PR curve unavailable for single-class holdout", ha="center", va="center"
        )
        axis.set_axis_off()
        figure.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    else:
        display = PrecisionRecallDisplay.from_predictions(y_test, y_score)
        display.ax_.set_title("Precision-Recall Curve")
        display.figure_.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(display.figure_)

    payload = {
        "bundle_path": str(bundle_path.relative_to(project_root)),
        "threshold": threshold,
        "metrics": metrics,
        "metadata": bundle.get("metadata", {}),
    }
    with open(metrics_dir / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
