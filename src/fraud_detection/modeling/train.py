from __future__ import annotations

import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from fraud_detection.config import load_yaml, save_yaml
from fraud_detection.data.features import (
    RAW_FEATURE_COLUMNS,
    FraudFeatureBuilder,
    build_preprocessor,
)
from fraud_detection.utils.mlflow_utils import (
    configure_mlflow,
    get_mlflow,
    log_artifact,
    log_dict,
    log_metrics,
    log_params,
    register_model_alias,
    start_run,
)
from fraud_detection.utils.paths import ensure_dirs, find_project_root

optuna: Any = None
try:
    import optuna as _optuna
except ImportError:  # pragma: no cover - optional at import time
    pass
else:
    optuna = _optuna

LGBMClassifier: Any = None
try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
except ImportError:  # pragma: no cover - optional at import time
    pass
else:
    LGBMClassifier = _LGBMClassifier


def _compute_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    if y_true.nunique() < 2:
        y_pred = (y_score >= threshold).astype(int)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auprc = precision if int((y_true == 1).sum()) > 0 else 0.0
        return {
            "auc_roc": 0.0,
            "auprc": float(auprc),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": fpr,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _tune_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    min_precision: float,
    max_fpr: float,
    fallback_threshold: float,
) -> tuple[float, dict[str, Any]]:
    if y_true.nunique() < 2:
        return fallback_threshold, {
            "selected_threshold": fallback_threshold,
            "selected_recall": 0.0,
            "selected_precision": 0.0,
            "policy": "fallback-single-class",
            "min_precision": float(min_precision),
            "max_fpr": float(max_fpr),
        }

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return fallback_threshold, {"selected_threshold": fallback_threshold, "policy": "fallback"}

    best_threshold = fallback_threshold
    best_recall = -1.0
    best_precision = -1.0

    for index, threshold in enumerate(thresholds):
        precision = float(precisions[index])
        recall = float(recalls[index])
        metrics = _compute_metrics(y_true, y_score, float(threshold))
        fpr = float(metrics["fpr"])
        if precision < min_precision or fpr > max_fpr:
            continue
        if recall > best_recall or (recall == best_recall and precision > best_precision):
            best_threshold = float(threshold)
            best_recall = recall
            best_precision = precision

    return best_threshold, {
        "selected_threshold": float(best_threshold),
        "selected_recall": float(best_recall) if best_recall >= 0 else 0.0,
        "selected_precision": float(best_precision) if best_precision >= 0 else 0.0,
        "policy": "recall_first",
        "min_precision": float(min_precision),
        "max_fpr": float(max_fpr),
    }


def _build_logistic_pipeline(model_cfg: dict[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_builder", FraudFeatureBuilder()),
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    C=float(model_cfg.get("C", 1.0)),
                    max_iter=int(model_cfg.get("max_iter", 1200)),
                    class_weight=str(model_cfg.get("class_weight", "balanced")),
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_lightgbm_pipeline(
    model_cfg: dict[str, Any], seed: int, scale_pos_weight: float
) -> Pipeline:
    if LGBMClassifier is None:
        raise ImportError("lightgbm is required to train the champion candidate")

    params = deepcopy(model_cfg)
    return Pipeline(
        steps=[
            ("feature_builder", FraudFeatureBuilder()),
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    random_state=seed,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=int(params.get("n_estimators", 400)),
                    learning_rate=float(params.get("learning_rate", 0.05)),
                    num_leaves=int(params.get("num_leaves", 64)),
                    subsample=float(params.get("subsample", 0.9)),
                    colsample_bytree=float(params.get("colsample_bytree", 0.9)),
                    reg_alpha=float(params.get("reg_alpha", 0.0)),
                    reg_lambda=float(params.get("reg_lambda", 0.0)),
                    min_child_samples=int(params.get("min_child_samples", 30)),
                ),
            ),
        ]
    )


def _optimize_lightgbm(
    base_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    scale_pos_weight: float,
    optuna_cfg: dict[str, Any],
) -> dict[str, Any]:
    if not optuna_cfg.get("enabled", True) or optuna is None:
        return base_params

    def objective(trial: Any) -> float:
        params = deepcopy(base_params)
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 2.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 2.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            }
        )
        pipeline = _build_lightgbm_pipeline(params, seed=seed, scale_pos_weight=scale_pos_weight)
        pipeline.fit(X_train, y_train)
        y_score = pipeline.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, y_score))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=int(optuna_cfg.get("n_trials", 12)),
        timeout=int(optuna_cfg.get("timeout_seconds", 600)),
    )
    tuned = deepcopy(base_params)
    tuned.update(study.best_params)
    return tuned


def _candidate_pipelines(
    train_cfg: dict[str, Any],
    seed: int,
    scale_pos_weight: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, Pipeline]:
    models_cfg = train_cfg.get("models", {})
    candidates: dict[str, Pipeline] = {}
    enabled = set(models_cfg.get("candidates", ["logistic_regression", "lightgbm"]))

    if "logistic_regression" in enabled:
        candidates["logistic_regression"] = _build_logistic_pipeline(
            models_cfg.get("logistic_regression", {}),
            seed=seed,
        )

    if "lightgbm" in enabled:
        tuned_params = _optimize_lightgbm(
            base_params=models_cfg.get("lightgbm", {}),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            seed=seed,
            scale_pos_weight=scale_pos_weight,
            optuna_cfg=models_cfg.get("optuna", {}),
        )
        candidates["lightgbm"] = _build_lightgbm_pipeline(
            tuned_params,
            seed=seed,
            scale_pos_weight=scale_pos_weight,
        )

    return candidates


def _limit_dataset_preserve_positives(
    frame: pd.DataFrame,
    sample_rows: int,
    target_column: str,
    seed: int,
) -> pd.DataFrame:
    if len(frame) <= sample_rows:
        return frame

    positives = frame.loc[frame[target_column] == 1]
    negatives = frame.loc[frame[target_column] == 0]
    remaining = max(sample_rows - len(positives), 0)
    if remaining == 0:
        return positives.sample(min(len(positives), sample_rows), random_state=seed)

    sampled_negatives = negatives.sample(min(len(negatives), remaining), random_state=seed)
    limited = pd.concat([positives, sampled_negatives], ignore_index=True)
    return limited.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_models(sample_rows: int | None = None) -> None:
    project_root = find_project_root()
    data_cfg = load_yaml("configs/data.yaml")
    train_cfg = load_yaml("configs/train.yaml")
    serve_cfg = load_yaml("configs/serve.yaml")

    seed = int(data_cfg.get("project", {}).get("seed", 42))
    target_column = str(data_cfg.get("data", {}).get("target_column", "isFraud"))
    processed_dir = project_root / str(
        data_cfg.get("data", {}).get("processed_dir", "data/processed")
    )
    reports_dir = project_root / "reports" / "metrics"
    model_dir = project_root / "models" / "trained"
    registry_dir = project_root / "models" / "registry"
    ensure_dirs(reports_dir, model_dir, registry_dir)

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    if sample_rows:
        train_df = _limit_dataset_preserve_positives(train_df, sample_rows, target_column, seed)
        val_df = _limit_dataset_preserve_positives(
            val_df, max(100, sample_rows // 5), target_column, seed
        )
        test_df = _limit_dataset_preserve_positives(
            test_df, max(100, sample_rows // 5), target_column, seed
        )

    X_train = train_df[RAW_FEATURE_COLUMNS]
    y_train = train_df[target_column]
    X_val = val_df[RAW_FEATURE_COLUMNS]
    y_val = val_df[target_column]
    X_test = test_df[RAW_FEATURE_COLUMNS]
    y_test = test_df[target_column]

    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = negatives / positives

    experiment_cfg = train_cfg.get("experiment", {})
    tracking_uri = str(experiment_cfg.get("tracking_uri", "./mlruns"))
    registry_uri = str(experiment_cfg.get("registry_uri", tracking_uri))
    mlflow_enabled = configure_mlflow(tracking_uri=tracking_uri, registry_uri=registry_uri)

    candidates = _candidate_pipelines(
        train_cfg=train_cfg,
        seed=seed,
        scale_pos_weight=scale_pos_weight,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    if not candidates:
        raise ValueError("No candidate models are enabled")

    threshold_cfg = train_cfg.get("threshold", {})
    run_name = f"train-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    with start_run(str(experiment_cfg.get("name", "fraud-detector")), run_name) as active_run:
        log_params(
            {
                "seed": seed,
                "sample_rows": sample_rows if sample_rows is not None else "full",
                "threshold_policy": threshold_cfg.get("policy", "recall_first"),
                "scale_pos_weight": round(scale_pos_weight, 4),
            }
        )

        candidate_results: list[dict[str, Any]] = []
        best_name = ""
        best_pipeline: Pipeline | None = None
        best_threshold = float(threshold_cfg.get("fallback_threshold", 0.5))
        best_validation: dict[str, Any] | None = None

        for name, pipeline in candidates.items():
            model = clone(pipeline)
            model.fit(X_train, y_train)
            val_score = model.predict_proba(X_val)[:, 1]
            threshold, threshold_details = _tune_threshold(
                y_true=y_val,
                y_score=val_score,
                min_precision=float(threshold_cfg.get("min_precision", 0.2)),
                max_fpr=float(threshold_cfg.get("max_fpr", 0.03)),
                fallback_threshold=float(threshold_cfg.get("fallback_threshold", 0.5)),
            )
            val_metrics = _compute_metrics(y_val, val_score, threshold)
            result = {
                "model_name": name,
                "threshold": threshold,
                "validation_metrics": val_metrics,
                "threshold_details": threshold_details,
            }
            candidate_results.append(result)
            log_metrics(
                {
                    f"{name}_val_auprc": float(val_metrics["auprc"]),
                    f"{name}_val_recall": float(val_metrics["recall"]),
                    f"{name}_val_precision": float(val_metrics["precision"]),
                    f"{name}_val_fpr": float(val_metrics["fpr"]),
                }
            )

            if best_validation is None or float(val_metrics["auprc"]) > float(
                best_validation["auprc"]
            ):
                best_name = name
                best_pipeline = model
                best_threshold = threshold
                best_validation = {
                    **val_metrics,
                    "threshold": threshold,
                    "threshold_details": threshold_details,
                }

        assert best_pipeline is not None
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        X_train_full = combined_df[RAW_FEATURE_COLUMNS]
        y_train_full = combined_df[target_column]
        best_pipeline.fit(X_train_full, y_train_full)

        test_score = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = _compute_metrics(y_test, test_score, best_threshold)
        selected_at = datetime.now(timezone.utc).isoformat()
        model_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        bundle = {
            "pipeline": best_pipeline,
            "threshold": float(best_threshold),
            "metadata": {
                "selected_model": best_name,
                "version": model_version,
                "trained_at": selected_at,
                "input_columns": RAW_FEATURE_COLUMNS,
                "sample_rows": sample_rows,
                "scale_pos_weight": scale_pos_weight,
                "training_environment": {
                    "python": platform.python_version(),
                    "scikit_learn": sklearn.__version__,
                },
            },
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
            "candidate_results": candidate_results,
        }

        bundle_path = model_dir / "model_bundle.joblib"
        joblib.dump(bundle, bundle_path)

        metrics_payload = {
            "selected_model": best_name,
            "validation_metrics": best_validation,
            "test_metrics": test_metrics,
            "candidate_results": candidate_results,
            "threshold": best_threshold,
            "bundle_path": str(bundle_path.relative_to(project_root)),
            "trained_at": selected_at,
            "version": model_version,
        }

        with open(reports_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)

        candidate_manifest: dict[str, Any] = {
            "status": "candidate",
            "selected_model": best_name,
            "version": model_version,
            "threshold": best_threshold,
            "trained_at": selected_at,
            "bundle_path": str(bundle_path.relative_to(project_root)),
            "metrics": test_metrics,
            "validation_metrics": best_validation,
            "mlflow": {"enabled": mlflow_enabled},
        }

        if mlflow_enabled and active_run is not None:
            mlflow_module, _ = get_mlflow()
            run_id = active_run.info.run_id
            log_artifact(bundle_path, artifact_path="model_bundle")
            log_dict(metrics_payload, "reports/train_metrics.json")
            log_metrics(
                {
                    "test_auprc": float(test_metrics["auprc"]),
                    "test_recall": float(test_metrics["recall"]),
                    "test_precision": float(test_metrics["precision"]),
                    "test_fpr": float(test_metrics["fpr"]),
                }
            )
            model_uri = None
            if mlflow_module is not None:
                mlflow_module.sklearn.log_model(best_pipeline, artifact_path="model")
                model_uri = f"runs:/{run_id}/model"
            candidate_manifest["mlflow"].update(
                {
                    "run_id": run_id,
                    "model_uri": model_uri,
                    "registration": register_model_alias(
                        model_name=str(
                            experiment_cfg.get("registered_model_name", "fraud-detector")
                        ),
                        source_uri=model_uri or f"runs:/{run_id}/model",
                        alias="candidate",
                    ),
                }
            )

        with open(registry_dir / "candidate.json", "w", encoding="utf-8") as handle:
            json.dump(candidate_manifest, handle, indent=2)

        updated_serve_cfg = deepcopy(serve_cfg)
        updated_serve_cfg.setdefault("service", {})
        updated_serve_cfg.setdefault("model", {})
        updated_serve_cfg["service"]["candidate_bundle_path"] = str(
            bundle_path.relative_to(project_root)
        )
        updated_serve_cfg["model"]["threshold"] = float(best_threshold)
        updated_serve_cfg["model"]["selected_model"] = best_name
        updated_serve_cfg["model"]["version"] = model_version
        updated_serve_cfg["model"]["trained_at"] = selected_at
        updated_serve_cfg["model"]["input_columns"] = RAW_FEATURE_COLUMNS
        save_yaml("configs/serve.yaml", updated_serve_cfg)

    print(json.dumps(metrics_payload, indent=2))
