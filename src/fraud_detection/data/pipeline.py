from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.config import load_yaml
from fraud_detection.data.schema import RAW_COLUMNS, RAW_DTYPES, build_raw_schema
from fraud_detection.utils.paths import ensure_dirs, find_project_root


def read_raw_dataset(raw_path: Path, csv_name: str, sample_rows: int | None = None) -> pd.DataFrame:
    read_kwargs: dict[str, Any] = {"dtype": RAW_DTYPES, "usecols": RAW_COLUMNS}
    if sample_rows is not None:
        read_kwargs["nrows"] = sample_rows
    with zipfile.ZipFile(raw_path) as archive:
        with archive.open(csv_name) as handle:
            return pd.read_csv(handle, **read_kwargs)


def split_by_step(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_ratio + val_ratio + test_ratio
    if round(total, 8) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    ordered_steps = sorted(frame["step"].unique().tolist())
    step_count = len(ordered_steps)
    train_end = max(1, int(step_count * train_ratio))
    val_end = max(train_end + 1, int(step_count * (train_ratio + val_ratio)))

    train_steps = set(ordered_steps[:train_end])
    val_steps = set(ordered_steps[train_end:val_end])
    test_steps = set(ordered_steps[val_end:])

    train_df = frame.loc[frame["step"].isin(train_steps)].reset_index(drop=True)
    val_df = frame.loc[frame["step"].isin(val_steps)].reset_index(drop=True)
    test_df = frame.loc[frame["step"].isin(test_steps)].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Temporal split produced an empty dataset split")
    return train_df, val_df, test_df


def prepare_datasets(sample_rows: int | None = None) -> None:
    project_root = find_project_root()
    cfg = load_yaml("configs/data.yaml")
    data_cfg = cfg.get("data", {})

    raw_path = project_root / str(data_cfg.get("raw_path", "data/raw/paysim_fraud.zip"))
    csv_name = str(data_cfg.get("raw_csv_name", "PS_20174392719_1491204439457_log.csv"))
    processed_dir = project_root / str(data_cfg.get("processed_dir", "data/processed"))
    reports_dir = project_root / str(data_cfg.get("reports_dir", "reports/metrics"))

    ensure_dirs(processed_dir, reports_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset does not exist at {raw_path}")

    effective_sample_rows = sample_rows or data_cfg.get("sample_rows")
    df = read_raw_dataset(
        raw_path=raw_path,
        csv_name=csv_name,
        sample_rows=int(effective_sample_rows) if effective_sample_rows else None,
    )

    schema = build_raw_schema(data_cfg.get("allowed_transaction_types", []))
    validated = schema.validate(df)

    split_cfg = data_cfg.get("split", {})
    train_df, val_df, test_df = split_by_step(
        validated,
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
    )

    reference_df = val_df.copy().reset_index(drop=True)
    current_df = test_df.copy().reset_index(drop=True)

    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)
    reference_df.to_parquet(processed_dir / "reference.parquet", index=False)
    current_df.to_parquet(processed_dir / "current.parquet", index=False)

    summary = {
        "row_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "reference": int(len(reference_df)),
            "current": int(len(current_df)),
        },
        "fraud_rate": {
            "train": float(train_df["isFraud"].mean()),
            "val": float(val_df["isFraud"].mean()),
            "test": float(test_df["isFraud"].mean()),
        },
        "step_ranges": {
            "train": [int(train_df["step"].min()), int(train_df["step"].max())],
            "val": [int(val_df["step"].min()), int(val_df["step"].max())],
            "test": [int(test_df["step"].min()), int(test_df["step"].max())],
        },
        "sample_rows": int(effective_sample_rows) if effective_sample_rows else None,
    }

    quality = {
        "columns": list(validated.columns),
        "dtypes": {column: str(dtype) for column, dtype in validated.dtypes.items()},
        "target_distribution": validated["isFraud"].value_counts(normalize=True).to_dict(),
        "transaction_type_distribution": validated["type"].value_counts(normalize=True).to_dict(),
        "row_count": int(len(validated)),
    }

    with open(reports_dir / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(reports_dir / "data_quality.json", "w", encoding="utf-8") as handle:
        json.dump(quality, handle, indent=2)

    print(json.dumps(summary, indent=2))
