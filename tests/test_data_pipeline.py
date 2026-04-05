from __future__ import annotations

import pandas as pd

from fraud_detection.data.pipeline import prepare_datasets


def test_prepare_datasets_temporal_split(project_root) -> None:
    prepare_datasets()

    train_df = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")
    val_df = pd.read_parquet(project_root / "data" / "processed" / "val.parquet")
    test_df = pd.read_parquet(project_root / "data" / "processed" / "test.parquet")

    assert train_df["step"].max() < val_df["step"].min()
    assert val_df["step"].max() < test_df["step"].min()
    assert train_df["isFraud"].sum() > 0
    assert val_df["isFraud"].sum() > 0
    assert test_df["isFraud"].sum() > 0
