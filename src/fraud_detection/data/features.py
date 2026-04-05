from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
]

CATEGORICAL_ENGINEERED_COLUMNS = ["type", "orig_prefix", "dest_prefix"]
NUMERIC_ENGINEERED_COLUMNS = [
    "step",
    "day_index",
    "hour_index",
    "amount",
    "amount_log1p",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_delta",
    "dest_balance_delta",
    "orig_balance_gap",
    "dest_balance_gap",
    "orig_zero_before",
    "orig_zero_after",
    "dest_zero_before",
    "dest_zero_after",
    "dest_is_merchant",
    "is_transfer",
    "is_cash_out",
    "is_payment",
    "is_cash_in",
    "is_debit",
    "orig_txn_count",
    "dest_txn_count",
    "orig_seen_before",
    "dest_seen_before",
]


@dataclass
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


def get_feature_spec() -> FeatureSpec:
    return FeatureSpec(
        numeric=list(NUMERIC_ENGINEERED_COLUMNS),
        categorical=list(CATEGORICAL_ENGINEERED_COLUMNS),
    )


class FraudFeatureBuilder(BaseEstimator, TransformerMixin):
    """Feature engineering shared by training and inference."""

    def __init__(self) -> None:
        self.origin_counts_: dict[str, int] = {}
        self.destination_counts_: dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: Iterable[int] | None = None) -> FraudFeatureBuilder:
        frame = self._coerce_frame(X)
        self.origin_counts_ = frame["nameOrig"].value_counts().to_dict()
        self.destination_counts_ = frame["nameDest"].value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._coerce_frame(X).copy()

        engineered = pd.DataFrame(index=frame.index)
        engineered["step"] = frame["step"].astype(float)
        engineered["day_index"] = (frame["step"] // 24).astype(float)
        engineered["hour_index"] = (frame["step"] % 24).astype(float)
        engineered["type"] = frame["type"].astype(str)
        engineered["amount"] = frame["amount"].astype(float)
        engineered["amount_log1p"] = np.log1p(frame["amount"].astype(float))
        engineered["oldbalanceOrg"] = frame["oldbalanceOrg"].astype(float)
        engineered["newbalanceOrig"] = frame["newbalanceOrig"].astype(float)
        engineered["oldbalanceDest"] = frame["oldbalanceDest"].astype(float)
        engineered["newbalanceDest"] = frame["newbalanceDest"].astype(float)
        engineered["orig_balance_delta"] = frame["oldbalanceOrg"] - frame["newbalanceOrig"]
        engineered["dest_balance_delta"] = frame["newbalanceDest"] - frame["oldbalanceDest"]
        engineered["orig_balance_gap"] = (engineered["orig_balance_delta"] - frame["amount"]).abs()
        engineered["dest_balance_gap"] = (engineered["dest_balance_delta"] - frame["amount"]).abs()
        engineered["orig_zero_before"] = (frame["oldbalanceOrg"] == 0).astype(float)
        engineered["orig_zero_after"] = (frame["newbalanceOrig"] == 0).astype(float)
        engineered["dest_zero_before"] = (frame["oldbalanceDest"] == 0).astype(float)
        engineered["dest_zero_after"] = (frame["newbalanceDest"] == 0).astype(float)
        engineered["is_transfer"] = (frame["type"] == "TRANSFER").astype(float)
        engineered["is_cash_out"] = (frame["type"] == "CASH_OUT").astype(float)
        engineered["is_payment"] = (frame["type"] == "PAYMENT").astype(float)
        engineered["is_cash_in"] = (frame["type"] == "CASH_IN").astype(float)
        engineered["is_debit"] = (frame["type"] == "DEBIT").astype(float)
        engineered["orig_prefix"] = frame["nameOrig"].str[:1].fillna("U")
        engineered["dest_prefix"] = frame["nameDest"].str[:1].fillna("U")
        engineered["dest_is_merchant"] = (engineered["dest_prefix"] == "M").astype(float)
        engineered["orig_txn_count"] = (
            frame["nameOrig"].map(self.origin_counts_).fillna(0).astype(float)
        )
        engineered["dest_txn_count"] = (
            frame["nameDest"].map(self.destination_counts_).fillna(0).astype(float)
        )
        engineered["orig_seen_before"] = (engineered["orig_txn_count"] > 0).astype(float)
        engineered["dest_seen_before"] = (engineered["dest_txn_count"] > 0).astype(float)

        return engineered[CATEGORICAL_ENGINEERED_COLUMNS + NUMERIC_ENGINEERED_COLUMNS]

    def _coerce_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        missing_columns = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        return frame[RAW_FEATURE_COLUMNS]


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    feature_spec = get_feature_spec()
    numeric_transformer: str | Pipeline
    if scale_numeric:
        numeric_transformer = Pipeline([("scaler", StandardScaler())])
    else:
        numeric_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_spec.categorical),
            ("numeric", numeric_transformer, feature_spec.numeric),
        ],
        sparse_threshold=0.3,
    )
