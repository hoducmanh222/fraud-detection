from __future__ import annotations

import pandas as pd

from fraud_detection.data.features import RAW_FEATURE_COLUMNS, FraudFeatureBuilder, get_feature_spec


def test_feature_builder_outputs_expected_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 1000.0,
                "nameOrig": "C123",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "M123",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            },
            {
                "step": 2,
                "type": "PAYMENT",
                "amount": 10.0,
                "nameOrig": "C123",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 90.0,
                "nameDest": "M111",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 10.0,
            },
        ]
    )
    builder = FraudFeatureBuilder().fit(frame)
    transformed = builder.transform(frame)
    feature_spec = get_feature_spec()

    assert list(frame.columns) == RAW_FEATURE_COLUMNS
    assert transformed.shape[0] == 2
    assert set(feature_spec.numeric).issubset(set(transformed.columns))
    assert set(feature_spec.categorical).issubset(set(transformed.columns))
    assert "nameOrig" not in transformed.columns
    assert transformed["orig_txn_count"].iloc[0] >= 1
