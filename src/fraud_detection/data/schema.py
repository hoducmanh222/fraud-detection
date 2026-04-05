from __future__ import annotations

import pandera.pandas as pa
from pandera import Check

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

RAW_DTYPES: dict[str, str] = {
    "step": "int32",
    "type": "string",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def build_raw_schema(allowed_types: list[str]) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "step": pa.Column(int, checks=[Check.ge(1)], nullable=False),
            "type": pa.Column(str, checks=[Check.isin(allowed_types)], nullable=False),
            "amount": pa.Column(float, checks=[Check.ge(0)], nullable=False),
            "nameOrig": pa.Column(str, nullable=False),
            "oldbalanceOrg": pa.Column(float, checks=[Check.ge(0)], nullable=False),
            "newbalanceOrig": pa.Column(float, checks=[Check.ge(0)], nullable=False),
            "nameDest": pa.Column(str, nullable=False),
            "oldbalanceDest": pa.Column(float, checks=[Check.ge(0)], nullable=False),
            "newbalanceDest": pa.Column(float, checks=[Check.ge(0)], nullable=False),
            "isFraud": pa.Column(int, checks=[Check.isin([0, 1])], nullable=False),
            "isFlaggedFraud": pa.Column(int, checks=[Check.isin([0, 1])], nullable=False),
        },
        strict=True,
        coerce=True,
    )
