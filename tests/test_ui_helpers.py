from __future__ import annotations

from fraud_detection.ui_helpers import default_transaction_payload, parse_batch_csv


def test_parse_batch_csv() -> None:
    csv_bytes = (
        b"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n"
        b"1,TRANSFER,1000.0,C1,1000.0,0.0,C2,0.0,0.0\n"
    )
    frame = parse_batch_csv(csv_bytes)
    assert frame.shape == (1, 9)


def test_default_transaction_payload_shape() -> None:
    payload = default_transaction_payload()
    assert set(payload.keys()) == {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    }
