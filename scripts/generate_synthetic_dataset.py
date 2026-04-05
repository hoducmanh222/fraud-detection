from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def build_dataset(steps: int, rows_per_step: int) -> pd.DataFrame:
    rows = []
    for step in range(1, steps + 1):
        for idx in range(rows_per_step):
            amount = float(100 + (idx * 20) + step)
            rows.append(
                {
                    "step": step,
                    "type": "PAYMENT" if idx % 2 == 0 else "CASH_IN",
                    "amount": amount,
                    "nameOrig": f"C{step:03d}{idx:03d}",
                    "oldbalanceOrg": amount + 1000.0,
                    "newbalanceOrig": 1000.0,
                    "nameDest": f"M{step:03d}{idx:03d}",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": amount,
                    "isFraud": 0,
                    "isFlaggedFraud": 0,
                }
            )

        rows.append(
            {
                "step": step,
                "type": "TRANSFER",
                "amount": 4000.0 + step,
                "nameOrig": f"CFR{step:03d}A",
                "oldbalanceOrg": 4000.0 + step,
                "newbalanceOrig": 0.0,
                "nameDest": f"CFR{step:03d}B",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            }
        )
        rows.append(
            {
                "step": step,
                "type": "CASH_OUT",
                "amount": 4000.0 + step,
                "nameOrig": f"CFR{step:03d}C",
                "oldbalanceOrg": 4000.0 + step,
                "newbalanceOrig": 0.0,
                "nameDest": f"CFR{step:03d}D",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--rows-per-step", type=int, default=12)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(steps=args.steps, rows_per_step=args.rows_per_step)
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "PS_20174392719_1491204439457_log.csv"
        dataset.to_csv(csv_path, index=False)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(csv_path, arcname=csv_path.name)

    print(f"Wrote synthetic dataset to {args.output}")


if __name__ == "__main__":
    main()
