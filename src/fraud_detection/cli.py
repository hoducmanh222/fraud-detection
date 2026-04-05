from __future__ import annotations

import argparse

from fraud_detection.data.pipeline import prepare_datasets
from fraud_detection.modeling.evaluate import evaluate_model
from fraud_detection.modeling.train import train_models
from fraud_detection.monitoring.drift import generate_drift_report
from fraud_detection.monitoring.promotion import evaluate_promotion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fraud detection MLOps CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validate and split raw data")
    prepare.add_argument(
        "--sample-rows", type=int, default=None, help="Optional row cap for smoke runs"
    )

    train = subparsers.add_parser("train", help="Train and select models")
    train.add_argument(
        "--sample-rows", type=int, default=None, help="Optional row cap for smoke runs"
    )

    subparsers.add_parser("evaluate", help="Evaluate the selected model")
    subparsers.add_parser("monitor", help="Generate drift report")
    subparsers.add_parser("promote", help="Evaluate promotion gates")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_datasets(sample_rows=args.sample_rows)
    elif args.command == "train":
        train_models(sample_rows=args.sample_rows)
    elif args.command == "evaluate":
        evaluate_model()
    elif args.command == "monitor":
        generate_drift_report()
    elif args.command == "promote":
        evaluate_promotion()
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
