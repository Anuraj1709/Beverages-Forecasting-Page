from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_CONFIG
from .pipeline import train_forecasting_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Train beverage sales forecasting models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", default="Forecasting Case- Study.xlsx", help="Path to Excel workbook.")
    train_parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for trained artifacts.")
    train_parser.add_argument("--horizon", type=int, default=8, help="Forecast horizon in weeks.")
    train_parser.add_argument("--validation-size", type=int, default=8, help="Validation size in weeks.")
    train_parser.add_argument("--states", nargs="*", help="Optional subset of states.")
    train_parser.add_argument(
        "--models",
        nargs="*",
        choices=["sarima", "prophet", "xgboost", "lstm"],
        help="Optional subset of models.",
    )

    args = parser.parse_args()
    if args.command == "train":
        config = replace(
            DEFAULT_CONFIG,
            horizon=args.horizon,
            validation_size=args.validation_size,
            artifacts_dir=Path(args.artifacts_dir),
        )
        run = train_forecasting_system(args.data, config, states=args.states, model_names=args.models)
        print(f"Trained {len(run.selections)} states. Manifest: {config.artifacts_dir / 'manifest.json'}")
        for selection in run.selections:
            metrics = selection.metrics
            print(
                f"{selection.state}: best={selection.best_model_name} "
                f"mape={metrics['mape']:.2f} rmse={metrics['rmse']:.2f} mae={metrics['mae']:.2f}"
            )


if __name__ == "__main__":
    main()
