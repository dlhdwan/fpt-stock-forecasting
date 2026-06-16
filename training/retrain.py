from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print("=" * 100)
    print("Running:", " ".join(command))
    print("=" * 100)

    subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodic retraining script for FPT stock forecasting models."
    )

    parser.add_argument(
        "--mode",
        choices=["raw", "merged", "both"],
        default="raw",
        help="Retrain raw model, merged model, or both.",
    )

    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip daily data crawling before retraining.",
    )

    parser.add_argument(
        "--raw-data",
        default="data/raw/fpt_stock_price.csv",
        help="Path to raw FPT OHLCV CSV.",
    )

    parser.add_argument(
        "--merged-data",
        default="data/processed/merged_dataset.csv",
        help="Path to merged dataset CSV.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--windows",
        nargs="+",
        default=["3", "5", "7", "10", "30"],
        help="Window sizes for retraining.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "cnn1d_lstm", "transformer", "informer", "autoformer"],
        help="Models to retrain.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("FPT STOCK MODEL RETRAINING")
    print("Started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Mode:", args.mode)
    print("=" * 100)

    if not args.skip_crawl:
        run_command(
            [
                sys.executable,
                "-m",
                "data_pipeline.daily_crawl",
            ]
        )

    common_args = [
        "--epochs",
        str(args.epochs),
        "--windows",
        *args.windows,
        "--models",
        *args.models,
    ]

    if args.mode in ["raw", "both"]:
        run_command(
            [
                sys.executable,
                "-m",
                "training.train",
                "--mode",
                "raw",
                "--data",
                args.raw_data,
                *common_args,
            ]
        )

    if args.mode in ["merged", "both"]:
        run_command(
            [
                sys.executable,
                "-m",
                "training.train",
                "--mode",
                "merged",
                "--data",
                args.merged_data,
                *common_args,
            ]
        )

    print("=" * 100)
    print("Retraining finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 100)


if __name__ == "__main__":
    main()