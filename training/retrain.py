from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def evaluate_and_get_rmse(artifact_dir: Path, data_path: str) -> float:
    """Run evaluation and extract RMSE from metadata."""
    try:
        run_command([
            sys.executable, "-m", "training.evaluate",
            "--data", data_path,
            "--artifact-dir", str(artifact_dir)
        ])
        
        metadata_path = artifact_dir / "output" / "evaluation_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata["metrics"]["RMSE"]
    except Exception as e:
        print(f"Warning: Evaluation failed at {artifact_dir}. Error: {e}")
        return float('inf')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic retraining script with automatic rollback.")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip daily data crawling before retraining.")
    parser.add_argument("--raw-data", default="data/raw/fpt_stock_price.csv", help="Path to raw FPT OHLCV CSV.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--windows", nargs="+", default=["3", "5", "7", "10", "30"], help="Window sizes for retraining.")
    parser.add_argument("--models", nargs="+", default=["lstm", "cnn1d_lstm", "transformer", "informer", "autoformer"], help="Models to retrain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    vn_tz = ZoneInfo("Asia/Ho_Chi_Minh")
    start_time = datetime.now(vn_tz)

    print("-" * 80)
    print("MONTHLY RETRAINING PROCESS STARTED")
    print(f"Time (VN): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

    if not args.skip_crawl:
        run_command([sys.executable, "-m", "data_pipeline.daily_crawl"])

    artifact_dir = ROOT_DIR / "artifacts" / "raw_fpt_only_residual_cnnlstm_transformer"
    backup_dir = ROOT_DIR / "artifacts" / "backup_raw_fpt"

    print("\nEvaluating current model...")
    old_rmse = evaluate_and_get_rmse(artifact_dir, args.raw_data)
    print(f"Current RMSE: {old_rmse:.4f}")

    print("\nCreating backup...")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(artifact_dir, backup_dir)
    print(f"Backup created at: {backup_dir.name}/")

    print("\nTraining new model...")
    run_command([
        sys.executable, "-m", "training.train",
        "--data", args.raw_data,
        "--epochs", str(args.epochs),
        "--windows", *args.windows,
        "--models", *args.models,
    ])

    print("\nEvaluating new model...")
    new_rmse = evaluate_and_get_rmse(artifact_dir, args.raw_data)
    print(f"New RMSE: {new_rmse:.4f}")

    print("\nComparison and Action:")
    if new_rmse < old_rmse:
        print(f"Result: New model is better ({new_rmse:.4f} < {old_rmse:.4f}).")
        print("Action: Keeping new model. Removing backup.")
        shutil.rmtree(backup_dir)
    else:
        print(f"Result: New model is not better ({new_rmse:.4f} >= {old_rmse:.4f}).")
        print("Action: Rolling back to previous model...")
        shutil.rmtree(artifact_dir)
        shutil.copytree(backup_dir, artifact_dir)
        shutil.rmtree(backup_dir)
        print("Rollback complete.")

    end_time = datetime.now(vn_tz)
    print("\n" + "-" * 80)
    print(f"Process finished at (VN): {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)


if __name__ == "__main__":
    main()