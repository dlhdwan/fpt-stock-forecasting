from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import socket
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from training.metrics import calculate_metrics
from training.predict import (
    choose_best_checkpoint,
    load_model_bundle,
    predict_from_dataframe,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT_DIR / path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained FPT stock forecasting model on latest data."
    )
    parser.add_argument("--data", default="data/raw/fpt_stock_price.csv", help="Path to input CSV data.")
    parser.add_argument("--artifact-dir", default="artifacts/raw_fpt_only_residual_cnnlstm_transformer", help="Path to trained model artifact folder.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint filename. If not provided, best checkpoint is selected automatically.")
    parser.add_argument("--max-rows", type=int, default=300, help="Number of latest prediction rows to evaluate.")
    parser.add_argument("--output-name", default="weekly_evaluation.csv", help="Output CSV filename.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    data_path = resolve_project_path(args.data)
    artifact_dir = resolve_project_path(args.artifact_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact folder not found: {artifact_dir}")

    # --- TẠO THƯ MỤC OUTPUT ---
    output_dir = artifact_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint is None:
        checkpoint_path = choose_best_checkpoint(artifact_dir)
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = artifact_dir / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    vn_tz = ZoneInfo("Asia/Ho_Chi_Minh")
    current_time_vn = datetime.now(vn_tz)

    print("=" * 100)
    print("FPT MODEL WEEKLY EVALUATION")
    print(f"Time (VN): {current_time_vn.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("Data:", data_path)
    print("Artifact:", artifact_dir)
    print("Output Dir:", output_dir)
    print("Checkpoint:", checkpoint_path.name)
    print("Device:", device)
    print("=" * 100)

    bundle = load_model_bundle(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    df = pd.read_csv(data_path)

    result_df = predict_from_dataframe(
        df_input=df,
        bundle=bundle,
        device=device,
        max_rows=args.max_rows,
    )

    eval_df = result_df.dropna(
        subset=["actual_next_close", "predicted_next_close", "last_close"]
    ).copy()

    if eval_df.empty:
        raise ValueError("Không có dòng nào có actual_next_close để evaluate.")

    metrics = calculate_metrics(
        y_true=eval_df["actual_next_close"].to_numpy(),
        y_pred=eval_df["predicted_next_close"].to_numpy(),
        last_close=eval_df["last_close"].to_numpy(),
    )

    output_row = {
        "evaluated_at": current_time_vn.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(data_path),
        "artifact_dir": str(artifact_dir),
        "checkpoint": checkpoint_path.name,
        "model": bundle["model_name"],
        "window": bundle["window_size"],
        "num_eval_rows": int(len(eval_df)),
        "start_date": str(pd.to_datetime(eval_df["predict_for_date"]).min().date()),
        "end_date": str(pd.to_datetime(eval_df["predict_for_date"]).max().date()),
        **metrics,
    }

    output_df = pd.DataFrame([output_row])

    # Lưu file vào thư mục output_dir
    output_path = output_dir / args.output_name
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    history_path = output_dir / "evaluation_history.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, output_df], ignore_index=True)
    else:
        history_df = output_df
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    predictions_path = output_dir / "weekly_predictions.csv"
    eval_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    metadata = {
        "evaluation_time": current_time_vn.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "Asia/Ho_Chi_Minh",
        "status": "success",
        "evaluator": "github-actions",
        "host": socket.gethostname(),
        "model_info": {
            "checkpoint": checkpoint_path.name,
            "model_name": bundle["model_name"]
        },
        "evaluation_window": {
            "start": output_row["start_date"],
            "end": output_row["end_date"],
            "rows": output_row["num_eval_rows"]
        },
        "metrics": metrics
    }
    
    metadata_path = output_dir / "evaluation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\nEvaluation metrics:")
    print(json.dumps(output_row, indent=4, ensure_ascii=False))
    print("\nSaved files to:", output_dir)

if __name__ == "__main__":
    main()