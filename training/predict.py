from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from training.config import RAW_FPT_FEATURES, build_config
from training.models import ModelFactoryContext, build_single_model
from training.utils import get_device, load_json


DEVICE = get_device()

REQUIRED_COLS = ["time", *RAW_FPT_FEATURES]


def safe_torch_load(path: str | Path, device: torch.device):
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_checkpoint_name(path: str | Path) -> tuple[str | None, int | None]:
    path = Path(path)
    match = re.match(r"best_(?P<model>.+)_w(?P<window>\d+)\.pt$", path.name)
    if not match:
        return None, None
    return match.group("model"), int(match.group("window"))


def list_checkpoints(artifact_dir: str | Path) -> list[Path]:
    artifact_dir = Path(artifact_dir)
    return sorted(artifact_dir.glob("best_*.pt"), key=lambda item: item.name)


def checkpoint_display_name(path: str | Path) -> str:
    path = Path(path)
    model_name, window_size = parse_checkpoint_name(path)
    if model_name is None or window_size is None:
        return path.name
    return f"{model_name} | window={window_size}"


def read_metrics(artifact_dir: str | Path) -> pd.DataFrame | None:
    artifact_dir = Path(artifact_dir)
    candidate_files = [
        artifact_dir / "metrics_test.csv",
        artifact_dir / "metrics_all_windows_dev_test.csv",
        artifact_dir / "metrics_all_windows_dev_test.xlsx",
    ]

    for path in candidate_files:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".xlsx":
                return pd.read_excel(path)
            return pd.read_csv(path)
        except Exception:
            continue
    return None


def choose_best_checkpoint(artifact_dir: str | Path) -> Path:
    artifact_dir = Path(artifact_dir)
    checkpoints = list_checkpoints(artifact_dir)

    if not checkpoints:
        raise FileNotFoundError(f"No best_*.pt checkpoints found in: {artifact_dir}")

    best_meta_path = artifact_dir / "best_model_meta.json"
    if best_meta_path.exists():
        try:
            best_meta = load_json(best_meta_path)
            checkpoint_name = best_meta.get("checkpoint_path")
            if checkpoint_name:
                checkpoint_path = artifact_dir / checkpoint_name
                if checkpoint_path.exists():
                    return checkpoint_path
        except Exception:
            pass

    metrics = read_metrics(artifact_dir)
    if metrics is None or metrics.empty:
        return checkpoints[0]

    lower_cols = {column.lower(): column for column in metrics.columns}
    required = {"model", "window", "rmse"}
    if not required.issubset(lower_cols):
        return checkpoints[0]

    df = metrics.copy()
    if "split" in lower_cols:
        split_col = lower_cols["split"]
        test_df = df[df[split_col].astype(str).str.lower().eq("test")]
        if not test_df.empty:
            df = test_df

    model_col = lower_cols["model"]
    window_col = lower_cols["window"]
    rmse_col = lower_cols["rmse"]

    df[rmse_col] = pd.to_numeric(df[rmse_col], errors="coerce")
    df = df.dropna(subset=[rmse_col]).sort_values(rmse_col, ascending=True)

    for _, row in df.iterrows():
        model_name = str(row[model_col])
        window_size = int(row[window_col])
        checkpoint_path = artifact_dir / f"best_{model_name}_w{window_size}.pt"
        if checkpoint_path.exists():
            return checkpoint_path

    return checkpoints[0]


def load_feature_columns(artifact_dir: str | Path) -> list[str]:
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(f"feature_columns.json not found: {path}")

    feature_cols = load_json(path)
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("Invalid feature_columns.json.")
    return [str(column) for column in feature_cols]


def load_scalers(artifact_dir: str | Path):
    artifact_dir = Path(artifact_dir)
    feature_scaler_path = artifact_dir / "feature_scaler.pkl"
    target_scaler_path = artifact_dir / "target_scaler.pkl"

    if not feature_scaler_path.exists():
        raise FileNotFoundError(f"feature_scaler.pkl not found: {feature_scaler_path}")
    if not target_scaler_path.exists():
        raise FileNotFoundError(f"target_scaler.pkl not found: {target_scaler_path}")

    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    return feature_scaler, target_scaler


def _restore_config_from_checkpoint(
    checkpoint: dict[str, Any],
    artifact_dir: Path,
    feature_cols: list[str],
):
    checkpoint_config = checkpoint.get("config")
    config = build_config(artifact_dir=artifact_dir)

    if checkpoint_config is None:
        return config

    for key, value in checkpoint_config.items():
        if key in {"data_path", "artifact_dir", "dataset_mode"}:
            continue
        if hasattr(config, key):
            try:
                setattr(config, key, value)
            except Exception:
                pass
    return config


def _restore_model_context_from_checkpoint(
    checkpoint: dict[str, Any],
    feature_cols: list[str],
    feature_scaler,
    target_scaler,
) -> ModelFactoryContext:
    context_dict = checkpoint.get("model_context")
    if context_dict is not None:
        return ModelFactoryContext(**context_dict)

    if "close" not in feature_cols:
        raise ValueError("Cannot create model_context: missing feature 'close'.")

    close_idx = feature_cols.index("close")

    return ModelFactoryContext(
        input_size=len(feature_cols),
        close_feature_index=close_idx,
        close_feature_mean=float(feature_scaler.mean_[close_idx]),
        close_feature_std=float(feature_scaler.scale_[close_idx]),
        target_mean=float(target_scaler.mean_[0]),
        target_std=float(target_scaler.scale_[0]),
        use_last_close_anchor=True,
    )


def load_model_bundle(
    artifact_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    device = device or DEVICE

    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact folder not found: {artifact_dir}")

    if checkpoint_path is None:
        checkpoint_path = choose_best_checkpoint(artifact_dir)
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            if checkpoint_path.exists():
                checkpoint_path = checkpoint_path.resolve()
            else:
                checkpoint_path = artifact_dir / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    feature_scaler, target_scaler = load_scalers(artifact_dir)
    feature_cols = load_feature_columns(artifact_dir)
    checkpoint = safe_torch_load(checkpoint_path, device)

    config = _restore_config_from_checkpoint(
        checkpoint=checkpoint,
        artifact_dir=artifact_dir,
        feature_cols=feature_cols,
    )

    context = _restore_model_context_from_checkpoint(
        checkpoint=checkpoint,
        feature_cols=feature_cols,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    model_name = checkpoint.get("model_name")
    window_size = checkpoint.get("window_size")

    if model_name is None:
        parsed_model_name, _ = parse_checkpoint_name(checkpoint_path)
        model_name = parsed_model_name

    if window_size is None:
        _, parsed_window_size = parse_checkpoint_name(checkpoint_path)
        window_size = parsed_window_size

    if model_name is None or window_size is None:
        raise KeyError("Could not determine model_name or window_size from checkpoint.")

    model = build_single_model(
        model_name=str(model_name),
        context=context,
        config=config,
    )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        if getattr(config, "use_last_close_anchor", True):
            config.use_last_close_anchor = False
            model = build_single_model(
                model_name=str(model_name),
                context=context,
                config=config,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise error

    model.to(device)
    model.eval()

    return {
        "model": model,
        "config": config,
        "context": context,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "artifact_dir": artifact_dir,
        "model_name": str(model_name),
        "window_size": int(window_size),
        "feature_cols": feature_cols,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    normalized_columns = []

    for column in df.columns:
        normalized = str(column).strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("-", "_")
        normalized_columns.append(normalized)

    df.columns = normalized_columns

    alias_map = {
        "date": "time",
        "datetime": "time",
        "trading_date": "time",
        "tradingtime": "time",
        "trading_time": "time",
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "adj_close": "close",
        "vol": "volume",
        "total_volume": "volume",
    }

    df = df.rename(columns=alias_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    missing_cols = [column for column in REQUIRED_COLS if column not in df.columns]

    if missing_cols:
        raise KeyError(f"CSV is missing required columns: {missing_cols}. Need at minimum: {REQUIRED_COLS}")

    df = df[REQUIRED_COLS].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for column in RAW_FPT_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=REQUIRED_COLS)
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Data is empty after cleaning.")
    return df


def read_uploaded_csv(file_bytes) -> pd.DataFrame:
    if isinstance(file_bytes, bytes):
        buffer = io.BytesIO(file_bytes)
    else:
        buffer = file_bytes

    try:
        return pd.read_csv(buffer)
    except UnicodeDecodeError:
        if isinstance(file_bytes, bytes):
            buffer = io.BytesIO(file_bytes)
        else:
            buffer.seek(0)
        return pd.read_csv(buffer, encoding="latin1")


def predict_from_dataframe(
    df_input: pd.DataFrame,
    bundle: dict[str, Any],
    device: torch.device | None = None,
    max_rows: int | None = 300,
) -> pd.DataFrame:
    device = device or DEVICE

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    feature_scaler = bundle["feature_scaler"]
    target_scaler = bundle["target_scaler"]
    window_size = int(bundle["window_size"])

    df = clean_stock_data(df_input)
    missing_features = [column for column in feature_cols if column not in df.columns]

    if missing_features:
        raise KeyError(f"Input data missing features required by model: {missing_features}. Required: {feature_cols}")

    if len(df) < window_size:
        raise ValueError(f"Data has {len(df)} rows, not enough for window_size={window_size}.")

    features_raw = df[feature_cols].to_numpy(dtype=np.float32)
    features_scaled = feature_scaler.transform(features_raw).astype(np.float32)

    if max_rows is None or max_rows <= 0:
        first_end_idx = window_size - 1
    else:
        first_end_idx = max(window_size - 1, len(df) - int(max_rows))

    rows = []
    model.eval()

    with torch.no_grad():
        for end_idx in range(first_end_idx, len(df)):
            start_idx = end_idx - window_size + 1
            X_window = features_scaled[start_idx:end_idx + 1]
            X_tensor = torch.as_tensor(
                X_window.reshape(1, window_size, len(feature_cols)),
                dtype=torch.float32,
            ).to(device)

            pred_scaled = model(X_tensor).cpu().numpy().reshape(-1, 1)
            pred_price = float(target_scaler.inverse_transform(pred_scaled).reshape(-1)[0])

            current_row = df.iloc[end_idx]
            last_close = float(current_row["close"])

            if end_idx + 1 < len(df):
                next_row = df.iloc[end_idx + 1]
                predict_for_date = next_row["time"]
                actual_next_close = float(next_row["close"])
            else:
                predict_for_date = pd.NaT
                actual_next_close = np.nan

            predicted_change = pred_price - last_close
            predicted_change_pct = predicted_change / max(abs(last_close), 1e-8) * 100

            rows.append(
                {
                    "input_end_date": current_row["time"],
                    "predict_for_date": predict_for_date,
                    "last_close": last_close,
                    "actual_next_close": actual_next_close,
                    "predicted_next_close": pred_price,
                    "predicted_change": predicted_change,
                    "predicted_change_pct": predicted_change_pct,
                    "model_name": bundle["model_name"],
                    "window_size": window_size,
                    "checkpoint": bundle["checkpoint_path"].name,
                }
            )

    return pd.DataFrame(rows)


def format_result_table(result_df: pd.DataFrame, descending: bool = True) -> pd.DataFrame:
    df = result_df.copy()
    sort_key = pd.to_datetime(df["predict_for_date"], errors="coerce")

    if descending:
        df["_sort_date"] = sort_key.fillna(pd.Timestamp.max)
    else:
        df["_sort_date"] = sort_key.fillna(pd.Timestamp.min)

    df = df.sort_values("_sort_date", ascending=not descending).drop(columns=["_sort_date"])

    display_df = pd.DataFrame(
        {
            "Input Date": pd.to_datetime(df["input_end_date"]).dt.strftime("%Y-%m-%d"),
            "Forecast Date": pd.to_datetime(df["predict_for_date"])
            .dt.strftime("%Y-%m-%d")
            .fillna("Next Session"),
            "Current Close": df["last_close"].round(2),
            "Actual Close": df["actual_next_close"].round(2),
            "Predicted Close": df["predicted_next_close"].round(2),
            "Predicted Change": df["predicted_change"].round(2),
            "Predicted Change %": df["predicted_change_pct"].round(2),
        }
    )
    return display_df


def load_model_for_inference(
    artifact_dir: str | Path,
    checkpoint_name: str | None = None,
):
    artifact_dir = Path(artifact_dir)
    if checkpoint_name is None:
        checkpoint_path = choose_best_checkpoint(artifact_dir)
    else:
        checkpoint_path = artifact_dir / checkpoint_name

    bundle = load_model_bundle(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        device=DEVICE,
    )

    best_meta = {
        "model_name": bundle["model_name"],
        "window_size": bundle["window_size"],
        "checkpoint_path": bundle["checkpoint_path"].name,
        "feature_cols": bundle["feature_cols"],
    }

    return (
        bundle["model"],
        bundle["config"],
        bundle["context"],
        best_meta,
        bundle["checkpoint_path"],
    )


def predict_next_close(
    data_path: str | Path,
    artifact_dir: str | Path,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)

    if checkpoint_name is None:
        checkpoint_path = choose_best_checkpoint(artifact_dir)
    else:
        checkpoint_path = checkpoint_name

    bundle = load_model_bundle(
        artifact_dir=artifact_dir,
        checkpoint_path=checkpoint_path,
        device=DEVICE,
    )

    df = pd.read_csv(data_path)
    result_df = predict_from_dataframe(
        df_input=df,
        bundle=bundle,
        device=DEVICE,
        max_rows=1,
    )
    latest_result = result_df.iloc[-1]

    return {
        "model_name": bundle["model_name"],
        "window_size": bundle["window_size"],
        "checkpoint_path": str(bundle["checkpoint_path"]),
        "latest_input_date": str(pd.to_datetime(latest_result["input_end_date"]).date()),
        "latest_close": float(latest_result["last_close"]),
        "predicted_next_close": float(latest_result["predicted_next_close"]),
        "predicted_change": float(latest_result["predicted_change"]),
        "predicted_change_pct": float(latest_result["predicted_change_pct"]),
        "feature_count": len(bundle["feature_cols"]),
        "feature_cols": bundle["feature_cols"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next close from trained artifact.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    result = predict_next_close(
        data_path=args.data,
        artifact_dir=args.artifact_dir,
        checkpoint_name=args.checkpoint,
    )

    for key, value in result.items():
        if key != "feature_cols":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()