from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from training.config import DEFAULT_MERGED_EXCLUDE_COLUMNS, RAW_FPT_FEATURES, ExperimentConfig
from training.utils import ensure_dir, save_json


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

        if len(self.X) != len(self.y):
            raise ValueError("X và y không cùng số lượng mẫu.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_cols: List[str]
    features_raw: np.ndarray
    target_raw: np.ndarray
    close_raw: np.ndarray
    features_scaled: np.ndarray
    target_scaled: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    train_end: int
    val_end: int
    train_fit_end: int
    all_dates: np.ndarray
    target_dates_raw: np.ndarray


def _clean_base_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy data file: {data_path}")

    df_raw = pd.read_csv(data_path)

    if "time" not in df_raw.columns:
        raise KeyError("Thiếu cột bắt buộc: time")

    if "close" not in df_raw.columns:
        raise KeyError("Thiếu cột bắt buộc: close")

    df_raw["time"] = pd.to_datetime(df_raw["time"], errors="coerce")

    df_model = (
        df_raw.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["time", "close"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    return df_model


def load_clean_dataframe(data_path: str | Path) -> pd.DataFrame:
    return _clean_base_dataframe(Path(data_path))


def make_target_dataframe(df_model: pd.DataFrame, config: ExperimentConfig, drop_last_target: bool = True) -> pd.DataFrame:
    df_model = df_model.copy()

    df_model[config.target_col] = df_model["close"].shift(-1)
    df_model[config.target_date_col] = df_model["time"].shift(-1)

    if drop_last_target:
        df_model = df_model.dropna(subset=[config.target_col, config.target_date_col]).reset_index(drop=True)

    return df_model


def select_feature_columns(df: pd.DataFrame, config: ExperimentConfig) -> List[str]:
    mode = config.dataset_mode.lower().strip()

    if mode == "raw":
        missing_features = [column for column in RAW_FPT_FEATURES if column not in df.columns]
        if missing_features:
            raise KeyError(f"Raw FPT data thiếu feature: {missing_features}")

        return RAW_FPT_FEATURES.copy()

    if mode == "merged":
        exclude_set = set(DEFAULT_MERGED_EXCLUDE_COLUMNS)
        exclude_set.add(config.target_col)
        exclude_set.add(config.target_date_col)

        feature_cols = [
            column
            for column in df.columns
            if column not in exclude_set and pd.api.types.is_numeric_dtype(df[column])
        ]

        if "close" not in feature_cols:
            raise ValueError("Merged dataset phải có cột close để dùng last-close residual anchor.")

        if not feature_cols:
            raise ValueError("Không tìm thấy numeric feature columns trong merged dataset.")

        return feature_cols

    raise ValueError("dataset_mode phải là 'raw' hoặc 'merged'.")


def load_and_prepare_dataframe(config: ExperimentConfig) -> Tuple[pd.DataFrame, List[str]]:
    df_model = _clean_base_dataframe(Path(config.data_path))
    df = make_target_dataframe(df_model, config, drop_last_target=True)

    feature_cols = select_feature_columns(df, config)

    required_cols = ["time", "close", config.target_col, config.target_date_col] + feature_cols
    missing_required = [column for column in required_cols if column not in df.columns]
    if missing_required:
        raise KeyError(f"Thiếu cột bắt buộc: {missing_required}")

    df = df.dropna(subset=feature_cols + [config.target_col, config.target_date_col]).copy()
    df = df[df["time"] >= pd.to_datetime(config.start_date)].reset_index(drop=True)

    if len(df) < max(config.window_sizes) + 10:
        raise ValueError("Dữ liệu quá ít sau khi filter start_date/window_sizes.")

    return df, feature_cols


def fit_scalers_and_create_arrays(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: ExperimentConfig,
    save_artifacts: bool = True,
) -> PreparedData:
    features_raw = df[feature_cols].to_numpy(dtype=np.float32)
    target_raw = df[config.target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
    close_raw = df["close"].to_numpy(dtype=np.float32)

    all_dates = df["time"].to_numpy()
    target_dates_raw = df[config.target_date_col].to_numpy()

    n_rows = len(df)
    train_end = int(n_rows * config.train_ratio)
    val_end = int(n_rows * (config.train_ratio + config.val_ratio))

    train_fit_end = max(train_end - 1, 1)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_scaler.fit(features_raw[:train_fit_end])
    target_scaler.fit(target_raw[:train_fit_end])

    features_scaled = feature_scaler.transform(features_raw).astype(np.float32)
    target_scaled = target_scaler.transform(target_raw).astype(np.float32)

    if save_artifacts and config.artifact_dir is not None:
        artifact_dir = ensure_dir(config.artifact_dir)

        joblib.dump(feature_scaler, artifact_dir / "feature_scaler.pkl")
        joblib.dump(target_scaler, artifact_dir / "target_scaler.pkl")
        save_json(feature_cols, artifact_dir / "feature_columns.json")
        save_json(config.to_jsonable(), artifact_dir / "experiment_config.json")

    return PreparedData(
        df=df,
        feature_cols=feature_cols,
        features_raw=features_raw,
        target_raw=target_raw,
        close_raw=close_raw,
        features_scaled=features_scaled,
        target_scaled=target_scaled,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_end=train_end,
        val_end=val_end,
        train_fit_end=train_fit_end,
        all_dates=all_dates,
        target_dates_raw=target_dates_raw,
    )


def create_sequences_for_window(
    prepared: PreparedData,
    window_size: int,
    batch_size: int,
) -> Dict[str, np.ndarray | DataLoader]:
    X_sequences = []
    y_sequences = []
    last_close_sequences = []
    input_end_dates = []
    target_dates = []
    sequence_target_indices = []

    for end_idx in range(window_size - 1, len(prepared.features_scaled)):
        start_idx = end_idx - window_size + 1
        target_idx = end_idx + 1

        X_sequences.append(prepared.features_scaled[start_idx:end_idx + 1])
        y_sequences.append(prepared.target_scaled[end_idx])
        last_close_sequences.append(prepared.close_raw[end_idx])
        input_end_dates.append(prepared.all_dates[end_idx])
        target_dates.append(prepared.target_dates_raw[end_idx])
        sequence_target_indices.append(target_idx)

    X_sequences = np.asarray(X_sequences, dtype=np.float32)
    y_sequences = np.asarray(y_sequences, dtype=np.float32).reshape(-1, 1)
    last_close_sequences = np.asarray(last_close_sequences, dtype=np.float32)
    input_end_dates = np.asarray(input_end_dates)
    target_dates = np.asarray(target_dates)
    sequence_target_indices = np.asarray(sequence_target_indices, dtype=np.int64)

    train_mask = sequence_target_indices < prepared.train_end
    val_mask = (sequence_target_indices >= prepared.train_end) & (sequence_target_indices < prepared.val_end)
    test_mask = sequence_target_indices >= prepared.val_end

    data: Dict[str, np.ndarray | DataLoader] = {
        "window_size": window_size,
        "X_train": X_sequences[train_mask],
        "y_train": y_sequences[train_mask],
        "X_val": X_sequences[val_mask],
        "y_val": y_sequences[val_mask],
        "X_test": X_sequences[test_mask],
        "y_test": y_sequences[test_mask],
        "train_dates": target_dates[train_mask],
        "val_dates": target_dates[val_mask],
        "test_dates": target_dates[test_mask],
        "train_input_end_dates": input_end_dates[train_mask],
        "val_input_end_dates": input_end_dates[val_mask],
        "test_input_end_dates": input_end_dates[test_mask],
        "last_close_train": last_close_sequences[train_mask],
        "last_close_val": last_close_sequences[val_mask],
        "last_close_test": last_close_sequences[test_mask],
    }

    data["train_loader"] = DataLoader(
        StockDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    data["val_loader"] = DataLoader(
        StockDataset(data["X_val"], data["y_val"]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    data["test_loader"] = DataLoader(
        StockDataset(data["X_test"], data["y_test"]),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return data


def create_window_datasets(prepared: PreparedData, config: ExperimentConfig) -> Dict[int, Dict[str, np.ndarray | DataLoader]]:
    return {
        window_size: create_sequences_for_window(
            prepared=prepared,
            window_size=window_size,
            batch_size=config.batch_size,
        )
        for window_size in config.window_sizes
    }


def prepare_latest_sequence(
    data_path: str | Path,
    feature_cols: List[str],
    feature_scaler: StandardScaler,
    window_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    df_model = load_clean_dataframe(data_path)

    missing_features = [column for column in feature_cols if column not in df_model.columns]
    if missing_features:
        raise KeyError(f"Data dùng deploy thiếu feature: {missing_features}")

    df_model = df_model.dropna(subset=feature_cols + ["time", "close"]).copy()
    df_model = df_model.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    if len(df_model) < window_size:
        raise ValueError(f"Dữ liệu chỉ có {len(df_model)} rows, không đủ window_size={window_size}.")

    latest_window_df = df_model.tail(window_size).copy()
    latest_features_raw = latest_window_df[feature_cols].to_numpy(dtype=np.float32)
    latest_features_scaled = feature_scaler.transform(latest_features_raw).astype(np.float32)

    X_latest = latest_features_scaled.reshape(1, window_size, len(feature_cols))

    return X_latest, latest_window_df
