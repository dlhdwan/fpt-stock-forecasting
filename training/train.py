from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from training.config import ExperimentConfig, build_config
from training.data import (
    create_window_datasets,
    fit_scalers_and_create_arrays,
    load_and_prepare_dataframe,
)
from training.metrics import calculate_metrics
from training.models import (
    ModelFactoryContext,
    build_single_model,
    count_trainable_parameters,
)
from training.utils import ensure_dir, get_device, save_json, set_seed

DEVICE = get_device()

def create_model_context(
    input_size: int,
    feature_cols: list[str],
    feature_scaler,
    target_scaler,
    config: ExperimentConfig,
) -> ModelFactoryContext:
    if "close" not in feature_cols:
        raise ValueError("Feature 'close' required for last-close residual anchor.")

    close_feature_index = feature_cols.index("close")

    return ModelFactoryContext(
        input_size=input_size,
        close_feature_index=close_feature_index,
        close_feature_mean=float(feature_scaler.mean_[close_feature_index]),
        close_feature_std=float(feature_scaler.scale_[close_feature_index]),
        target_mean=float(target_scaler.mean_[0]),
        target_std=float(target_scaler.scale_[0]),
        use_last_close_anchor=config.use_last_close_anchor,
    )

def train_one_epoch(model, data_loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(X_batch)

        if predictions.shape != y_batch.shape:
            raise RuntimeError(f"Shape mismatch: {predictions.shape} != {y_batch.shape}")

        loss = criterion(predictions, y_batch)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def evaluate_scaled_loss(model, data_loader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            predictions = model(X_batch)

            if predictions.shape != y_batch.shape:
                raise RuntimeError(f"Shape mismatch: {predictions.shape} != {y_batch.shape}")

            loss = criterion(predictions, y_batch)
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

def train_model(
    model,
    model_name: str,
    window_size: int,
    train_loader,
    val_loader,
    artifact_dir: Path,
    config: ExperimentConfig,
    model_context: ModelFactoryContext,
    feature_cols: list[str],
):
    model = model.to(DEVICE)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    checkpoint_path = artifact_dir / f"best_{model_name}_w{window_size}.pt"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_scaled_loss(model, val_loader, criterion)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_name": model_name,
                    "window_size": window_size,
                    "input_size": model_context.input_size,
                    "feature_cols": feature_cols,
                    "model_state_dict": best_state,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "model_context": model_context.__dict__,
                    "config": config.to_jsonable(),
                },
                checkpoint_path,
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"W={window_size:02d} | {model_name:12s} | Epoch={epoch:03d} | Train={train_loss:.6f} | Dev={val_loss:.6f} | LR={current_lr:.2e}")

        if patience_counter >= config.patience:
            print(f"W={window_size:02d} | {model_name} early stopping at epoch {epoch}. Best dev loss={best_val_loss:.6f}")
            break

    if best_state is None:
        raise RuntimeError(f"Could not find best_state for {model_name}, window={window_size}.")

    model.load_state_dict(best_state)
    return model, history, checkpoint_path

def predict_original_scale(model, data_loader, target_scaler) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_scaled = []
    y_pred_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            predictions = model(X_batch)
            y_pred_scaled.append(predictions.cpu().numpy())
            y_true_scaled.append(y_batch.numpy())

    y_true_scaled = np.concatenate(y_true_scaled, axis=0).reshape(-1, 1)
    y_pred_scaled = np.concatenate(y_pred_scaled, axis=0).reshape(-1, 1)
    y_true = target_scaler.inverse_transform(y_true_scaled).reshape(-1)
    y_pred = target_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    return y_true, y_pred

def train_experiment(config: ExperimentConfig) -> pd.DataFrame:
    set_seed(config.seed)
    artifact_dir = ensure_dir(config.artifact_dir)

    print("Device:", DEVICE)
    print("Data path:", config.data_path)
    print("Artifact dir:", artifact_dir)

    df, feature_cols = load_and_prepare_dataframe(config)
    prepared = fit_scalers_and_create_arrays(df, feature_cols, config, save_artifacts=True)
    window_datasets = create_window_datasets(prepared, config)

    print("Data shape:", df.shape)
    print("Date range:", df["time"].min(), "->", df["time"].max())
    print("Feature count:", len(feature_cols))
    print("Features:", feature_cols)
    print("Train rows:", prepared.train_end)
    print("Dev rows:", prepared.val_end - prepared.train_end)
    print("Test rows:", len(df) - prepared.val_end)

    all_results = []
    all_predictions: Dict[tuple, dict] = {}
    all_histories: Dict[int, dict] = {}

    for window_size, data in window_datasets.items():
        print("-" * 80)
        print(f"WINDOW SIZE = {window_size}")
        print("-" * 80)

        input_size = data["X_train"].shape[-1]
        model_context = create_model_context(
            input_size=input_size,
            feature_cols=feature_cols,
            feature_scaler=prepared.feature_scaler,
            target_scaler=prepared.target_scaler,
            config=config,
        )

        all_histories[window_size] = {}

        for model_idx, model_name in enumerate(config.model_names):
            model_seed = config.seed + window_size * 100 + model_idx
            set_seed(model_seed)
            
            print(f"Training model={model_name}, window={window_size}, seed={model_seed}")
            model_to_train = build_single_model(model_name=model_name, context=model_context, config=config)
            print(f"Parameters: {count_trainable_parameters(model_to_train):,}")

            trained_model, history, checkpoint_path = train_model(
                model=model_to_train,
                model_name=model_name,
                window_size=window_size,
                train_loader=data["train_loader"],
                val_loader=data["val_loader"],
                artifact_dir=artifact_dir,
                config=config,
                model_context=model_context,
                feature_cols=feature_cols,
            )

            all_histories[window_size][model_name] = history

            y_val_true, y_val_pred = predict_original_scale(trained_model, data["val_loader"], prepared.target_scaler)
            val_metrics = calculate_metrics(y_val_true, y_val_pred, data["last_close_val"])
            all_results.append({"window": window_size, "model": model_name, "split": "dev", **val_metrics})

            y_test_true, y_test_pred = predict_original_scale(trained_model, data["test_loader"], prepared.target_scaler)
            test_metrics = calculate_metrics(y_test_true, y_test_pred, data["last_close_test"])
            all_results.append({"window": window_size, "model": model_name, "split": "test", **test_metrics})

            all_predictions[(window_size, model_name, "test")] = {
                "date": data["test_dates"],
                "input_end_date": data["test_input_end_dates"],
                "y_true": y_test_true,
                "y_pred": y_test_pred,
                "last_close": data["last_close_test"],
            }

            print("Dev metrics:", val_metrics)
            print("Test metrics:", test_metrics)

            del trained_model, model_to_train
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(artifact_dir / "metrics_all_windows_dev_test.csv", index=False)

    dev_results_df = results_df[results_df["split"] == "dev"].sort_values("RMSE").reset_index(drop=True)
    test_results_df = results_df[results_df["split"] == "test"].sort_values("RMSE").reset_index(drop=True)

    dev_results_df.to_csv(artifact_dir / "metrics_dev.csv", index=False)
    test_results_df.to_csv(artifact_dir / "metrics_test.csv", index=False)

    save_json(all_results, artifact_dir / "metrics_all_windows_dev_test.json")
    save_json(all_histories, artifact_dir / "histories.json")

    if len(test_results_df) > 0:
        best_test = test_results_df.iloc[0].to_dict()
        best_meta = {
            "model_name": best_test["model"],
            "window_size": int(best_test["window"]),
            "checkpoint_path": f"best_{best_test['model']}_w{int(best_test['window'])}.pt",
            "metrics": best_test,
            "feature_cols": feature_cols,
            "target_col": config.target_col,
            "target_date_col": config.target_date_col,
        }
        save_json(best_meta, artifact_dir / "best_model_meta.json")

        key = (int(best_test["window"]), best_test["model"], "test")
        if key in all_predictions:
            pred = all_predictions[key]
            pred_df = pd.DataFrame({
                "target_date": pred["date"],
                "input_end_date": pred["input_end_date"],
                "y_true": pred["y_true"],
                "y_pred": pred["y_pred"],
                "last_close": pred["last_close"],
            })
            pred_df.to_csv(artifact_dir / "best_test_predictions.csv", index=False)

    print("Saved metrics to:", artifact_dir)
    return results_df

def parse_args():
    parser = argparse.ArgumentParser(description="Train FPT stock forecasting models.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data.")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Path to output artifact directory.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--windows", type=int, nargs="*", default=None)
    parser.add_argument("--models", type=str, nargs="*", default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    config = build_config(
        data_path=args.data,
        artifact_dir=args.artifact_dir,
    )

    if args.epochs is not None: config.epochs = args.epochs
    if args.windows: config.window_sizes = args.windows
    if args.models: config.model_names = args.models
    if args.start_date: config.start_date = args.start_date
    if args.batch_size is not None: config.batch_size = args.batch_size

    train_experiment(config)

if __name__ == "__main__":
    main()