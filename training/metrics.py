from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred, last_close) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    last_close = np.asarray(last_close).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(
        np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)
    ) * 100

    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    true_direction = np.sign(y_true - last_close)
    pred_direction = np.sign(y_pred - last_close)
    directional_accuracy = np.mean(true_direction == pred_direction) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "Bias": float(bias),
        "Directional_Accuracy": float(directional_accuracy),
    }
