from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _validate_qddot_shape(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError(f"Expected [N,2] arrays, got {y_true.shape}")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred shape {y_pred.shape} does not match y_true {y_true.shape}")


def mse_per_component(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    _validate_qddot_shape(y_true, y_pred)
    mse_1 = float(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2))
    mse_2 = float(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2))
    return {
        "mse_qddot_1": mse_1,
        "mse_qddot_2": mse_2,
        "mse_qddot_mean": float(0.5 * (mse_1 + mse_2)),
    }


def mae_per_component(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    _validate_qddot_shape(y_true, y_pred)
    mae_1 = float(np.mean(np.abs(y_pred[:, 0] - y_true[:, 0])))
    mae_2 = float(np.mean(np.abs(y_pred[:, 1] - y_true[:, 1])))
    return {
        "mae_qddot_1": mae_1,
        "mae_qddot_2": mae_2,
        "mae_qddot_mean": float(0.5 * (mae_1 + mae_2)),
    }


def r2_per_component(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    _validate_qddot_shape(y_true, y_pred)

    results: dict[str, float] = {}
    labels = ["qddot_1", "qddot_2"]
    for idx, label in enumerate(labels):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        results[f"r2_{label}"] = 1.0 - ss_res / (ss_tot + 1e-12)

    # Backward-compatible aliases (deprecated naming).
    results["r2_theta1_component"] = results["r2_qddot_1"]
    results["r2_theta2_component"] = results["r2_qddot_2"]
    results["r2_qddot_mean"] = float(0.5 * (results["r2_qddot_1"] + results["r2_qddot_2"]))
    return results


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    _validate_qddot_shape(y_true, y_pred)

    mse_mean = mse(y_true, y_pred)
    mae_mean = mae(y_true, y_pred)

    metrics = {
        # Backward-compatible aggregate keys (deprecated naming retained).
        "mse_qddot": mse_mean,
        "mae_qddot": mae_mean,
        # Preferred aggregate names.
        "mse_qddot_mean": mse_mean,
        "mae_qddot_mean": mae_mean,
    }
    metrics.update(mse_per_component(y_true, y_pred))
    metrics.update(mae_per_component(y_true, y_pred))
    metrics.update(r2_per_component(y_true, y_pred))
    return metrics
