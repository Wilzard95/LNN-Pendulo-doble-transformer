from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _identity_bounds(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    low = float(min(np.nanmin(x), np.nanmin(y)))
    high = float(max(np.nanmax(x), np.nanmax(y)))
    if not np.isfinite(low) or not np.isfinite(high):
        low, high = -1.0, 1.0
    if abs(high - low) < 1e-12:
        high = low + 1.0
    return low, high


def plot_qddot_scatter(y_true: np.ndarray, y_pred: np.ndarray, plots_dir: Path, prefix: str = "eval") -> list[Path]:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    labels = ["theta1", "theta2"]
    for i, label in enumerate(labels):
        x = y_true[:, i]
        y = y_pred[:, i]
        lo, hi = _identity_bounds(x, y)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, s=4, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"qddot_true ({label})")
        ax.set_ylabel(f"qddot_pred ({label})")
        ax.set_title(f"Scatter qddot: {label}")
        ax.grid(True, alpha=0.2)
        out = plots_dir / f"{prefix}_scatter_{label}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

        lo_q = float(np.quantile(x, 0.02))
        hi_q = float(np.quantile(x, 0.98))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, s=4, alpha=0.3)
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlim(lo_q, hi_q)
        ax.set_ylim(lo_q, hi_q)
        ax.set_xlabel(f"qddot_true ({label})")
        ax.set_ylabel(f"qddot_pred ({label})")
        ax.set_title(f"Scatter qddot zoom: {label}")
        ax.grid(True, alpha=0.2)
        out_zoom = plots_dir / f"{prefix}_scatter_{label}_zoom.png"
        fig.tight_layout()
        fig.savefig(out_zoom, dpi=160)
        plt.close(fig)
        outputs.append(out_zoom)

    return outputs


def plot_error_histograms(errors: np.ndarray, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["theta1", "theta2"]
    for i, ax in enumerate(axes):
        ax.hist(errors[:, i], bins=80, alpha=0.8)
        ax.set_title(f"Error histogram: {labels[i]}")
        ax.set_xlabel("qddot_pred - qddot_true")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_loss_curves(metrics_csv_path: Path, out_path: Path) -> Path:
    metrics_csv_path = Path(metrics_csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv_path)
    x_col = "epoch" if "epoch" in df.columns else "step"
    if x_col not in df.columns:
        raise KeyError(f"Expected 'epoch' or 'step' in {metrics_csv_path.name}, found {list(df.columns)}")

    train_col = "train_loss"
    eval_col = "val_loss" if "val_loss" in df.columns else "test_loss"
    if train_col not in df.columns or eval_col not in df.columns:
        raise KeyError(
            f"Expected '{train_col}' and one of 'val_loss'/'test_loss' in {metrics_csv_path.name}, found {list(df.columns)}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[x_col], df[train_col], label=train_col)
    ax.plot(df[x_col], df[eval_col], label=eval_col)
    ax.set_xlabel(x_col)
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_rollout(
    time_s: np.ndarray,
    true_state: np.ndarray,
    pred_state: np.ndarray,
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors = pred_state - true_state

    labels = ["theta1_rad", "theta2_rad", "omega1_rad_s", "omega2_rad_s"]
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    for i in range(4):
        axes[i].plot(time_s, true_state[:, i], label="true", linewidth=1.8)
        axes[i].plot(time_s, pred_state[:, i], label="pred", linewidth=1.2)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.2)
        axes[i].legend(loc="best")

    axes[4].plot(time_s, errors[:, 0], label="error theta1")
    axes[4].plot(time_s, errors[:, 1], label="error theta2")
    axes[4].plot(time_s, errors[:, 2], label="error omega1")
    axes[4].plot(time_s, errors[:, 3], label="error omega2")
    axes[4].set_xlabel("time_s")
    axes[4].set_ylabel("error")
    axes[4].grid(True, alpha=0.2)
    axes[4].legend(loc="best", ncol=2)

    fig.suptitle("Rollout comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_sanity_trajectory(time_s: np.ndarray, q: np.ndarray, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_s, q[:, 0], label="theta1_rad")
    ax.plot(time_s, q[:, 1], label="theta2_rad")
    ax.set_xlabel("time_s")
    ax.set_ylabel("angle_rad")
    ax.set_title("Sanity trajectory example")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
