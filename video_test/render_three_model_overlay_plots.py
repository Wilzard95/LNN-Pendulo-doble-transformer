from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render overlay analysis plots for LNN, baseline MLP, and Transformer-LNN.")
    parser.add_argument("--lnn_csv", type=str, default="Otrointento/experiments/official_jax_cpu_retry/results/official_rollout_traj_045.csv")
    parser.add_argument("--mlp_csv", type=str, default="Otrointento/experiments/compare_baseline_mlp/results/official_rollout_traj_045.csv")
    parser.add_argument("--transformer_csv", type=str, default="Otrointento/experiments/compare_transformer_lnn_torch/results/official_rollout_traj_045.csv")
    parser.add_argument("--lnn_energy_csv", type=str, default="Otrointento/experiments/official_jax_cpu_retry/results/paper_energy_curve.csv")
    parser.add_argument("--mlp_energy_csv", type=str, default="Otrointento/experiments/compare_baseline_mlp/results/paper_energy_curve.csv")
    parser.add_argument("--transformer_energy_csv", type=str, default="Otrointento/experiments/compare_transformer_lnn_torch/results/paper_energy_curve.csv")
    parser.add_argument("--output_dir", type=str, default="Otrointento/experiments/model_video_compare/plots")
    return parser.parse_args()


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def load_rollout(path_str: str) -> pd.DataFrame:
    path = resolve(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout CSV: {path}")
    return pd.read_csv(path)


def load_energy(path_str: str) -> pd.DataFrame:
    path = resolve(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing energy curve CSV: {path}")
    return pd.read_csv(path)


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def validate_reference(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"Rollout length mismatch for {name}: {len(reference)} vs {len(other)}")
    for column in ("time_s", "true_theta1_rad", "true_theta2_rad", "true_omega1_rad_s", "true_omega2_rad_s"):
        if not np.allclose(reference[column].to_numpy(), other[column].to_numpy(), atol=5e-3):
            raise ValueError(f"Reference mismatch for {name} in column {column}")


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lnn = load_rollout(args.lnn_csv)
    mlp = load_rollout(args.mlp_csv)
    transformer = load_rollout(args.transformer_csv)

    validate_reference(lnn, mlp, "Baseline MLP")
    validate_reference(lnn, transformer, "Transformer-LNN")

    times = lnn["time_s"].to_numpy()
    model_rollouts = [
        ("LNN", lnn, "tab:blue"),
        ("Baseline MLP", mlp, "tab:green"),
        ("Transformer-LNN", transformer, "tab:orange"),
    ]

    state_specs = [
        ("theta1", "true_theta1_rad", "pred_theta1_rad", "angle [rad]"),
        ("theta2", "true_theta2_rad", "pred_theta2_rad", "angle [rad]"),
        ("omega1", "true_omega1_rad_s", "pred_omega1_rad_s", "omega [rad/s]"),
        ("omega2", "true_omega2_rad_s", "pred_omega2_rad_s", "omega [rad/s]"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.ravel()
    for ax, (label, true_col, pred_col, ylabel) in zip(axes, state_specs):
        ax.plot(times, lnn[true_col].to_numpy(), color="black", lw=2.0, label="Ground Truth")
        for model_label, df, color in model_rollouts:
            ax.plot(times, df[pred_col].to_numpy(), color=color, lw=1.5, alpha=0.95, label=model_label)
        ax.set_title(label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[2].set_xlabel("time [s]")
    axes[3].set_xlabel("time [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Rollout Overlay | Ground Truth vs 3 Models", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    rollout_plot_path = output_dir / "three_model_rollout_overlay_traj_045.png"
    fig.savefig(rollout_plot_path, dpi=170)
    plt.close(fig)

    err_fig, err_axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for model_label, df, color in model_rollouts:
        e1 = np.abs(wrap_angle_rad(df["pred_theta1_rad"].to_numpy() - df["true_theta1_rad"].to_numpy()))
        e2 = np.abs(wrap_angle_rad(df["pred_theta2_rad"].to_numpy() - df["true_theta2_rad"].to_numpy()))
        err_axes[0].plot(times, e1, color=color, lw=1.7, label=model_label)
        err_axes[1].plot(times, e2, color=color, lw=1.7, label=model_label)
    err_axes[0].set_title("Absolute Wrapped Error | theta1")
    err_axes[1].set_title("Absolute Wrapped Error | theta2")
    err_axes[0].set_ylabel("|error| [rad]")
    err_axes[1].set_ylabel("|error| [rad]")
    err_axes[1].set_xlabel("time [s]")
    for ax in err_axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    err_fig.tight_layout()
    error_plot_path = output_dir / "three_model_angle_error_overlay_traj_045.png"
    err_fig.savefig(error_plot_path, dpi=170)
    plt.close(err_fig)

    lnn_energy = load_energy(args.lnn_energy_csv)
    mlp_energy = load_energy(args.mlp_energy_csv)
    transformer_energy = load_energy(args.transformer_energy_csv)
    energy_specs = [
        ("LNN", lnn_energy, "tab:blue"),
        ("Baseline MLP", mlp_energy, "tab:green"),
        ("Transformer-LNN", transformer_energy, "tab:orange"),
    ]

    energy_fig, ax = plt.subplots(figsize=(11, 5))
    for model_label, df, color in energy_specs:
        ax.plot(df["time_s"].to_numpy(), df["mean_abs_energy_discrepancy_frac"].to_numpy(), lw=2.0, color=color, label=model_label)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean |energy gap| / max potential")
    ax.set_title("Paper-Style Energy Discrepancy Overlay")
    ax.legend()
    energy_fig.tight_layout()
    energy_plot_path = output_dir / "three_model_paper_energy_overlay.png"
    energy_fig.savefig(energy_plot_path, dpi=170)
    plt.close(energy_fig)

    print(f"Saved rollout overlay: {rollout_plot_path}")
    print(f"Saved angle error overlay: {error_plot_path}")
    print(f"Saved paper energy overlay: {energy_plot_path}")


if __name__ == "__main__":
    main()
