from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an energy-focused comparison video for LNN, baseline MLP, and Transformer-LNN.")
    parser.add_argument("--lnn_csv", type=str, default="Otrointento/experiments/model_video_compare_140s/official_lnn_140s/results/official_rollout_traj_045.csv")
    parser.add_argument("--mlp_csv", type=str, default="Otrointento/experiments/model_video_compare_140s/baseline_mlp_140s/results/official_rollout_traj_045.csv")
    parser.add_argument("--transformer_csv", type=str, default="Otrointento/experiments/model_video_compare_140s/transformer_lnn_140s/results/official_rollout_traj_045.csv")
    parser.add_argument("--output_path", type=str, default="Otrointento/experiments/model_video_compare_140s/videos/energy_comparison_traj_045_140s.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--bitrate", type=int, default=3200)
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


def validate_reference(reference: pd.DataFrame, other: pd.DataFrame, name: str) -> None:
    if len(reference) != len(other):
        raise ValueError(f"Rollout length mismatch for {name}: {len(reference)} vs {len(other)}")
    if not np.allclose(reference["time_s"].to_numpy(), other["time_s"].to_numpy(), atol=1e-10):
        raise ValueError(f"time mismatch for {name}")
    for column in ("true_theta1_rad", "true_theta2_rad", "true_omega1_rad_s", "true_omega2_rad_s"):
        if not np.allclose(reference[column].to_numpy(), other[column].to_numpy(), atol=5e-3):
            raise ValueError(f"Reference mismatch for {name} in column {column}")


def hamiltonian_np(
    theta1: np.ndarray,
    theta2: np.ndarray,
    omega1: np.ndarray,
    omega2: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.8,
) -> np.ndarray:
    t1v = 0.5 * m1 * (l1 * omega1) ** 2
    t2v = 0.5 * m2 * ((l1 * omega1) ** 2 + (l2 * omega2) ** 2 + 2.0 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
    y1 = -l1 * np.cos(theta1)
    y2 = y1 - l2 * np.cos(theta2)
    v = m1 * g * y1 + m2 * g * y2
    return t1v + t2v + v


def max_potential_energy(m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.8) -> float:
    return float(m1 * g * l1 + m2 * g * (l1 + l2))


def series_energy(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return hamiltonian_np(
        theta1=df[f"{prefix}_theta1_rad"].to_numpy(),
        theta2=df[f"{prefix}_theta2_rad"].to_numpy(),
        omega1=df[f"{prefix}_omega1_rad_s"].to_numpy(),
        omega2=df[f"{prefix}_omega2_rad_s"].to_numpy(),
    )


def main() -> None:
    args = parse_args()

    lnn = load_rollout(args.lnn_csv)
    mlp = load_rollout(args.mlp_csv)
    transformer = load_rollout(args.transformer_csv)

    validate_reference(lnn, mlp, "Baseline MLP")
    validate_reference(lnn, transformer, "Transformer-LNN")

    times = lnn["time_s"].to_numpy()
    true_energy = series_energy(lnn, "true")
    lnn_energy = series_energy(lnn, "pred")
    mlp_energy = series_energy(mlp, "pred")
    transformer_energy = series_energy(transformer, "pred")

    energy_norm = max(max_potential_energy(), 1.0e-9)
    lnn_err = np.abs(lnn_energy - true_energy) / energy_norm
    mlp_err = np.abs(mlp_energy - true_energy) / energy_norm
    transformer_err = np.abs(transformer_energy - true_energy) / energy_norm

    output_path = resolve(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.18)
    ax_energy = fig.add_subplot(grid[0, 0])
    ax_err = fig.add_subplot(grid[1, 0], sharex=ax_energy)

    ax_energy.plot(times, true_energy, color="black", lw=2.2, label="Ground Truth")
    ax_energy.plot(times, lnn_energy, color="tab:blue", lw=1.5, label="LNN")
    ax_energy.plot(times, mlp_energy, color="tab:green", lw=1.5, label="Baseline MLP")
    ax_energy.plot(times, transformer_energy, color="tab:orange", lw=1.5, label="Transformer-LNN")
    ax_energy.set_ylabel("energy")
    ax_energy.set_title("Energy Along the Same 140 s Rollout")
    ax_energy.grid(True, alpha=0.25)
    ax_energy.legend(loc="upper right")

    ax_err.plot(times, lnn_err, color="tab:blue", lw=1.7, label="LNN")
    ax_err.plot(times, mlp_err, color="tab:green", lw=1.7, label="Baseline MLP")
    ax_err.plot(times, transformer_err, color="tab:orange", lw=1.7, label="Transformer-LNN")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel(r"|$\Delta E$| / $E_{pot,max}$")
    ax_err.set_title("Paper-Style Energy Discrepancy on This Trajectory")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="upper right")

    energy_cursor = ax_energy.axvline(times[0], color="black", lw=1.2, alpha=0.75)
    err_cursor = ax_err.axvline(times[0], color="black", lw=1.2, alpha=0.75)

    summary_text = fig.text(0.5, 0.985, "", ha="center", va="top", family="monospace")
    lower_text = ax_err.text(0.01, 0.98, "", transform=ax_err.transAxes, ha="left", va="top", family="monospace")

    writer = FFMpegWriter(fps=int(args.fps), bitrate=int(args.bitrate))
    with writer.saving(fig, str(output_path), dpi=int(args.dpi)):
        for idx, t in enumerate(times):
            energy_cursor.set_xdata([t, t])
            err_cursor.set_xdata([t, t])

            summary_text.set_text(
                f"Energy Comparison | trajectory=045 | t={t:.2f}s"
            )
            lower_text.set_text(
                f"LNN          : {lnn_err[idx]:.4f}\n"
                f"Baseline MLP : {mlp_err[idx]:.4f}\n"
                f"Transformer  : {transformer_err[idx]:.4f}"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved energy video: {output_path}")
    print(f"frames={len(times)} | fps={args.fps}")


if __name__ == "__main__":
    main()
