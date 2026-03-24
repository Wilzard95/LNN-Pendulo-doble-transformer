from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import matplotlib
import numpy as np
from matplotlib.animation import FFMpegWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lnn.utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a true-vs-pred rollout video from a saved repo-faithful rollout CSV.")
    parser.add_argument("--out_dir", type=str, default="experiments/repo_faithful")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--trajectory_id", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--bitrate", type=int, default=2600)
    parser.add_argument("--trail_frames", type=int, default=20)
    parser.add_argument("--length1", type=float, default=None)
    parser.add_argument("--length2", type=float, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    return parser.parse_args()


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def resolve_csv_path(args: argparse.Namespace, out_dir: Path) -> Path:
    if args.csv_path is not None:
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        return csv_path

    results_dir = out_dir / "results"
    if args.trajectory_id is not None:
        return results_dir / f"rollout_traj_{int(args.trajectory_id):03d}.csv"

    csv_candidates = sorted(results_dir.glob("rollout_traj_*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No rollout CSV files found in {results_dir}")
    return csv_candidates[0]


def infer_trajectory_id(csv_path: Path) -> str:
    match = re.search(r"rollout_traj_(\d+)", csv_path.stem)
    return match.group(1) if match else "unknown"


def load_lengths(out_dir: Path, args: argparse.Namespace) -> tuple[float, float]:
    if args.length1 is not None and args.length2 is not None:
        return float(args.length1), float(args.length2)

    run_config = load_json(out_dir / "run_config.json")
    physics = run_config.get("physics", {})
    length1 = float(args.length1) if args.length1 is not None else float(physics.get("l1", 1.0))
    length2 = float(args.length2) if args.length2 is not None else float(physics.get("l2", 1.0))
    return length1, length2


def to_cartesian(theta1: np.ndarray, theta2: np.ndarray, length1: float, length2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = length1 * np.sin(theta1)
    y1 = -length1 * np.cos(theta1)
    x2 = x1 + length2 * np.sin(theta2)
    y2 = y1 - length2 * np.cos(theta2)
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    csv_path = resolve_csv_path(args, out_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Rollout CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    times = np.asarray(data["time_s"], dtype=np.float64)
    true_theta1 = np.asarray(data["true_theta1_rad"], dtype=np.float64)
    true_theta2 = np.asarray(data["true_theta2_rad"], dtype=np.float64)
    true_omega1 = np.asarray(data["true_omega1_rad_s"], dtype=np.float64)
    true_omega2 = np.asarray(data["true_omega2_rad_s"], dtype=np.float64)
    pred_theta1 = np.asarray(data["pred_theta1_rad"], dtype=np.float64)
    pred_theta2 = np.asarray(data["pred_theta2_rad"], dtype=np.float64)
    pred_omega1 = np.asarray(data["pred_omega1_rad_s"], dtype=np.float64)
    pred_omega2 = np.asarray(data["pred_omega2_rad_s"], dtype=np.float64)

    length1, length2 = load_lengths(out_dir, args)
    true_x1, true_y1, true_x2, true_y2 = to_cartesian(true_theta1, true_theta2, length1, length2)
    pred_x1, pred_y1, pred_x2, pred_y2 = to_cartesian(pred_theta1, pred_theta2, length1, length2)

    err_theta1 = wrap_angle_rad(pred_theta1 - true_theta1)
    err_theta2 = wrap_angle_rad(pred_theta2 - true_theta2)

    median_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    fps = int(args.fps) if args.fps is not None else max(8, int(round(1.0 / max(median_dt, 1e-6))))

    trajectory_id = infer_trajectory_id(csv_path)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{csv_path.stem}_true_vs_pred.mp4"
    output_path = videos_dir / output_name

    reach = float(length1 + length2) + 0.25
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.5], hspace=0.18, wspace=0.12)
    ax_true = fig.add_subplot(grid[0, 0])
    ax_pred = fig.add_subplot(grid[0, 1])
    ax_err = fig.add_subplot(grid[1, :])

    for ax, title in ((ax_true, "Ground truth"), (ax_pred, "Prediction")):
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(title)

    ax_err.set_title("Wrapped angular error")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("error [rad]")
    ax_err.grid(True, alpha=0.25)
    ax_err.plot(times, err_theta1, lw=1.6, color="tab:green", label="theta1 error")
    ax_err.plot(times, err_theta2, lw=1.6, color="tab:red", label="theta2 error")
    ax_err.legend(loc="upper right")
    cursor = ax_err.axvline(times[0], color="black", lw=1.2, alpha=0.7)

    true_line, = ax_true.plot([], [], "o-", lw=2.5, color="tab:blue")
    pred_line, = ax_pred.plot([], [], "o-", lw=2.5, color="tab:orange")
    true_trail, = ax_true.plot([], [], "-", lw=1.2, color="tab:blue", alpha=0.35)
    pred_trail, = ax_pred.plot([], [], "-", lw=1.2, color="tab:orange", alpha=0.35)

    true_text = ax_true.text(0.02, 0.98, "", transform=ax_true.transAxes, ha="left", va="top", family="monospace")
    pred_text = ax_pred.text(0.02, 0.98, "", transform=ax_pred.transAxes, ha="left", va="top", family="monospace")
    time_text = fig.text(0.5, 0.97, "", ha="center", va="top", family="monospace")
    fig.suptitle(f"Repo-faithful rollout comparison | trajectory {trajectory_id}", y=0.995, fontsize=13)

    trail_frames = max(2, int(args.trail_frames))

    writer = FFMpegWriter(fps=fps, bitrate=int(args.bitrate))
    with writer.saving(fig, str(output_path), dpi=int(args.dpi)):
        for idx in range(len(times)):
            start = max(0, idx - trail_frames + 1)

            true_line.set_data([0.0, true_x1[idx], true_x2[idx]], [0.0, true_y1[idx], true_y2[idx]])
            pred_line.set_data([0.0, pred_x1[idx], pred_x2[idx]], [0.0, pred_y1[idx], pred_y2[idx]])
            true_trail.set_data(true_x2[start : idx + 1], true_y2[start : idx + 1])
            pred_trail.set_data(pred_x2[start : idx + 1], pred_y2[start : idx + 1])
            cursor.set_xdata([times[idx], times[idx]])

            true_text.set_text(
                f"theta1={true_theta1[idx]: .3f}\n"
                f"theta2={true_theta2[idx]: .3f}\n"
                f"omega1={true_omega1[idx]: .3f}\n"
                f"omega2={true_omega2[idx]: .3f}"
            )
            pred_text.set_text(
                f"theta1={pred_theta1[idx]: .3f}\n"
                f"theta2={pred_theta2[idx]: .3f}\n"
                f"omega1={pred_omega1[idx]: .3f}\n"
                f"omega2={pred_omega2[idx]: .3f}"
            )
            time_text.set_text(
                f"t={times[idx]:.2f}s | "
                f"|e_theta1|={abs(err_theta1[idx]):.3f} rad | "
                f"|e_theta2|={abs(err_theta2[idx]):.3f} rad"
            )

            writer.grab_frame()

    plt.close(fig)

    print(f"Saved video: {output_path}")
    print(f"Source CSV: {csv_path}")
    print(f"fps={fps} | frames={len(times)} | trail_frames={trail_frames}")


if __name__ == "__main__":
    main()
