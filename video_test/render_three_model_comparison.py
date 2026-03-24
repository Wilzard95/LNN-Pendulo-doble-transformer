from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.animation import FFMpegWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a single comparison video for ground truth, LNN, baseline MLP, and transformer rollouts.")
    parser.add_argument("--lnn_csv", type=str, default="Otrointento/experiments/official_jax_cpu_retry/results/official_rollout_traj_045.csv")
    parser.add_argument("--mlp_csv", type=str, default="Otrointento/experiments/compare_baseline_mlp/results/official_rollout_traj_045.csv")
    parser.add_argument("--transformer_csv", type=str, default="Otrointento/experiments/compare_transformer_lnn_torch/results/official_rollout_traj_045.csv")
    parser.add_argument("--transformer_label", type=str, default="Transformer-LNN")
    parser.add_argument("--length1", type=float, default=1.0)
    parser.add_argument("--length2", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--bitrate", type=int, default=3200)
    parser.add_argument("--trail_frames", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="Otrointento/experiments/model_video_compare/videos/three_model_comparison_traj_045.mp4")
    return parser.parse_args()


def wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def to_cartesian(theta1: np.ndarray, theta2: np.ndarray, length1: float, length2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1 = length1 * np.sin(theta1)
    y1 = -length1 * np.cos(theta1)
    x2 = x1 + length2 * np.sin(theta2)
    y2 = y1 - length2 * np.cos(theta2)
    return x1, y1, x2, y2


def load_rollout_csv(path_str: str) -> np.ndarray:
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)


def validate_shared_reference(reference: np.ndarray, other: np.ndarray, name: str) -> None:
    if reference.shape != other.shape:
        raise ValueError(f"CSV shape mismatch for {name}: {reference.shape} vs {other.shape}")
    if not np.allclose(reference["time_s"], other["time_s"], atol=1e-10):
        raise ValueError(f"time_s mismatch for {name}")
    for field in ("true_theta1_rad", "true_theta2_rad", "true_omega1_rad_s", "true_omega2_rad_s"):
        if not np.allclose(reference[field], other[field], atol=5e-3):
            raise ValueError(f"ground-truth mismatch for {name} field {field}")


def main() -> None:
    args = parse_args()

    lnn = load_rollout_csv(args.lnn_csv)
    mlp = load_rollout_csv(args.mlp_csv)
    transformer = load_rollout_csv(args.transformer_csv)

    validate_shared_reference(lnn, mlp, "baseline_mlp")
    validate_shared_reference(lnn, transformer, "transformer")

    times = np.asarray(lnn["time_s"], dtype=np.float64)
    true_theta1 = np.asarray(lnn["true_theta1_rad"], dtype=np.float64)
    true_theta2 = np.asarray(lnn["true_theta2_rad"], dtype=np.float64)

    gt_x1, gt_y1, gt_x2, gt_y2 = to_cartesian(true_theta1, true_theta2, args.length1, args.length2)

    model_specs = [
        ("LNN", lnn, "tab:blue"),
        ("Baseline MLP", mlp, "tab:green"),
        (str(args.transformer_label), transformer, "tab:orange"),
    ]

    panels = []
    for label, data, color in model_specs:
        pred_theta1 = np.asarray(data["pred_theta1_rad"], dtype=np.float64)
        pred_theta2 = np.asarray(data["pred_theta2_rad"], dtype=np.float64)
        pred_omega1 = np.asarray(data["pred_omega1_rad_s"], dtype=np.float64)
        pred_omega2 = np.asarray(data["pred_omega2_rad_s"], dtype=np.float64)
        x1, y1, x2, y2 = to_cartesian(pred_theta1, pred_theta2, args.length1, args.length2)
        err_theta1 = np.abs(wrap_angle_rad(pred_theta1 - true_theta1))
        err_theta2 = np.abs(wrap_angle_rad(pred_theta2 - true_theta2))
        panels.append(
            {
                "label": label,
                "color": color,
                "theta1": pred_theta1,
                "theta2": pred_theta2,
                "omega1": pred_omega1,
                "omega2": pred_omega2,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "err_theta1": err_theta1,
                "err_theta2": err_theta2,
            }
        )

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reach = float(args.length1 + args.length2) + 0.25
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    titles = ["Ground Truth", "LNN", "Baseline MLP", str(args.transformer_label)]
    for ax, title in zip(axes, titles):
        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        ax.set_title(title)

    gt_line, = axes[0].plot([], [], "o-", lw=2.6, color="black")
    gt_trail, = axes[0].plot([], [], "-", lw=1.2, color="black", alpha=0.28)
    gt_text = axes[0].text(0.02, 0.98, "", transform=axes[0].transAxes, ha="left", va="top", family="monospace")

    artists = []
    for ax, panel in zip(axes[1:], panels):
        line, = ax.plot([], [], "o-", lw=2.6, color=panel["color"])
        trail, = ax.plot([], [], "-", lw=1.2, color=panel["color"], alpha=0.30)
        ghost, = ax.plot([], [], "o--", lw=1.0, color="black", alpha=0.18)
        text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", family="monospace")
        artists.append((line, trail, ghost, text, panel))

    time_text = fig.text(0.5, 0.985, "", ha="center", va="top", family="monospace")
    fig.suptitle("Double Pendulum Comparison | Same Initial State", y=0.998, fontsize=14)

    trail_frames = max(2, int(args.trail_frames))
    writer = FFMpegWriter(fps=int(args.fps), bitrate=int(args.bitrate))

    with writer.saving(fig, str(output_path), dpi=int(args.dpi)):
        for idx in range(len(times)):
            start = max(0, idx - trail_frames + 1)

            gt_line.set_data([0.0, gt_x1[idx], gt_x2[idx]], [0.0, gt_y1[idx], gt_y2[idx]])
            gt_trail.set_data(gt_x2[start : idx + 1], gt_y2[start : idx + 1])
            gt_text.set_text(
                f"theta1={true_theta1[idx]: .3f}\n"
                f"theta2={true_theta2[idx]: .3f}\n"
                f"t={times[idx]:.2f}s"
            )

            for line, trail, ghost, text, panel in artists:
                line.set_data([0.0, panel["x1"][idx], panel["x2"][idx]], [0.0, panel["y1"][idx], panel["y2"][idx]])
                trail.set_data(panel["x2"][start : idx + 1], panel["y2"][start : idx + 1])
                ghost.set_data([0.0, gt_x1[idx], gt_x2[idx]], [0.0, gt_y1[idx], gt_y2[idx]])
                text.set_text(
                    f"theta1={panel['theta1'][idx]: .3f}\n"
                    f"theta2={panel['theta2'][idx]: .3f}\n"
                    f"|e1|={panel['err_theta1'][idx]:.3f}\n"
                    f"|e2|={panel['err_theta2'][idx]:.3f}"
                )

            time_text.set_text(f"trajectory=045 | t={times[idx]:.2f}s")
            writer.grab_frame()

    plt.close(fig)

    print(f"Saved comparison video: {output_path}")
    print(f"frames={len(times)} | fps={args.fps} | trail_frames={trail_frames}")


if __name__ == "__main__":
    main()
