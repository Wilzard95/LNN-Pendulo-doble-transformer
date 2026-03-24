from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, FFMpegWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lnn.dynamics import qddot_from_lagrangian
from lnn.model import LagrangianMLP
from lnn.utils import select_device, wrap_coords_torch

# Editable initial state (radians and radians/second).
INITIAL_THETA1_RAD = -1.40
INITIAL_THETA2_RAD = 0.70
INITIAL_OMEGA1_RAD_S = 1.7
INITIAL_OMEGA2_RAD_S = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 10-second video from NN predicted rollout.")
    parser.add_argument("--model_path", type=str, default="checkpoints/model_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--length1", type=float, default=1.0)
    parser.add_argument("--length2", type=float, default=1.0)
    parser.add_argument("--output_name", type=str, default="mejor simulacion.mp4")
    return parser.parse_args()


def _predict_qddot(
    model: torch.nn.Module,
    q: torch.Tensor,
    qdot: torch.Tensor,
) -> torch.Tensor:
    return qddot_from_lagrangian(
        model,
        q,
        qdot,
        for_training=False,
    )


def _rollout(
    model: torch.nn.Module,
    device: torch.device,
    torch_dtype: torch.dtype,
    duration_s: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(duration_s / dt))
    q = torch.tensor(
        [[INITIAL_THETA1_RAD, INITIAL_THETA2_RAD]],
        dtype=torch_dtype,
        device=device,
    )
    qdot = torch.tensor(
        [[INITIAL_OMEGA1_RAD_S, INITIAL_OMEGA2_RAD_S]],
        dtype=torch_dtype,
        device=device,
    )
    state0 = wrap_coords_torch(torch.cat([q, qdot], dim=1))
    q = state0[:, :2]
    qdot = state0[:, 2:]

    q_hist = np.zeros((n_steps + 1, 2), dtype=np.float64)
    qdot_hist = np.zeros((n_steps + 1, 2), dtype=np.float64)
    q_hist[0] = q.detach().cpu().numpy()[0]
    qdot_hist[0] = qdot.detach().cpu().numpy()[0]

    for i in range(n_steps):
        qddot = _predict_qddot(
            model=model,
            q=q,
            qdot=qdot,
        )
        # Semi-implicit Euler
        qdot_next = qdot + dt * qddot
        q_next = q + dt * qdot_next
        state_next = wrap_coords_torch(torch.cat([q_next, qdot_next], dim=1))
        q = state_next[:, :2]
        qdot = state_next[:, 2:]
        q_hist[i + 1] = q.detach().cpu().numpy()[0]
        qdot_hist[i + 1] = qdot.detach().cpu().numpy()[0]
    return q_hist, qdot_hist


def _to_cartesian(q: np.ndarray, length1: float, length2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta1 = q[:, 0]
    theta2 = q[:, 1]
    x1 = length1 * np.sin(theta1)
    y1 = -length1 * np.cos(theta1)
    x2 = x1 + length2 * np.sin(theta2)
    y2 = y1 - length2 * np.cos(theta2)
    return x1, y1, x2, y2


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / args.output_name

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")

    model_cfg = ckpt.get("model_config", {"input_dim": 4, "hidden_dim": 500, "num_hidden_layers": 4, "activation": "softplus"})
    use_double = bool(ckpt.get("double", False))
    torch_dtype = torch.float64 if use_double else torch.float32

    device = select_device(args.device)
    model = LagrangianMLP(**model_cfg).to(device=device, dtype=torch_dtype)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    q_hist, _ = _rollout(
        model=model,
        device=device,
        torch_dtype=torch_dtype,
        duration_s=float(args.duration_s),
        dt=float(args.dt),
    )

    x1, y1, x2, y2 = _to_cartesian(q_hist, length1=float(args.length1), length2=float(args.length2))
    n_frames = max(2, int(round(float(args.duration_s) * int(args.fps))))
    frame_ids = np.linspace(0, len(q_hist) - 1, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(6, 6))
    reach = float(args.length1 + args.length2) + 0.2
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Double Pendulum - NN Predicted Rollout")
    line, = ax.plot([], [], "o-", lw=2.2)
    time_txt = ax.text(0.03, 0.95, "", transform=ax.transAxes, va="top")

    def _update(frame_idx: int):
        i = frame_ids[frame_idx]
        line.set_data([0.0, x1[i], x2[i]], [0.0, y1[i], y2[i]])
        t = i * float(args.dt)
        time_txt.set_text(f"t={t:.2f}s")
        return line, time_txt

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000.0 / float(args.fps), blit=True)
    writer = FFMpegWriter(fps=int(args.fps), bitrate=2400)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)

    print(f"Model: {model_path}")
    print(
        "Initial state (theta1, theta2, omega1, omega2)=("
        f"{INITIAL_THETA1_RAD:.6f}, {INITIAL_THETA2_RAD:.6f}, "
        f"{INITIAL_OMEGA1_RAD_S:.6f}, {INITIAL_OMEGA2_RAD_S:.6f})"
    )
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
