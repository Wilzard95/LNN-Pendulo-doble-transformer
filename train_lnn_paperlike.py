from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.optim import Adam
from tqdm import tqdm

from lnn.dynamics import qddot_from_lagrangian, state_delta_from_lagrangian, xdot_from_lagrangian
from lnn.model import LagrangianMLP
from lnn.utils import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, ensure_output_dirs, save_json, select_device, set_seed, wrap_coords_np


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PhysicsParams:
    m1: float
    m2: float
    l1: float
    l2: float
    g: float


@dataclass
class TrainPreset:
    name: str
    hidden_dim: int
    num_hidden_layers: int
    activation: str
    output_dim: int
    objective: str
    loss_mode: str
    training_mode: str
    dataset_size: int
    num_steps: int
    batch_size: int
    lr: float
    lr2: float
    l2reg: float
    dt: float
    n_updates: int
    eval_every: int
    test_ratio: float
    sample_mode: str
    use_double: bool


PRESETS = {
    "paper": TrainPreset(
        name="paper",
        hidden_dim=128,
        num_hidden_layers=2,
        activation="softplus",
        output_dim=1,
        objective="continuous",
        loss_mode="mse",
        training_mode="full_batch",
        dataset_size=3000,
        num_steps=50000,
        batch_size=100,
        lr=1e-3,
        lr2=1e-4,
        l2reg=0.0,
        dt=0.1,
        n_updates=1,
        eval_every=100,
        test_ratio=0.5,
        sample_mode="ordered",
        use_double=True,
    ),
    "repo_hyperopt_best": TrainPreset(
        name="repo_hyperopt_best",
        hidden_dim=159,
        num_hidden_layers=4,
        activation="softplus",
        output_dim=1,
        objective="delta",
        loss_mode="l1_sum",
        training_mode="minibatch",
        dataset_size=25000,
        num_steps=40000,
        batch_size=41,
        lr=0.005491648617408025,
        lr2=1.9651742128096374e-05,
        l2reg=0.14544610264965038,
        dt=0.08986129562057266,
        n_updates=6,
        eval_every=1000,
        test_ratio=0.1,
        sample_mode="random",
        use_double=False,
    ),
}


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret as bool: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper-like LNN in PyTorch/CUDA for double pendulum data.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="paper")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR / "experiments" / "paperlike_lnn"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=0, help="Limit number of trajectories. 0 means all.")
    parser.add_argument("--dataset_size", type=int, default=None, help="Override preset dataset size.")
    parser.add_argument("--num_steps", type=int, default=None, help="Override preset number of optimizer updates.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override preset batch size/chunk size.")
    parser.add_argument("--test_ratio", type=float, default=None, help="Override preset test ratio.")
    parser.add_argument("--eval_every", type=int, default=None, help="Override preset evaluation interval.")
    parser.add_argument("--show_plot", type=str2bool, default=True, help="Show live plot when possible.")
    parser.add_argument("--progress_bar", type=str2bool, default=True, help="Show tqdm progress bar.")
    return parser.parse_args()


def parse_header_value(text: str, key: str) -> float | None:
    import re

    match = re.search(rf"{key}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    return float(match.group(1)) if match else None


def read_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def load_table(path: Path) -> tuple[pd.DataFrame, PhysicsParams]:
    lines = read_text_lines(path)
    header_lines = [line for line in lines if line.strip().startswith("#")]
    header_text = " ".join(header_lines)
    params = PhysicsParams(
        m1=parse_header_value(header_text, "m1") or 1.0,
        m2=parse_header_value(header_text, "m2") or 1.0,
        l1=parse_header_value(header_text, "L1") or 1.0,
        l2=parse_header_value(header_text, "L2") or 1.0,
        g=parse_header_value(header_text, "g") or 9.8,
    )

    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=r"\s+", comment="#", encoding=encoding, engine="python")
            return df, params
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to parse table from {path}")


def read_states_from_file(path: Path) -> tuple[np.ndarray, np.ndarray, PhysicsParams]:
    df, params = load_table(path)
    columns = set(df.columns)
    time_key = "time_s" if "time_s" in columns else "tiempo"
    if time_key not in columns:
        raise ValueError(f"Missing time column in {path.name}")

    if {"theta1_rad", "theta2_rad", "omega1_rad_s", "omega2_rad_s"}.issubset(columns):
        theta1 = df["theta1_rad"].to_numpy(dtype=np.float32)
        theta2 = df["theta2_rad"].to_numpy(dtype=np.float32)
        omega1 = df["omega1_rad_s"].to_numpy(dtype=np.float32)
        omega2 = df["omega2_rad_s"].to_numpy(dtype=np.float32)
    elif {"theta1_deg", "theta2_deg", "omega1_deg_s", "omega2_deg_s"}.issubset(columns):
        theta1 = np.deg2rad(df["theta1_deg"].to_numpy(dtype=np.float32))
        theta2 = np.deg2rad(df["theta2_deg"].to_numpy(dtype=np.float32))
        omega1 = np.deg2rad(df["omega1_deg_s"].to_numpy(dtype=np.float32))
        omega2 = np.deg2rad(df["omega2_deg_s"].to_numpy(dtype=np.float32))
    else:
        raise ValueError(f"Unsupported angle columns in {path.name}")

    times = df[time_key].to_numpy(dtype=np.float32)
    states = np.stack([theta1, theta2, omega1, omega2], axis=1).astype(np.float32, copy=False)
    return states, times, params


def analytical_xdot_np(states: np.ndarray, physics: PhysicsParams) -> np.ndarray:
    t1 = states[:, 0]
    t2 = states[:, 1]
    w1 = states[:, 2]
    w2 = states[:, 3]

    a1 = (physics.l2 / physics.l1) * (physics.m2 / (physics.m1 + physics.m2)) * np.cos(t1 - t2)
    a2 = (physics.l1 / physics.l2) * np.cos(t1 - t2)
    f1 = (
        -(physics.l2 / physics.l1)
        * (physics.m2 / (physics.m1 + physics.m2))
        * (w2**2)
        * np.sin(t1 - t2)
        - (physics.g / physics.l1) * np.sin(t1)
    )
    f2 = (physics.l1 / physics.l2) * (w1**2) * np.sin(t1 - t2) - (physics.g / physics.l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1.0 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1.0 - a1 * a2)
    return np.stack([w1, w2, g1, g2], axis=1).astype(np.float32, copy=False)


def build_samples_for_file(
    path: Path,
    preset: TrainPreset,
    lookahead: int,
) -> tuple[np.ndarray, np.ndarray, PhysicsParams, float]:
    states, times, physics = read_states_from_file(path)
    if len(times) < 2:
        raise ValueError(f"{path.name} has too few rows")

    dt_median = float(np.median(np.diff(times)))
    if preset.objective == "continuous":
        x = wrap_coords_np(states).astype(np.float32, copy=False)
        y = analytical_xdot_np(states, physics)
        return x, y, physics, dt_median

    if len(states) <= lookahead:
        raise ValueError(f"{path.name} is too short for lookahead={lookahead}")

    x = wrap_coords_np(states[:-lookahead]).astype(np.float32, copy=False)
    y = states[lookahead:] - states[:-lookahead]
    effective_dt = float(np.median(times[lookahead:] - times[:-lookahead]))
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False), physics, effective_dt


def select_files(data_dir: Path, max_files: int) -> list[Path]:
    files = sorted(data_dir.glob("sim_data_*.txt"))
    if not files:
        raise FileNotFoundError(f"No sim_data_*.txt files found in {data_dir}")
    return files if max_files <= 0 else files[:max_files]


def sample_indices(total_size: int, keep_size: int, mode: str, seed: int) -> np.ndarray:
    if keep_size <= 0 or keep_size >= total_size:
        return np.arange(total_size, dtype=np.int64)
    if mode == "ordered":
        return np.arange(keep_size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_size, size=keep_size, replace=False))


def split_indices(total_size: int, test_ratio: float, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_test = max(1, int(round(total_size * test_ratio)))
    n_test = min(n_test, total_size - 1)
    if mode == "ordered":
        n_train = total_size - n_test
        return np.arange(n_train, dtype=np.int64), np.arange(n_train, total_size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_size)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def resolve_overrides(args: argparse.Namespace) -> TrainPreset:
    preset = PRESETS[args.preset]
    overrides = asdict(preset)
    if args.dataset_size is not None:
        overrides["dataset_size"] = int(args.dataset_size)
    if args.num_steps is not None:
        overrides["num_steps"] = int(args.num_steps)
    if args.batch_size is not None:
        overrides["batch_size"] = int(args.batch_size)
    if args.test_ratio is not None:
        overrides["test_ratio"] = float(args.test_ratio)
    if args.eval_every is not None:
        overrides["eval_every"] = int(args.eval_every)
    return TrainPreset(**overrides)


def load_dataset(args: argparse.Namespace, preset: TrainPreset, np_dtype: np.dtype) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    max_files = int(args.max_files)
    if preset.name == "paper" and max_files <= 0:
        max_files = 1
    files = select_files(data_dir=data_dir, max_files=max_files)

    reference_physics: PhysicsParams | None = None
    base_dt: float | None = None
    states_by_file: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for path in files:
        states, times, physics = read_states_from_file(path)
        if reference_physics is None:
            reference_physics = physics
        elif reference_physics != physics:
            raise ValueError("All files must share the same physics parameters for paper-like training.")
        if base_dt is None:
            base_dt = float(np.median(np.diff(times)))
        states_by_file.append((path, states.astype(np_dtype, copy=False), times.astype(np_dtype, copy=False)))

    assert reference_physics is not None
    assert base_dt is not None

    lookahead = 1
    if preset.objective == "delta":
        lookahead = max(1, int(round(preset.dt / base_dt)))

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []
    effective_dt = base_dt
    for path, _, _ in states_by_file:
        x_file, y_file, _, file_dt = build_samples_for_file(path=path, preset=preset, lookahead=lookahead)
        x_parts.append(x_file.astype(np_dtype, copy=False))
        y_parts.append(y_file.astype(np_dtype, copy=False))
        file_parts.append(np.array([path.name] * len(x_file), dtype=object))
        effective_dt = file_dt

    all_x = np.concatenate(x_parts, axis=0)
    all_y = np.concatenate(y_parts, axis=0)
    all_files = np.concatenate(file_parts, axis=0)

    keep_idx = sample_indices(
        total_size=len(all_x),
        keep_size=int(preset.dataset_size),
        mode=preset.sample_mode,
        seed=int(args.seed),
    )
    all_x = all_x[keep_idx]
    all_y = all_y[keep_idx]
    all_files = all_files[keep_idx]

    split_mode = "ordered" if preset.sample_mode == "ordered" else "random"
    train_idx, test_idx = split_indices(
        total_size=len(all_x),
        test_ratio=float(preset.test_ratio),
        mode=split_mode,
        seed=int(args.seed),
    )

    train_x = torch.from_numpy(all_x[train_idx])
    train_y = torch.from_numpy(all_y[train_idx])
    test_x = torch.from_numpy(all_x[test_idx])
    test_y = torch.from_numpy(all_y[test_idx])

    return {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "physics": reference_physics,
        "num_files": len(files),
        "effective_dt": float(effective_dt),
        "lookahead": int(lookahead),
        "train_files": sorted(np.unique(all_files[train_idx]).tolist()),
        "test_files": sorted(np.unique(all_files[test_idx]).tolist()),
        "source_files": [str(path) for path in files],
    }


def piecewise_lr(preset: TrainPreset, step: int) -> float:
    if preset.training_mode == "full_batch":
        one_third = preset.num_steps // 3
        two_third = (2 * preset.num_steps) // 3
        if step < one_third:
            return float(preset.lr)
        if step < two_third:
            return float(preset.lr / 10.0)
        return float(preset.lr / 100.0)

    if step < preset.num_steps // 2:
        return float(preset.lr)
    return float(preset.lr2)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def tree_l2_norm(module: torch.nn.Module) -> torch.Tensor:
    leaves = [param for param in module.parameters() if param.requires_grad]
    return sum(torch.sum(param * param) for param in leaves)


def loss_from_prediction(pred: torch.Tensor, target: torch.Tensor, preset: TrainPreset) -> torch.Tensor:
    if preset.loss_mode == "mse":
        return torch.mean((pred - target) ** 2)
    if preset.loss_mode == "l1_sum":
        return torch.sum(torch.abs(pred - target))
    raise ValueError(f"Unsupported loss_mode: {preset.loss_mode}")


def metric_from_prediction(pred: torch.Tensor, target: torch.Tensor, preset: TrainPreset) -> torch.Tensor:
    if preset.loss_mode == "mse":
        return torch.mean((pred - target) ** 2)
    if preset.loss_mode == "l1_sum":
        return torch.sum(torch.abs(pred - target)) / pred.shape[0]
    raise ValueError(f"Unsupported loss_mode: {preset.loss_mode}")


def predict_batch(
    model: torch.nn.Module,
    state_batch: torch.Tensor,
    preset: TrainPreset,
    for_training: bool,
    integration_dt: float | None = None,
) -> torch.Tensor:
    q = state_batch[:, :2]
    qdot = state_batch[:, 2:]
    if preset.objective == "continuous":
        return xdot_from_lagrangian(model=model, q=q, qdot=qdot, for_training=for_training)
    if preset.objective == "delta":
        dt = float(preset.dt if integration_dt is None else integration_dt)
        return state_delta_from_lagrangian(
            model=model,
            q=q,
            qdot=qdot,
            dt=dt,
            n_updates=preset.n_updates,
            for_training=for_training,
        )
    raise ValueError(f"Unsupported objective: {preset.objective}")


def init_live_plot(show_plot: bool):
    if show_plot:
        try:
            fig, ax = plt.subplots(figsize=(8.0, 4.5))
            plt.ion()
            fig.show()
            state = {"live_enabled": True, "window_closed": False}

            def _on_close(_event) -> None:
                state["live_enabled"] = False
                state["window_closed"] = True

            fig.canvas.mpl_connect("close_event", _on_close)
            return fig, ax, state
        except Exception as exc:
            print(f"Live plot disabled: {exc}")

    fig = Figure(figsize=(8.0, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    return fig, ax, {"live_enabled": False, "window_closed": False}


def update_live_plot(
    fig,
    ax,
    out_dir: Path,
    history: dict[str, list[dict[str, float | int]]],
    plot_state: dict[str, bool],
) -> None:
    steps = [int(row["step"]) for row in history["train"]]
    train_vals = [float(row["loss"]) for row in history["train"]]
    test_vals = [float(row["loss"]) for row in history["test"]]

    ax.clear()
    marker = "o" if len(steps) <= 1 else None
    ax.plot(steps, train_vals, label="train", color="#1f77b4", linewidth=2.0, marker=marker, markersize=5)
    ax.plot(steps, test_vals, label="test", color="#d62728", linewidth=2.0, marker=marker, markersize=5)
    ax.set_title("Paper-like LNN")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if len(steps) == 1:
        center = float(steps[0])
        ax.set_xlim(center - 1.0, center + 1.0)
    if train_vals and min(train_vals + test_vals) > 0:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    if plot_state["live_enabled"]:
        try:
            if hasattr(fig, "number") and not plt.fignum_exists(fig.number):
                plot_state["live_enabled"] = False
                plot_state["window_closed"] = True
                return
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception as exc:
            plot_state["live_enabled"] = False
            plot_state["window_closed"] = True
            print(f"Live plot disabled after window close/error: {exc}")


def write_metrics_csv(out_dir: Path, history: dict[str, list[dict[str, float | int]]]) -> None:
    rows = []
    for train_row, test_row in zip(history["train"], history["test"]):
        rows.append(
            {
                "step": int(train_row["step"]),
                "train_loss": float(train_row["loss"]),
                "test_loss": float(test_row["loss"]),
                "lr": float(train_row["lr"]),
                "elapsed_s": float(train_row["elapsed_s"]),
                "steps_per_s": float(train_row["steps_per_s"]),
            }
        )

    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "train_loss", "test_loss", "lr", "elapsed_s", "steps_per_s"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_best_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    preset: TrainPreset,
    dataset: dict[str, Any],
    step: int,
    best_loss: float,
    torch_dtype: torch.dtype,
) -> None:
    payload = {
        "step": int(step),
        "best_test_loss": float(best_loss),
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),
        "preset": asdict(preset),
        "physics": asdict(dataset["physics"]),
        "lookahead": int(dataset["lookahead"]),
        "effective_dt": float(dataset["effective_dt"]),
        "dtype": str(torch_dtype),
    }
    torch.save(payload, out_dir / "checkpoints" / "model_best.pth")


def save_monitoring_artifacts(
    out_dir: Path,
    history: dict[str, list[dict[str, float | int]]],
    fig,
    ax,
    plot_state: dict[str, bool],
) -> None:
    save_json(out_dir / "history.json", history)
    write_metrics_csv(out_dir, history)
    update_live_plot(fig, ax, out_dir, history, plot_state)


def evaluate_loss(
    model: torch.nn.Module,
    state_x: torch.Tensor,
    target_y: torch.Tensor,
    preset: TrainPreset,
    eval_batch_size: int,
    integration_dt: float | None = None,
) -> float:
    losses: list[torch.Tensor] = []
    count = state_x.shape[0]
    for start in range(0, count, eval_batch_size):
        end = min(count, start + eval_batch_size)
        pred = predict_batch(
            model=model,
            state_batch=state_x[start:end],
            preset=preset,
            for_training=False,
            integration_dt=integration_dt,
        )
        loss = metric_from_prediction(pred, target_y[start:end], preset=preset)
        weight = (end - start) / count
        losses.append(loss * weight)
    return float(torch.stack(losses).sum().item())


def full_batch_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    preset: TrainPreset,
    chunk_size: int,
    integration_dt: float | None = None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    total_count = train_x.shape[0]
    for start in range(0, total_count, chunk_size):
        end = min(total_count, start + chunk_size)
        pred = predict_batch(
            model=model,
            state_batch=train_x[start:end],
            preset=preset,
            for_training=True,
            integration_dt=integration_dt,
        )
        chunk_loss = loss_from_prediction(pred, train_y[start:end], preset=preset)
        if preset.loss_mode == "mse":
            scaled_loss = chunk_loss * ((end - start) / total_count)
        else:
            scaled_loss = chunk_loss
        scaled_loss.backward()

    if preset.l2reg > 0.0:
        reg = (preset.l2reg * tree_l2_norm(model)) / max(1, chunk_size)
        reg.backward()
    optimizer.step()


def minibatch_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    preset: TrainPreset,
    integration_dt: float | None = None,
) -> None:
    batch_size = min(int(preset.batch_size), int(train_x.shape[0]))
    indices = torch.randint(0, train_x.shape[0], (batch_size,), device=train_x.device)
    batch_x = train_x[indices]
    batch_y = train_y[indices]

    optimizer.zero_grad(set_to_none=True)
    pred = predict_batch(
        model=model,
        state_batch=batch_x,
        preset=preset,
        for_training=True,
        integration_dt=integration_dt,
    )
    loss = loss_from_prediction(pred, batch_y, preset=preset)
    if preset.l2reg > 0.0:
        loss = loss + (preset.l2reg * tree_l2_norm(model)) / batch_size
    loss.backward()
    optimizer.step()


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: dict[str, Any],
    preset: TrainPreset,
    out_dir: Path,
    torch_dtype: torch.dtype,
    show_plot: bool,
    progress_bar: bool,
) -> dict[str, Any]:
    fig, ax, plot_state = init_live_plot(show_plot=show_plot)
    history: dict[str, list[dict[str, float | int]]] = {"train": [], "test": []}
    best_loss = math.inf
    best_step = -1
    start_time = time.time()
    eval_batch_size = max(256, int(preset.batch_size))
    integration_dt = float(dataset["effective_dt"]) if preset.objective == "delta" else None
    close_notice_printed = False

    pbar = tqdm(range(int(preset.num_steps)), desc="Training", unit="step", dynamic_ncols=True) if progress_bar else None
    step_iter = pbar if pbar is not None else range(int(preset.num_steps))
    log = tqdm.write if pbar is not None else print

    for step in step_iter:
        current_lr = piecewise_lr(preset, step)
        set_lr(optimizer, current_lr)

        if preset.training_mode == "full_batch":
            full_batch_step(
                model=model,
                optimizer=optimizer,
                train_x=dataset["train_x"],
                train_y=dataset["train_y"],
                preset=preset,
                chunk_size=max(1, int(preset.batch_size)),
                integration_dt=integration_dt,
            )
        else:
            minibatch_step(
                model=model,
                optimizer=optimizer,
                train_x=dataset["train_x"],
                train_y=dataset["train_y"],
                preset=preset,
                integration_dt=integration_dt,
            )

        if step % int(preset.eval_every) == 0 or step == int(preset.num_steps) - 1:
            elapsed_s = max(time.time() - start_time, 1e-9)
            steps_per_s = float((step + 1) / elapsed_s)
            train_loss = evaluate_loss(
                model=model,
                state_x=dataset["train_x"],
                target_y=dataset["train_y"],
                preset=preset,
                eval_batch_size=eval_batch_size,
                integration_dt=integration_dt,
            )
            test_loss = evaluate_loss(
                model=model,
                state_x=dataset["test_x"],
                target_y=dataset["test_y"],
                preset=preset,
                eval_batch_size=eval_batch_size,
                integration_dt=integration_dt,
            )

            history["train"].append(
                {
                    "step": int(step),
                    "loss": float(train_loss),
                    "lr": float(current_lr),
                    "elapsed_s": float(elapsed_s),
                    "steps_per_s": float(steps_per_s),
                }
            )
            history["test"].append(
                {
                    "step": int(step),
                    "loss": float(test_loss),
                    "lr": float(current_lr),
                    "elapsed_s": float(elapsed_s),
                    "steps_per_s": float(steps_per_s),
                }
            )

            if math.isfinite(test_loss) and test_loss < best_loss:
                best_loss = float(test_loss)
                best_step = int(step)
                save_best_checkpoint(
                    out_dir=out_dir,
                    model=model,
                    preset=preset,
                    dataset=dataset,
                    step=step,
                    best_loss=best_loss,
                    torch_dtype=torch_dtype,
                )
                log(f"[BEST] step={step:06d} | test_loss={best_loss:.8e}")

            if pbar is not None:
                pbar.set_postfix(
                    train=f"{train_loss:.3e}",
                    test=f"{test_loss:.3e}",
                    best=f"{best_loss:.3e}" if math.isfinite(best_loss) else "inf",
                    lr=f"{current_lr:.2e}",
                    sps=f"{steps_per_s:.2f}",
                )
            log(
                f"step={step:06d}/{int(preset.num_steps):06d} | "
                f"train_loss={train_loss:.8e} | "
                f"test_loss={test_loss:.8e} | "
                f"lr={current_lr:.3e} | "
                f"elapsed_s={elapsed_s:.1f} | "
                f"steps_per_s={steps_per_s:.2f}"
            )
            save_monitoring_artifacts(out_dir=out_dir, history=history, fig=fig, ax=ax, plot_state=plot_state)
            if plot_state["window_closed"] and not close_notice_printed:
                log("Live plot window closed. Training continues headless and still saves loss_curve.png.")
                close_notice_printed = True

    if pbar is not None:
        pbar.close()
    try:
        plt.close(fig)
    except Exception:
        pass
    return {
        "history": history,
        "best_test_loss": float(best_loss),
        "best_step": int(best_step),
    }


def print_run_header(device: torch.device, torch_dtype: torch.dtype, preset: TrainPreset, dataset: dict[str, Any]) -> None:
    if device.type == "cuda":
        print(
            f"Using device: {device} | GPU: {torch.cuda.get_device_name(device)} | "
            f"torch={torch.__version__} | dtype={torch_dtype}"
        )
    else:
        print(f"Using device: {device} | torch={torch.__version__} | dtype={torch_dtype}")

    print(
        "Dataset summary: "
        f"files={dataset['num_files']} | "
        f"train_samples={int(dataset['train_x'].shape[0])} | "
        f"test_samples={int(dataset['test_x'].shape[0])} | "
        f"lookahead={int(dataset['lookahead'])} | "
        f"effective_dt={float(dataset['effective_dt']):.6f}"
    )
    print(
        "Preset summary: "
        f"name={preset.name} | "
        f"objective={preset.objective} | "
        f"loss={preset.loss_mode} | "
        f"mode={preset.training_mode} | "
        f"steps={preset.num_steps} | "
        f"batch_size={preset.batch_size} | "
        f"preset_dt={float(preset.dt):.6f} | "
        f"use_double={preset.use_double} | "
        f"hidden_dim={preset.hidden_dim} | "
        f"layers={preset.num_hidden_layers}"
    )
    if preset.objective == "delta":
        print(
            "Delta integration: "
            f"using effective_dt={float(dataset['effective_dt']):.6f} "
            f"from lookahead={int(dataset['lookahead'])} instead of raw preset_dt={float(preset.dt):.6f}"
        )


def main() -> None:
    args = parse_args()
    preset = resolve_overrides(args)
    set_seed(int(args.seed))

    device = select_device(str(args.device))
    use_double = bool(args.double or preset.use_double)
    torch_dtype = torch.float64 if use_double else torch.float32
    np_dtype = np.float64 if use_double else np.float32

    out_paths = ensure_output_dirs(Path(args.out_dir))
    out_dir = out_paths["out_dir"]
    dataset = load_dataset(args=args, preset=preset, np_dtype=np_dtype)

    dataset["train_x"] = dataset["train_x"].to(device=device, dtype=torch_dtype)
    dataset["train_y"] = dataset["train_y"].to(device=device, dtype=torch_dtype)
    dataset["test_x"] = dataset["test_x"].to(device=device, dtype=torch_dtype)
    dataset["test_y"] = dataset["test_y"].to(device=device, dtype=torch_dtype)

    print_run_header(device=device, torch_dtype=torch_dtype, preset=preset, dataset=dataset)

    model = LagrangianMLP(
        input_dim=4,
        hidden_dim=int(preset.hidden_dim),
        num_hidden_layers=int(preset.num_hidden_layers),
        activation=str(preset.activation),
        output_dim=int(preset.output_dim),
        init_seed=int(args.seed),
        init_mode="stax",
    ).to(device=device, dtype=torch_dtype)

    optimizer = Adam(model.parameters(), lr=float(preset.lr), weight_decay=0.0)
    result = train_model(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        preset=preset,
        out_dir=out_dir,
        torch_dtype=torch_dtype,
        show_plot=bool(args.show_plot),
        progress_bar=bool(args.progress_bar),
    )

    final_payload = {
        "step": int(preset.num_steps),
        "best_test_loss": float(result["best_test_loss"]),
        "best_step": int(result["best_step"]),
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),
        "preset": asdict(preset),
        "physics": asdict(dataset["physics"]),
        "lookahead": int(dataset["lookahead"]),
        "effective_dt": float(dataset["effective_dt"]),
        "dtype": str(torch_dtype),
    }
    torch.save(final_payload, out_dir / "model_final.pth")

    run_config = {
        "preset": asdict(preset),
        "device": str(device),
        "torch_version": torch.__version__,
        "double": bool(args.double),
        "effective_use_double": bool(use_double),
        "seed": int(args.seed),
        "data_dir": str(Path(args.data_dir)),
        "out_dir": str(out_dir),
        "summary": {
            "num_files": int(dataset["num_files"]),
            "train_samples": int(dataset["train_x"].shape[0]),
            "test_samples": int(dataset["test_x"].shape[0]),
            "train_files": dataset["train_files"],
            "test_files": dataset["test_files"],
            "lookahead": int(dataset["lookahead"]),
            "effective_dt": float(dataset["effective_dt"]),
            "preset_dt": float(preset.dt),
        },
        "best_test_loss": float(result["best_test_loss"]),
        "best_step": int(result["best_step"]),
        "model_best_path": str(out_dir / "checkpoints" / "model_best.pth"),
        "model_final_path": str(out_dir / "model_final.pth"),
        "metrics_csv": str(out_dir / "metrics.csv"),
        "plot_path": str(out_dir / "loss_curve.png"),
        "physics": asdict(dataset["physics"]),
        "source_files": dataset["source_files"],
    }
    save_json(out_dir / "run_config.json", run_config)

    print(f"Training complete. Final model: {out_dir / 'model_final.pth'}")
    print(f"Best model: {out_dir / 'checkpoints' / 'model_best.pth'}")


if __name__ == "__main__":
    main()
