from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from lnn.dynamics import state_delta_from_lagrangian, xdot_from_lagrangian
from lnn.plotting import plot_loss_curves
from lnn.utils import ensure_output_dirs, set_seed
from paperlike_double_pendulum import load_paperlike_temporal_cache
from torch_delta_models import StructuredTransformerLagrangian, TransformerLagrangian


LOCAL_ROOT = Path(__file__).resolve().parent


def configure_torch_attention_backends() -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer-LNN on the official double-pendulum cache using PyTorch.")
    parser.add_argument("--out_dir", type=str, default="experiments/compare_transformer_lnn_torch")
    parser.add_argument("--reference_out_dir", type=str, default="experiments/official_jax_cpu_retry")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument(
        "--sweep_all_attempts",
        type=lambda s: str(s).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Run all seeds up to max_attempts instead of stopping after the first completed attempt.",
    )
    parser.add_argument("--objective", choices=["auto", "delta", "xdot"], default="auto")
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr2", type=float, default=1.0e-5)
    parser.add_argument("--l2reg", type=float, default=1.0e-2)
    parser.add_argument("--dt", type=float, default=0.09609870774790222)
    parser.add_argument("--n_updates", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--ff_multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lagrangian_form", choices=["structured_tv", "free"], default="structured_tv")
    parser.add_argument("--mass_eps", type=float, default=1.0e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_abort_threshold", type=float, default=1.0e4)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--warmup_eval_every", type=int, default=100)
    parser.add_argument("--warmup_eval_until", type=int, default=1000)
    parser.add_argument("--quick_eval_every", type=int, default=100)
    parser.add_argument("--quick_eval_until", type=int, default=1000)
    parser.add_argument(
        "--final_eval_only",
        type=lambda s: str(s).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Skip all intermediate evaluations and run a single full evaluation at the end.",
    )
    parser.add_argument("--final_eval_mode", choices=["full", "quick", "none"], default="full")
    parser.add_argument("--monitor_train_samples", type=int, default=256)
    parser.add_argument("--monitor_test_samples", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_train_samples", type=int, default=100000)
    parser.add_argument("--eval_test_samples", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        return torch.device("cpu")
    return torch.device(requested)


def wrap_coords_np(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    y[:, :2] = (y[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
    return y


def resolve_reference_paths(reference_out_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    attempts_summary_path = reference_out_dir / "results" / "attempts_summary.json"
    if not attempts_summary_path.exists():
        raise FileNotFoundError(f"Could not find reference summary: {attempts_summary_path}")
    attempts_summary = json.loads(attempts_summary_path.read_text(encoding="utf-8"))
    dataset_summary = attempts_summary["dataset_summary"]
    return attempts_summary, dataset_summary


def load_dataset(cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.is_dir():
        data_np, _ = load_paperlike_temporal_cache(cache_path)
        return data_np

    cached = np.load(cache_path)
    data_np = {
        "x": np.asarray(cached["x"], dtype=np.float32),
        "test_x": np.asarray(cached["test_x"], dtype=np.float32),
    }
    if "dx" in cached:
        data_np["dx"] = np.asarray(cached["dx"], dtype=np.float32)
        data_np["test_dx"] = np.asarray(cached["test_dx"], dtype=np.float32)
    if "xdot" in cached:
        data_np["xdot"] = np.asarray(cached["xdot"], dtype=np.float32)
        data_np["test_xdot"] = np.asarray(cached["test_xdot"], dtype=np.float32)
    return data_np


def resolve_objective(args: argparse.Namespace, dataset_summary: dict[str, object]) -> str:
    if str(args.objective) != "auto":
        return str(args.objective)
    summary_objective = dataset_summary.get("objective")
    if summary_objective:
        return str(summary_objective)
    return "delta"


def current_lr(step: int, total_steps: int, lr1: float, lr2: float) -> float:
    return float(lr1 if step < total_steps // 2 else lr2)


def apply_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def tree_l2_norm(module: torch.nn.Module) -> torch.Tensor:
    leaves = [param for param in module.parameters() if param.requires_grad]
    return sum(torch.sum(param * param) for param in leaves)


def compute_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad_sq = float(torch.sum(param.grad.detach() * param.grad.detach()).item())
        total += grad_sq
    return float(math.sqrt(max(total, 0.0)))


def loss_l1_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(pred - target))


def metric_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(pred - target)) / max(1, pred.shape[0])


def build_monitor_indices(total_size: int, monitor_samples: int) -> np.ndarray:
    count = max(1, min(int(monitor_samples), int(total_size)))
    if count >= total_size:
        return np.arange(total_size, dtype=np.int64)
    idx = np.linspace(0, total_size - 1, count, dtype=np.float64)
    return np.round(idx).astype(np.int64, copy=False)


def torch_batch_from_numpy(state_x: np.ndarray, target_y: np.ndarray | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    state_batch = torch.from_numpy(wrap_coords_np(state_x)).to(device=device, dtype=torch.float32)
    if target_y is None:
        return state_batch, None
    target_batch = torch.from_numpy(np.asarray(target_y, dtype=np.float32).copy()).to(device=device, dtype=torch.float32)
    return state_batch, target_batch


def predict_target(
    model: torch.nn.Module,
    state_batch: torch.Tensor,
    dt: float,
    n_updates: int,
    for_training: bool,
    objective: str,
) -> torch.Tensor:
    q = state_batch[:, :2]
    qdot = state_batch[:, 2:]
    if str(objective) == "xdot":
        return xdot_from_lagrangian(model=model, q=q, qdot=qdot, for_training=for_training)
    return state_delta_from_lagrangian(
        model=model,
        q=q,
        qdot=qdot,
        dt=float(dt),
        n_updates=int(n_updates),
        for_training=for_training,
    )


def evaluate_l1(
    model: torch.nn.Module,
    state_x: np.ndarray,
    target_y: np.ndarray,
    dt: float,
    n_updates: int,
    eval_batch_size: int,
    objective: str,
    device: torch.device,
) -> tuple[float, int]:
    total_count = int(state_x.shape[0])
    batch_size = max(1, min(int(eval_batch_size), total_count))

    while True:
        try:
            weighted_loss_sum = 0.0
            for start in range(0, total_count, batch_size):
                end = min(total_count, start + batch_size)
                state_batch, target_batch = torch_batch_from_numpy(state_x[start:end], target_y[start:end], device=device)
                pred = predict_target(
                    model=model,
                    state_batch=state_batch,
                    dt=float(dt),
                    n_updates=int(n_updates),
                    for_training=False,
                    objective=str(objective),
                )
                loss = metric_l1_per_sample(pred, target_batch)
                weight = (end - start) / total_count
                weighted_loss_sum += float(loss.detach().item()) * weight
            return float(weighted_loss_sum), int(batch_size)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message or batch_size <= 1:
                raise
            next_batch_size = max(1, batch_size // 2)
            print(
                f"[eval] CUDA OOM with eval_batch_size={batch_size}. "
                f"Retrying with eval_batch_size={next_batch_size}."
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = next_batch_size


def eval_mode_for_step(
    step: int,
    eval_every: int,
    num_steps: int,
    quick_eval_every: int,
    quick_eval_until: int,
    final_eval_only: bool,
    final_eval_mode: str,
) -> str | None:
    if bool(final_eval_only):
        return None if step != num_steps - 1 else (None if str(final_eval_mode) == "none" else str(final_eval_mode))
    if step == num_steps - 1:
        return None if str(final_eval_mode) == "none" else str(final_eval_mode)
    if step == 0:
        return "quick"
    if step % int(eval_every) == 0:
        return "full"
    if int(quick_eval_every) > 0 and step < int(quick_eval_until) and step % int(quick_eval_every) == 0:
        return "quick"
    return None


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
                "train_step_s": float(train_row.get("train_step_s", float("nan"))),
                "eval_s": float(train_row.get("eval_s", float("nan"))),
                "total_step_s": float(train_row.get("total_step_s", float("nan"))),
                "eval_mode": str(train_row.get("eval_mode", "")),
            }
        )

    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "train_loss", "test_loss", "lr", "elapsed_s", "steps_per_s", "train_step_s", "eval_s", "total_step_s", "eval_mode"],
        )
        writer.writeheader()
        writer.writerows(rows)


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    architecture = "structured_transformer_lagrangian_tv" if str(args.lagrangian_form) == "structured_tv" else "transformer_lagrangian"
    return {
        "architecture": architecture,
        "d_model": int(args.d_model),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "ff_multiplier": int(args.ff_multiplier),
        "dropout": float(args.dropout),
        "num_tokens": 2 if str(args.lagrangian_form) == "structured_tv" else 4,
        "output_dim": 1,
        "lagrangian_form": str(args.lagrangian_form),
        "mass_eps": float(args.mass_eps),
    }


def create_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if str(args.lagrangian_form) == "structured_tv":
        model = StructuredTransformerLagrangian(
            d_model=int(args.d_model),
            num_heads=int(args.num_heads),
            num_layers=int(args.num_layers),
            ff_multiplier=int(args.ff_multiplier),
            dropout=float(args.dropout),
            mass_eps=float(args.mass_eps),
        )
    else:
        model = TransformerLagrangian(
            d_model=int(args.d_model),
            num_heads=int(args.num_heads),
            num_layers=int(args.num_layers),
            ff_multiplier=int(args.ff_multiplier),
            dropout=float(args.dropout),
        )
    return model.to(device)


def checkpoint_payload(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataset_summary: dict[str, object],
    reference_out_dir: Path,
) -> dict[str, object]:
    return {
        "framework": "torch",
        "model_kind": "transformer_lnn_torch",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.get_config(),
        "train_args": {
            "dt": float(args.dt),
            "n_updates": int(args.n_updates),
            "batch_size": int(args.batch_size),
            "loss": "l1_sum",
            "objective": str(resolve_objective(args, dataset_summary)),
            "num_steps": int(args.num_steps),
            "lr": float(args.lr),
            "lr2": float(args.lr2),
            "l2reg": float(args.l2reg),
            "grad_clip": float(args.grad_clip),
            "loss_abort_threshold": float(args.loss_abort_threshold),
            "lagrangian_form": str(args.lagrangian_form),
            "mass_eps": float(args.mass_eps),
        },
        "dataset_summary": dataset_summary,
        "reference_out_dir": str(reference_out_dir),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataset_summary: dict[str, object],
    reference_out_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        torch.save(checkpoint_payload(model, args, dataset_summary, reference_out_dir), fh)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_attempt(
    base_out_dir: Path,
    args: argparse.Namespace,
    dataset: dict[str, np.ndarray],
    dataset_summary: dict[str, object],
    reference_out_dir: Path,
    attempt_idx: int,
    attempt_seed: int,
    device: torch.device,
) -> dict[str, object]:
    attempt_name = f"attempt_{attempt_idx:02d}_seed_{attempt_seed:04d}"
    attempt_dir = base_out_dir / "attempts" / attempt_name
    paths = ensure_output_dirs(attempt_dir)

    set_seed(int(attempt_seed))
    model = create_model(args, device)
    optimizer = Adam(model.parameters(), lr=float(args.lr))

    train_x = dataset["x"]
    objective = resolve_objective(args, dataset_summary)
    train_y = dataset["xdot"] if objective == "xdot" else dataset["dx"]
    test_x = dataset["test_x"]
    test_y = dataset["test_xdot"] if objective == "xdot" else dataset["test_dx"]
    np_rng = np.random.default_rng(int(attempt_seed))

    history: dict[str, list[dict[str, float | int]]] = {"train": [], "test": []}
    best_loss = math.inf
    best_step = -1
    best_state = None
    eval_batch_size = max(1, int(args.eval_batch_size))
    exception_text: str | None = None

    train_monitor_idx = build_monitor_indices(
        total_size=int(train_x.shape[0]),
        monitor_samples=int(args.eval_train_samples if int(args.eval_train_samples) > 0 else args.monitor_train_samples),
    )
    test_monitor_idx = build_monitor_indices(
        total_size=int(test_x.shape[0]),
        monitor_samples=int(args.eval_test_samples if int(args.eval_test_samples) > 0 else args.monitor_test_samples),
    )

    log_path = paths["results"] / "training.log"
    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_fh, contextlib.redirect_stdout(Tee(sys.stdout, log_fh)):
        print(
            f"[attempt] index={attempt_idx} seed={attempt_seed} model_kind=transformer_lnn_torch "
            f"| lagrangian_form={args.lagrangian_form}"
        )
        try:
            with tqdm(
                total=int(args.num_steps),
                desc=attempt_name,
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            ) as progress:
                for step in range(int(args.num_steps)):
                    step_start = time.perf_counter()
                    model.train()
                    lr_value = current_lr(step, int(args.num_steps), float(args.lr), float(args.lr2))
                    apply_lr(optimizer, lr_value)

                    idx = np_rng.integers(0, int(train_x.shape[0]), size=int(args.batch_size), endpoint=False)
                    batch_x, batch_y = torch_batch_from_numpy(train_x[idx], train_y[idx], device=device)

                    optimizer.zero_grad(set_to_none=True)
                    pred = predict_target(
                        model=model,
                        state_batch=batch_x,
                        dt=float(args.dt),
                        n_updates=int(args.n_updates),
                        for_training=True,
                        objective=str(objective),
                    )
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError(f"Non-finite prediction at train step {step}")
                    data_loss = loss_l1_sum(pred, batch_y)
                    if not torch.isfinite(data_loss):
                        raise FloatingPointError(f"Non-finite batch_loss at train step {step}")
                    if float(data_loss.detach().item()) > float(args.loss_abort_threshold):
                        raise FloatingPointError(
                            f"Exploding batch_loss at train step {step}: "
                            f"{float(data_loss.detach().item()):.6f} > {float(args.loss_abort_threshold):.6f}"
                        )
                    loss = data_loss + (float(args.l2reg) * tree_l2_norm(model)) / max(1, int(args.batch_size))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite total loss at train step {step}")
                    loss.backward()
                    grad_norm = compute_grad_norm(model)
                    if not math.isfinite(grad_norm):
                        raise FloatingPointError(f"Non-finite grad_norm at train step {step}")
                    if float(args.grad_clip) > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                    optimizer.step()

                    train_step_s = time.perf_counter() - step_start
                    if step % 100 == 0 or step == int(args.num_steps) - 1:
                        progress.set_postfix(
                            step=f"{step:06d}",
                            batch_loss=f"{float(data_loss.item()):.4f}",
                            lr=f"{lr_value:.1e}",
                        )
                    if int(args.log_every) > 0 and step % int(args.log_every) == 0:
                        print(
                            f"train_step={step:06d}, batch_loss={float(data_loss.item()):.6f}, "
                            f"grad_norm={grad_norm:.4f}, lr={lr_value:.6e}"
                        )

                    eval_mode = eval_mode_for_step(
                        step=step,
                        eval_every=int(args.eval_every),
                        num_steps=int(args.num_steps),
                        quick_eval_every=int(args.quick_eval_every),
                        quick_eval_until=int(args.quick_eval_until),
                        final_eval_only=bool(args.final_eval_only),
                        final_eval_mode=str(args.final_eval_mode),
                    )
                    if eval_mode is not None:
                        eval_start = time.perf_counter()
                        eval_train_x = train_x
                        eval_train_y = train_y
                        eval_test_x = test_x
                        eval_test_y = test_y
                        if eval_mode == "quick" or int(args.eval_train_samples) > 0 or int(args.eval_test_samples) > 0:
                            eval_train_x = eval_train_x[train_monitor_idx]
                            eval_train_y = eval_train_y[train_monitor_idx]
                            eval_test_x = eval_test_x[test_monitor_idx]
                            eval_test_y = eval_test_y[test_monitor_idx]

                        train_loss, used_train_eval_batch = evaluate_l1(
                            model=model,
                            state_x=eval_train_x,
                            target_y=eval_train_y,
                            dt=float(args.dt),
                            n_updates=int(args.n_updates),
                            eval_batch_size=eval_batch_size,
                            objective=str(objective),
                            device=device,
                        )
                        test_loss, used_test_eval_batch = evaluate_l1(
                            model=model,
                            state_x=eval_test_x,
                            target_y=eval_test_y,
                            dt=float(args.dt),
                            n_updates=int(args.n_updates),
                            eval_batch_size=eval_batch_size,
                            objective=str(objective),
                            device=device,
                        )
                        eval_batch_size = min(int(used_train_eval_batch), int(used_test_eval_batch))
                        eval_s = time.perf_counter() - eval_start
                        elapsed_s = time.perf_counter() - start_time
                        steps_per_s = float((step + 1) / max(elapsed_s, 1e-9))
                        total_step_s = time.perf_counter() - step_start
                        eta_s = (elapsed_s / max(1, step + 1)) * max(0, int(args.num_steps) - (step + 1))

                        history["train"].append(
                            {
                                "step": int(step),
                                "loss": float(train_loss),
                                "lr": float(lr_value),
                                "elapsed_s": float(elapsed_s),
                                "steps_per_s": float(steps_per_s),
                                "train_step_s": float(train_step_s),
                                "eval_s": float(eval_s),
                                "total_step_s": float(total_step_s),
                                "eval_mode": str(eval_mode),
                            }
                        )
                        history["test"].append(
                            {
                                "step": int(step),
                                "loss": float(test_loss),
                                "lr": float(lr_value),
                                "elapsed_s": float(elapsed_s),
                                "steps_per_s": float(steps_per_s),
                                "train_step_s": float(train_step_s),
                                "eval_s": float(eval_s),
                                "total_step_s": float(total_step_s),
                                "eval_mode": str(eval_mode),
                            }
                        )
                        write_metrics_csv(attempt_dir, history)

                        progress.set_postfix(
                            step=f"{step:06d}",
                            train_loss=f"{train_loss:.4f}",
                            test_loss=f"{test_loss:.4f}",
                        )

                        print(
                            f"step={step:06d}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}, "
                            f"lr={lr_value:.6e}, eval_mode={eval_mode}, eval_batch_size={eval_batch_size}, "
                            f"step_s={total_step_s:.1f}, eta_m={eta_s / 60.0:.1f}"
                        )

                        if eval_mode == "full" and np.isfinite(test_loss) and test_loss < best_loss:
                            best_loss = float(test_loss)
                            best_step = int(step)
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            best_model_live = create_model(args, device=torch.device("cpu"))
                            best_model_live.load_state_dict(best_state)
                            # Persist the best checkpoint immediately so abrupt terminal closes do not lose it.
                            save_checkpoint(paths["checkpoints"] / "model_best.pth", best_model_live, args, dataset_summary, reference_out_dir)
                            save_checkpoint(base_out_dir / "checkpoints" / "model_best.pth", best_model_live, args, dataset_summary, reference_out_dir)
                            print(f"[BEST] step={step:06d} | test_loss={test_loss:.8e}")

                        if not np.isfinite(test_loss):
                            raise FloatingPointError(f"Non-finite test loss at step {step}")

                    progress.update(1)
        except Exception:
            exception_text = traceback.format_exc()
            print(exception_text, flush=True)

    elapsed_s = time.perf_counter() - start_time

    if best_state is None:
        if history["test"]:
            best_loss = float(min(float(row["loss"]) for row in history["test"]))
            best_candidates = [row for row in history["test"] if float(row["loss"]) == best_loss]
            best_step = int(best_candidates[-1]["step"])
        else:
            best_loss = float("nan")
            best_step = int(args.num_steps) - 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    best_model = create_model(args, device=torch.device("cpu"))
    best_model.load_state_dict(best_state)
    final_model = create_model(args, device=torch.device("cpu"))
    final_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})

    save_checkpoint(paths["checkpoints"] / "model_best.pth", best_model, args, dataset_summary, reference_out_dir)
    save_checkpoint(attempt_dir / "model_final.pth", final_model, args, dataset_summary, reference_out_dir)
    save_loss_artifacts(history, paths["results"], paths["plots"])

    completed = exception_text is None
    summary = {
        "attempt_index": int(attempt_idx),
        "attempt_name": attempt_name,
        "seed": int(attempt_seed),
        "attempt_dir": str(attempt_dir),
        "framework": "torch",
        "model_kind": "transformer_lnn_torch",
        "completed": bool(completed),
        "stop_reason": "completed" if completed else "exception",
        "stop_iteration": int(args.num_steps - 1 if completed else history["train"][-1]["step"] if history["train"] else -1),
        "best_iteration": int(best_step),
        "best_loss": float(best_loss),
        "elapsed_s": float(elapsed_s),
        "num_train_samples": int(train_x.shape[0]),
        "num_test_samples": int(test_x.shape[0]),
        "logged_points": int(len(history["train"])),
        "exception": exception_text,
        "files": {
            "model_best": str(paths["checkpoints"] / "model_best.pth"),
            "model_final": str(attempt_dir / "model_final.pth"),
            "training_log": str(log_path),
            "loss_csv": str(paths["results"] / "loss_history.csv"),
            "loss_plot": str(paths["plots"] / "loss_curve.png"),
        },
    }
    (paths["results"] / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_loss_artifacts(history: dict[str, list[dict[str, float | int]]], results_dir: Path, plots_dir: Path) -> None:
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
                "train_step_s": float(train_row.get("train_step_s", float("nan"))),
                "eval_s": float(train_row.get("eval_s", float("nan"))),
                "total_step_s": float(train_row.get("total_step_s", float("nan"))),
                "eval_mode": str(train_row.get("eval_mode", "")),
            }
        )

    csv_path = results_dir / "loss_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "train_loss", "test_loss", "lr", "elapsed_s", "steps_per_s", "train_step_s", "eval_s", "total_step_s", "eval_mode"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        plot_loss_curves(csv_path, plots_dir / "loss_curve.png")


def promote_best_attempt(base_paths: dict[str, Path], best_attempt: dict[str, object]) -> None:
    attempt_dir = Path(str(best_attempt["attempt_dir"]))
    copy_if_exists(attempt_dir / "checkpoints" / "model_best.pth", base_paths["checkpoints"] / "model_best.pth")
    copy_if_exists(attempt_dir / "model_final.pth", base_paths["out_dir"] / "model_final.pth")
    copy_if_exists(attempt_dir / "results" / "training.log", base_paths["results"] / "training.log")
    copy_if_exists(attempt_dir / "results" / "loss_history.csv", base_paths["results"] / "loss_history.csv")
    copy_if_exists(attempt_dir / "plots" / "loss_curve.png", base_paths["plots"] / "loss_curve.png")
    copy_if_exists(attempt_dir / "results" / "train_summary.json", base_paths["results"] / "best_attempt_summary.json")


def main() -> None:
    args = parse_args()
    configure_torch_attention_backends()
    if args.smoke:
        args.num_steps = 20
        args.max_attempts = 1
        args.batch_size = 8
        args.d_model = 64
        args.num_layers = 1
        args.eval_every = 10
        args.warmup_eval_every = 5
        args.warmup_eval_until = 10
        args.quick_eval_every = 5
        args.quick_eval_until = 10
        args.final_eval_mode = "quick"
        args.monitor_train_samples = 64
        args.monitor_test_samples = 64
        args.eval_batch_size = 4
        args.grad_clip = 1.0
        args.loss_abort_threshold = 1.0e4

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (LOCAL_ROOT / out_dir).resolve()
    reference_out_dir = Path(args.reference_out_dir)
    if not reference_out_dir.is_absolute():
        reference_out_dir = (LOCAL_ROOT / reference_out_dir).resolve()

    paths = ensure_output_dirs(out_dir)
    (out_dir / "attempts").mkdir(parents=True, exist_ok=True)

    attempts_summary, dataset_summary = resolve_reference_paths(reference_out_dir)
    cache_path = Path(dataset_summary["cache_path"])
    device = resolve_device(args.device)

    run_config = {
        "framework": "torch",
        "mode": "transformer_lnn_torch",
        "reference_out_dir": str(reference_out_dir),
        "reference_best_attempt": attempts_summary["best_attempt"]["attempt_name"],
        "dataset_summary": dataset_summary,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "args": vars(args),
        "model_config": model_config_from_args(args),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if device.type == "cuda":
        print(f"Using device: {device} | GPU: {torch.cuda.get_device_name(device)} | torch={torch.__version__}")
    else:
        print(f"Using device: {device} | torch={torch.__version__}")
    print(f"[dataset] loading cache: {cache_path}")
    dataset = load_dataset(cache_path)
    print(f"[dataset] ready | train_samples={dataset['x'].shape[0]} | test_samples={dataset['test_x'].shape[0]}")
    print(
        "Transformer-LNN config: "
        f"lagrangian_form={args.lagrangian_form} | "
        f"d_model={args.d_model} | heads={args.num_heads} | layers={args.num_layers} | "
        f"ff_multiplier={args.ff_multiplier} | dt={args.dt:.8f} | n_updates={args.n_updates} | "
        f"steps={args.num_steps} | batch_size={args.batch_size}"
    )

    attempts: list[dict[str, object]] = []
    best_attempt: dict[str, object] | None = None
    total_start = time.perf_counter()

    for attempt_idx in range(max(1, int(args.max_attempts))):
        attempt_seed = int(args.seed) + attempt_idx * int(args.seed_stride)
        print(f"[attempt {attempt_idx + 1}/{max(1, int(args.max_attempts))}] seed={attempt_seed}")
        summary = run_attempt(
            base_out_dir=out_dir,
            args=args,
            dataset=dataset,
            dataset_summary=dataset_summary,
            reference_out_dir=reference_out_dir,
            attempt_idx=attempt_idx,
            attempt_seed=attempt_seed,
            device=device,
        )
        attempts.append(summary)
        if np.isfinite(summary["best_loss"]) and (best_attempt is None or float(summary["best_loss"]) < float(best_attempt["best_loss"])):
            best_attempt = summary

        print(
            f"[attempt {attempt_idx + 1}] stop_reason={summary['stop_reason']} | "
            f"best_iteration={summary['best_iteration']} | best_loss={float(summary['best_loss']):.8e}"
        )
        if bool(summary["completed"]) and not bool(args.sweep_all_attempts):
            print("[attempt] completed full schedule. Stopping retries.")
            break
        if attempt_idx + 1 < int(args.max_attempts):
            if bool(args.sweep_all_attempts):
                print("[attempt] moving to next scheduled seed.")
            else:
                print("[attempt] retrying with next seed.")

    total_elapsed = time.perf_counter() - total_start
    attempts_summary_payload = {
        "framework": "torch",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "reference_out_dir": str(reference_out_dir),
        "dataset_summary": dataset_summary,
        "total_elapsed_s": float(total_elapsed),
        "num_attempts_run": int(len(attempts)),
        "best_attempt": best_attempt,
        "attempts": attempts,
    }
    (paths["results"] / "attempts_summary.json").write_text(json.dumps(attempts_summary_payload, indent=2), encoding="utf-8")

    attempts_csv = paths["results"] / "attempts_summary.csv"
    with attempts_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "attempt_index",
                "attempt_name",
                "seed",
                "completed",
                "stop_reason",
                "best_iteration",
                "best_loss",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        for attempt in attempts:
            writer.writerow(
                {
                    "attempt_index": int(attempt["attempt_index"]),
                    "attempt_name": str(attempt["attempt_name"]),
                    "seed": int(attempt["seed"]),
                    "completed": bool(attempt["completed"]),
                    "stop_reason": str(attempt["stop_reason"]),
                    "best_iteration": int(attempt["best_iteration"]),
                    "best_loss": float(attempt["best_loss"]),
                    "elapsed_s": float(attempt["elapsed_s"]),
                }
            )

    if best_attempt is None:
        raise RuntimeError("No valid attempt was produced.")
    promote_best_attempt(paths, best_attempt)

    print("Framework: torch")
    print(f"Device: {device}")
    print(f"Total elapsed seconds: {total_elapsed:.1f}")
    print(f"Attempts run: {len(attempts)}")
    print(f"Best attempt: {best_attempt['attempt_name']}")
    print(f"Best loss: {float(best_attempt['best_loss']):.8e}")
    print(f"Best params: {paths['checkpoints'] / 'model_best.pth'}")
    print(f"Training log: {paths['results'] / 'training.log'}")
    print(f"Loss CSV: {paths['results'] / 'loss_history.csv'}")
    print(f"Loss plot: {paths['plots'] / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
