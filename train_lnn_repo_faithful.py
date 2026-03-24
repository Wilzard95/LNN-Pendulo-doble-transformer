from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from tqdm import tqdm

from lnn.dynamics import state_delta_from_lagrangian
from lnn.model import LagrangianMLP
from lnn.plotting import plot_loss_curves
from lnn.repo_faithful_data import (
    RepoDoublePendulumPhysics,
    RepoFaithfulDataConfig,
    build_repo_faithful_dataset,
    save_dataset_metadata,
)
from lnn.utils import DEFAULT_OUT_DIR, ensure_output_dirs, save_json, select_device, set_seed


@dataclass
class RepoFaithfulTrainPreset:
    name: str
    hidden_dim: int
    num_hidden_layers: int
    activation: str
    output_dim: int
    batch_size: int
    lr: float
    lr2: float
    l2reg: float
    dt: float
    n_updates: int
    num_steps: int
    eval_every: int
    use_double: bool


PRESETS = {
    # This comes from the published DoublePendulum notebook comment block.
    "repo_notebook_best": RepoFaithfulTrainPreset(
        name="repo_notebook_best",
        hidden_dim=596,
        num_hidden_layers=4,
        activation="softplus",
        output_dim=1,
        batch_size=27,
        lr=0.005516656601005163,
        lr2=1.897157209816416e-05,
        l2reg=0.24927677946969878,
        dt=0.09609870774790222,
        n_updates=4,
        num_steps=40000,
        eval_every=1000,
        use_double=False,
    ),
    # Kept only as a fallback reference from the earlier local port.
    "repo_local_guess": RepoFaithfulTrainPreset(
        name="repo_local_guess",
        hidden_dim=159,
        num_hidden_layers=4,
        activation="softplus",
        output_dim=1,
        batch_size=41,
        lr=0.005491648617408025,
        lr2=1.9651742128096374e-05,
        l2reg=0.14544610264965038,
        dt=0.08986129562057266,
        n_updates=6,
        num_steps=40000,
        eval_every=1000,
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


class ProgressHeartbeat:
    def __init__(self, label: str, interval_s: float, writer) -> None:
        self.label = str(label)
        self.interval_s = max(1.0, float(interval_s))
        self.writer = writer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def __enter__(self) -> "ProgressHeartbeat":
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            elapsed_s = time.time() - self._start_time
            self.writer(f"[heartbeat] {self.label} still running | elapsed_s={elapsed_s:.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a repo-faithful LNN attempt on synthetic double pendulum data.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="repo_notebook_best")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR / "experiments" / "repo_faithful"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0, help="Training seed.")
    parser.add_argument("--data_seed", type=int, default=0, help="Synthetic data seed.")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--duration_s", type=float, default=50.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr2", type=float, default=None)
    parser.add_argument("--l2reg", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--n_updates", type=int, default=None)
    parser.add_argument("--double", type=str2bool, default=None)
    parser.add_argument("--progress_bar", type=str2bool, default=True)
    parser.add_argument("--heartbeat_interval_s", type=float, default=0.0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--monitor_train_samples", type=int, default=512)
    parser.add_argument("--monitor_test_samples", type=int, default=512)
    parser.add_argument("--quick_eval_every", type=int, default=100)
    parser.add_argument("--quick_eval_until", type=int, default=1000)
    return parser.parse_args()


def resolve_preset(args: argparse.Namespace) -> RepoFaithfulTrainPreset:
    preset = PRESETS[str(args.preset)]
    overrides = asdict(preset)
    for key in ("num_steps", "eval_every", "batch_size", "hidden_dim", "num_hidden_layers", "lr", "lr2", "l2reg", "dt", "n_updates"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.double is not None:
        overrides["use_double"] = bool(args.double)
    return RepoFaithfulTrainPreset(**overrides)


def piecewise_lr(preset: RepoFaithfulTrainPreset, step: int) -> float:
    if int(step) < int(preset.num_steps) // 2:
        return float(preset.lr)
    return float(preset.lr2)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def tree_l2_norm(module: torch.nn.Module) -> torch.Tensor:
    leaves = [param for param in module.parameters() if param.requires_grad]
    return sum(torch.sum(param * param) for param in leaves)


def loss_l1_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(pred - target))


def metric_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(pred - target)) / max(1, pred.shape[0])


def predict_delta(
    model: torch.nn.Module,
    state_batch: torch.Tensor,
    dt: float,
    n_updates: int,
    for_training: bool,
) -> torch.Tensor:
    q = state_batch[:, :2]
    qdot = state_batch[:, 2:]
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
    state_x: torch.Tensor,
    target_y: torch.Tensor,
    dt: float,
    n_updates: int,
    eval_batch_size: int,
) -> tuple[float, int]:
    total_count = int(state_x.shape[0])
    batch_size = max(1, min(int(eval_batch_size), total_count))

    while True:
        try:
            weighted_loss_sum = 0.0
            for start in range(0, total_count, batch_size):
                end = min(total_count, start + batch_size)
                pred = predict_delta(
                    model=model,
                    state_batch=state_x[start:end],
                    dt=float(dt),
                    n_updates=int(n_updates),
                    for_training=False,
                )
                loss = metric_l1_per_sample(pred, target_y[start:end])
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
            if state_x.is_cuda:
                torch.cuda.empty_cache()
            batch_size = next_batch_size


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


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    preset: RepoFaithfulTrainPreset,
    data_cfg: RepoFaithfulDataConfig,
    physics: RepoDoublePendulumPhysics,
    step: int,
    metric: float,
    torch_dtype: torch.dtype,
) -> None:
    payload = {
        "step": int(step),
        "metric": float(metric),
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),
        "train_preset": asdict(preset),
        "data_config": asdict(data_cfg),
        "physics": asdict(physics),
        "dtype": str(torch_dtype),
        "objective": "delta",
        "loss_mode": "l1_sum",
    }
    torch.save(payload, path)


def eval_mode_for_step(
    step: int,
    eval_every: int,
    num_steps: int,
    quick_eval_every: int,
    quick_eval_until: int,
) -> str | None:
    if step == num_steps - 1:
        return "full"
    if step == 0:
        return "quick"
    if step % int(eval_every) == 0:
        return "full"
    if int(quick_eval_every) > 0 and step < int(quick_eval_until) and step % int(quick_eval_every) == 0:
        return "quick"
    return None


def build_monitor_indices(total_size: int, monitor_samples: int, device: torch.device) -> torch.Tensor:
    count = max(1, min(int(monitor_samples), int(total_size)))
    if count >= total_size:
        return torch.arange(total_size, device=device, dtype=torch.long)
    idx = torch.linspace(0, total_size - 1, count, device=device)
    return torch.round(idx).to(dtype=torch.long)


def main() -> None:
    args = parse_args()
    preset = resolve_preset(args)

    set_seed(int(args.seed))
    device = select_device(args.device)
    torch_dtype = torch.float64 if bool(preset.use_double) else torch.float32

    out_paths = ensure_output_dirs(Path(args.out_dir))
    out_dir = out_paths["out_dir"]
    checkpoints_dir = out_paths["checkpoints"]
    results_dir = out_paths["results"]
    plots_dir = out_paths["plots"]

    data_cfg = RepoFaithfulDataConfig(
        seed=int(args.data_seed),
        samples=int(args.samples),
        duration_s=float(args.duration_s),
        fps=int(args.fps),
        lookahead=int(args.lookahead),
        train_fraction=float(args.train_fraction),
    )
    physics = RepoDoublePendulumPhysics()
    dataset = build_repo_faithful_dataset(
        cfg=data_cfg,
        physics=physics,
        torch_dtype=torch_dtype,
    )
    save_dataset_metadata(results_dir / "dataset_metadata.json", dataset)

    dataset["train_x"] = dataset["train_x"].to(device=device, dtype=torch_dtype)
    dataset["train_y"] = dataset["train_y"].to(device=device, dtype=torch_dtype)
    dataset["test_x"] = dataset["test_x"].to(device=device, dtype=torch_dtype)
    dataset["test_y"] = dataset["test_y"].to(device=device, dtype=torch_dtype)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using device: {device} | GPU: {gpu_name} | torch={torch.__version__} | dtype={torch_dtype}")
    else:
        print(f"Using device: {device} | torch={torch.__version__} | dtype={torch_dtype}")

    summary = dataset["summary"]
    print(
        "Repo-faithful dataset: "
        f"samples={summary['samples']} | "
        f"frames_per_trajectory={summary['frames_per_trajectory']} | "
        f"train_trajectories={summary['train_trajectories']} | "
        f"test_trajectories={summary['test_trajectories']} | "
        f"train_samples={summary['train_samples']} | "
        f"test_samples={summary['test_samples']} | "
        f"effective_dt={summary['effective_dt']:.8f}"
    )
    print(
        "Train preset: "
        f"name={preset.name} | "
        f"loss=l1_sum | objective=delta | "
        f"steps={preset.num_steps} | batch_size={preset.batch_size} | "
        f"model_dt={preset.dt:.8f} | n_updates={preset.n_updates} | "
        f"hidden_dim={preset.hidden_dim} | layers={preset.num_hidden_layers}"
    )
    if abs(float(summary["effective_dt"]) - float(preset.dt)) > 1e-6:
        print(
            "Timing note: dataset_dt and model_dt differ. "
            f"dataset_dt={float(summary['effective_dt']):.8f} comes from linspace(0, duration, frames), "
            f"while model_dt={float(preset.dt):.8f} comes from the official hyperopt preset."
        )

    model = LagrangianMLP(
        input_dim=4,
        hidden_dim=int(preset.hidden_dim),
        num_hidden_layers=int(preset.num_hidden_layers),
        activation=str(preset.activation),
        output_dim=int(preset.output_dim),
        init_seed=int(args.seed),
        init_mode="stax",
    ).to(device=device, dtype=torch_dtype)
    optimizer = Adam(model.parameters(), lr=float(preset.lr))

    history: dict[str, list[dict[str, float | int]]] = {"train": [], "test": []}
    best_test_loss = math.inf
    best_step = -1
    eval_batch_size = max(1, int(args.eval_batch_size))
    start_time = time.time()
    train_monitor_idx = build_monitor_indices(
        total_size=int(dataset["train_x"].shape[0]),
        monitor_samples=int(args.monitor_train_samples),
        device=device,
    )
    test_monitor_idx = build_monitor_indices(
        total_size=int(dataset["test_x"].shape[0]),
        monitor_samples=int(args.monitor_test_samples),
        device=device,
    )

    iterator = tqdm(range(int(preset.num_steps)), desc="Training", unit="step", dynamic_ncols=True) if bool(args.progress_bar) else range(int(preset.num_steps))
    log = tqdm.write if bool(args.progress_bar) else print

    for step in iterator:
        step_start_time = time.time()
        current_lr = piecewise_lr(preset, step)
        set_lr(optimizer, current_lr)

        batch_size = min(int(preset.batch_size), int(dataset["train_x"].shape[0]))
        indices = torch.randint(0, dataset["train_x"].shape[0], (batch_size,), device=device)
        batch_x = dataset["train_x"][indices]
        batch_y = dataset["train_y"][indices]

        optimizer.zero_grad(set_to_none=True)
        heartbeat = (
            ProgressHeartbeat(
                label=f"train step={step:06d}",
                interval_s=float(args.heartbeat_interval_s),
                writer=log,
            )
            if float(args.heartbeat_interval_s) > 0.0
            else nullcontext()
        )
        with heartbeat:
            pred = predict_delta(
                model=model,
                state_batch=batch_x,
                dt=float(preset.dt),
                n_updates=int(preset.n_updates),
                for_training=True,
            )
            batch_loss = loss_l1_sum(pred, batch_y)
            if float(preset.l2reg) > 0.0:
                batch_loss = batch_loss + (float(preset.l2reg) * tree_l2_norm(model)) / batch_size
            batch_loss.backward()
            optimizer.step()
        train_step_s = time.time() - step_start_time

        eval_mode = eval_mode_for_step(
            step=step,
            eval_every=int(preset.eval_every),
            num_steps=int(preset.num_steps),
            quick_eval_every=int(args.quick_eval_every),
            quick_eval_until=int(args.quick_eval_until),
        )
        if eval_mode is None:
            elapsed_s = max(time.time() - start_time, 1e-9)
            avg_step_s = elapsed_s / max(1, step + 1)
            eta_s = avg_step_s * max(0, int(preset.num_steps) - (step + 1))
            if bool(args.progress_bar):
                iterator.set_postfix(
                    step_s=f"{train_step_s:.1f}",
                    avg_s=f"{avg_step_s:.1f}",
                    eta_m=f"{eta_s / 60.0:.1f}",
                    lr=f"{current_lr:.2e}",
                )
            continue

        eval_start_time = time.time()
        eval_heartbeat = (
            ProgressHeartbeat(
                label=f"eval step={step:06d}",
                interval_s=float(args.heartbeat_interval_s),
                writer=log,
            )
            if float(args.heartbeat_interval_s) > 0.0
            else nullcontext()
        )
        eval_train_x = dataset["train_x"]
        eval_train_y = dataset["train_y"]
        eval_test_x = dataset["test_x"]
        eval_test_y = dataset["test_y"]
        if eval_mode == "quick":
            eval_train_x = eval_train_x[train_monitor_idx]
            eval_train_y = eval_train_y[train_monitor_idx]
            eval_test_x = eval_test_x[test_monitor_idx]
            eval_test_y = eval_test_y[test_monitor_idx]

        with eval_heartbeat:
            train_loss, used_train_eval_batch = evaluate_l1(
                model=model,
                state_x=eval_train_x,
                target_y=eval_train_y,
                dt=float(preset.dt),
                n_updates=int(preset.n_updates),
                eval_batch_size=eval_batch_size,
            )
            test_loss, used_test_eval_batch = evaluate_l1(
                model=model,
                state_x=eval_test_x,
                target_y=eval_test_y,
                dt=float(preset.dt),
                n_updates=int(preset.n_updates),
                eval_batch_size=eval_batch_size,
            )
            eval_batch_size = min(int(used_train_eval_batch), int(used_test_eval_batch))
        eval_s = time.time() - eval_start_time
        elapsed_s = max(time.time() - start_time, 1e-9)
        steps_per_s = float((step + 1) / elapsed_s)
        total_step_s = time.time() - step_start_time
        avg_step_s = elapsed_s / max(1, step + 1)
        eta_s = avg_step_s * max(0, int(preset.num_steps) - (step + 1))

        history["train"].append(
            {
                "step": int(step),
                "loss": float(train_loss),
                "lr": float(current_lr),
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
                "lr": float(current_lr),
                "elapsed_s": float(elapsed_s),
                "steps_per_s": float(steps_per_s),
                "train_step_s": float(train_step_s),
                "eval_s": float(eval_s),
                "total_step_s": float(total_step_s),
                "eval_mode": str(eval_mode),
            }
        )
        write_metrics_csv(out_dir, history)

        if eval_mode == "full" and math.isfinite(test_loss) and test_loss < best_test_loss:
            best_test_loss = float(test_loss)
            best_step = int(step)
            save_checkpoint(
                path=checkpoints_dir / "model_best.pth",
                model=model,
                preset=preset,
                data_cfg=data_cfg,
                physics=physics,
                step=step,
                metric=best_test_loss,
                torch_dtype=torch_dtype,
            )
            log(f"[BEST] step={step:06d} | test_l1={best_test_loss:.8e}")

        if bool(args.progress_bar):
            iterator.set_postfix(
                train=f"{train_loss:.3e}",
                test=f"{test_loss:.3e}",
                best=f"{best_test_loss:.3e}" if math.isfinite(best_test_loss) else "inf",
                lr=f"{current_lr:.2e}",
                sps=f"{steps_per_s:.2f}",
                step_s=f"{total_step_s:.1f}",
                eta_m=f"{eta_s / 60.0:.1f}",
                eval_bs=f"{eval_batch_size}",
                mode=str(eval_mode),
            )
        log(
            f"step={step:06d}/{int(preset.num_steps):06d} | "
            f"train_l1={train_loss:.8e} | "
            f"test_l1={test_loss:.8e} | "
            f"lr={current_lr:.3e} | "
            f"elapsed_s={elapsed_s:.1f} | "
            f"steps_per_s={steps_per_s:.2f} | "
            f"train_step_s={train_step_s:.1f} | "
            f"eval_s={eval_s:.1f} | "
            f"total_step_s={total_step_s:.1f} | "
            f"eta_m={eta_s / 60.0:.1f} | "
            f"eval_batch_size={eval_batch_size} | "
            f"eval_mode={eval_mode}"
        )

    save_checkpoint(
        path=out_dir / "model_final.pth",
        model=model,
        preset=preset,
        data_cfg=data_cfg,
        physics=physics,
        step=int(preset.num_steps) - 1,
        metric=best_test_loss,
        torch_dtype=torch_dtype,
    )
    save_json(out_dir / "history.json", history)
    write_metrics_csv(out_dir, history)
    plot_loss_curves(out_dir / "metrics.csv", out_path=plots_dir / "loss_curve.png")

    run_config = {
        "train_preset": asdict(preset),
        "data_config": asdict(data_cfg),
        "physics": asdict(physics),
        "summary": summary,
        "dataset_dt": float(summary["effective_dt"]),
        "model_dt": float(preset.dt),
        "best_test_loss": float(best_test_loss),
        "best_step": int(best_step),
        "out_dir": str(out_dir),
        "model_best_path": str(checkpoints_dir / "model_best.pth"),
        "model_final_path": str(out_dir / "model_final.pth"),
        "metrics_csv": str(out_dir / "metrics.csv"),
        "loss_curve": str(plots_dir / "loss_curve.png"),
        "objective": "delta",
        "loss_mode": "l1_sum",
    }
    save_json(out_dir / "run_config.json", run_config)

    print(f"Training complete. Final model: {out_dir / 'model_final.pth'}")
    print(f"Best model: {checkpoints_dir / 'model_best.pth'}")


if __name__ == "__main__":
    main()
