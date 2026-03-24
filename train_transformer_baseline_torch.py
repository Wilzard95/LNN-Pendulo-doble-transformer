from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
from torch import nn

from torch_delta_models import StateTransformerEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOCAL_ROOT = Path(__file__).resolve().parent


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
    parser = argparse.ArgumentParser(description="Train a transformer delta baseline on the official double-pendulum cache using PyTorch.")
    parser.add_argument("--out_dir", type=str, default="experiments/compare_transformer_torch")
    parser.add_argument("--reference_out_dir", type=str, default="experiments/official_jax_cpu_retry")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=27)
    parser.add_argument("--lr", type=float, default=0.005516656601005163)
    parser.add_argument("--lr2", type=float, default=1.897157209816416e-05)
    parser.add_argument("--l2reg", type=float, default=0.24927677946969878)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_multiplier", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--warmup_eval_every", type=int, default=100)
    parser.add_argument("--warmup_eval_until", type=int, default=1000)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_coords_np(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[:, :2] = (y[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
    return y


def resolve_reference_paths(reference_out_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    attempts_summary_path = reference_out_dir / "results" / "attempts_summary.json"
    if not attempts_summary_path.exists():
        raise FileNotFoundError(f"Could not find reference summary: {attempts_summary_path}")
    attempts_summary = json.loads(attempts_summary_path.read_text(encoding="utf-8"))
    dataset_summary = attempts_summary["dataset_summary"]
    return attempts_summary, dataset_summary


def load_dataset(cache_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    cached = np.load(cache_path)
    data_np = {
        "x": wrap_coords_np(np.asarray(cached["x"], dtype=np.float32)),
        "dx": np.asarray(cached["dx"], dtype=np.float32),
        "test_x": wrap_coords_np(np.asarray(cached["test_x"], dtype=np.float32)),
        "test_dx": np.asarray(cached["test_dx"], dtype=np.float32),
    }
    return {k: torch.from_numpy(v).to(device) for k, v in data_np.items()}


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "out_dir": out_dir,
        "checkpoints": out_dir / "checkpoints",
        "plots": out_dir / "plots",
        "results": out_dir / "results",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_loss_artifacts(rows: list[dict[str, float]], results_dir: Path, plots_dir: Path) -> None:
    csv_path = results_dir / "loss_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "train_loss", "test_loss", "lr"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if not rows:
        return

    steps = np.array([row["step"] for row in rows], dtype=np.float64)
    train_loss = np.array([row["train_loss"] for row in rows], dtype=np.float64)
    test_loss = np.array([row["test_loss"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, train_loss, label="train")
    ax.plot(steps, test_loss, label="test")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Transformer baseline training")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def l1_sum_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).sum(dim=-1).mean()


@torch.no_grad()
def full_eval(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, x.shape[0], batch_size):
        end = min(x.shape[0], start + batch_size)
        pred = model(x[start:end])
        loss = l1_sum_loss(pred, y[start:end])
        total += float(loss.item()) * (end - start)
        count += int(end - start)
    return total / max(count, 1)


def current_lr(step: int, total_steps: int, lr1: float, lr2: float) -> float:
    return float(lr1 if step < total_steps // 2 else lr2)


def apply_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "architecture": "transformer_encoder",
        "d_model": int(args.d_model),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "ff_multiplier": int(args.ff_multiplier),
        "dropout": float(args.dropout),
        "num_tokens": 4,
        "output_dim": 4,
    }


def create_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = StateTransformerEncoder(
        d_model=int(args.d_model),
        num_heads=int(args.num_heads),
        num_layers=int(args.num_layers),
        ff_multiplier=int(args.ff_multiplier),
        dropout=float(args.dropout),
    )
    return model.to(device)


def checkpoint_payload(
    model: nn.Module,
    args: argparse.Namespace,
    dataset_summary: dict[str, object],
    reference_out_dir: Path,
) -> dict[str, object]:
    return {
        "framework": "torch",
        "model_kind": "baseline_transformer_torch",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model_config_from_args(args),
        "train_args": {
            "dt": 0.09609870774790222,
            "batch_size": int(args.batch_size),
            "loss": "l1_sum",
            "num_steps": int(args.num_steps),
            "lr": float(args.lr),
            "lr2": float(args.lr2),
            "l2reg": float(args.l2reg),
        },
        "dataset_summary": dataset_summary,
        "reference_out_dir": str(reference_out_dir),
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run_attempt(
    base_out_dir: Path,
    args: argparse.Namespace,
    dataset: dict[str, torch.Tensor],
    dataset_summary: dict[str, object],
    reference_out_dir: Path,
    attempt_idx: int,
    attempt_seed: int,
    device: torch.device,
) -> dict[str, object]:
    attempt_name = f"attempt_{attempt_idx:02d}_seed_{attempt_seed:04d}"
    attempt_dir = base_out_dir / "attempts" / attempt_name
    paths = ensure_dirs(attempt_dir)

    set_seed(int(attempt_seed))
    model = create_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    train_x = dataset["x"]
    train_y = dataset["dx"]
    test_x = dataset["test_x"]
    test_y = dataset["test_dx"]

    rows: list[dict[str, float]] = []
    best_loss = math.inf
    best_step = -1
    best_state = None
    exception_text: str | None = None

    log_path = paths["results"] / "training.log"
    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_fh, contextlib.redirect_stdout(Tee(sys.stdout, log_fh)):
        print(f"[attempt] index={attempt_idx} seed={attempt_seed} model_kind=baseline_transformer_torch")
        try:
            for step in range(int(args.num_steps)):
                model.train()
                lr_value = current_lr(step, int(args.num_steps), float(args.lr), float(args.lr2))
                apply_lr(optimizer, lr_value)

                idx = torch.randint(0, train_x.shape[0], (int(args.batch_size),), device=device)
                batch_x = train_x[idx]
                batch_y = train_y[idx]

                pred = model(batch_x)
                data_loss = l1_sum_loss(pred, batch_y)
                l2_penalty = sum((p.pow(2).sum() for p in model.parameters())) * (float(args.l2reg) / float(args.batch_size))
                loss = data_loss + l2_penalty

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if int(args.log_every) > 0 and step % int(args.log_every) == 0:
                    print(f"train_step={step:06d}, batch_loss={float(data_loss.item()):.6f}, lr={lr_value:.6e}")

                should_eval = False
                if step < int(args.warmup_eval_until) and step % int(args.warmup_eval_every) == 0:
                    should_eval = True
                elif step % int(args.eval_every) == 0:
                    should_eval = True
                elif step == int(args.num_steps) - 1:
                    should_eval = True

                if not should_eval:
                    continue

                train_loss = full_eval(model, train_x, train_y, batch_size=512)
                test_loss = full_eval(model, test_x, test_y, batch_size=512)
                rows.append(
                    {
                        "step": float(step),
                        "train_loss": float(train_loss),
                        "test_loss": float(test_loss),
                        "lr": float(lr_value),
                    }
                )
                print(f"step={step:06d}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}, lr={lr_value:.6e}")

                if np.isfinite(test_loss) and test_loss < best_loss:
                    best_loss = float(test_loss)
                    best_step = int(step)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    print(f"[BEST] step={step:06d} | test_loss={test_loss:.8e}")

                if not np.isfinite(test_loss):
                    raise FloatingPointError(f"Non-finite test loss at step {step}")
        except Exception:
            exception_text = traceback.format_exc()
            print(exception_text, flush=True)

    elapsed_s = time.perf_counter() - start_time

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    best_model = create_model(args, device=torch.device("cpu"))
    best_model.load_state_dict(best_state)
    final_model = create_model(args, device=torch.device("cpu"))
    final_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})

    with (paths["checkpoints"] / "model_best.pt").open("wb") as fh:
        torch.save(checkpoint_payload(best_model, args, dataset_summary, reference_out_dir), fh)
    with (attempt_dir / "model_final.pt").open("wb") as fh:
        torch.save(checkpoint_payload(final_model, args, dataset_summary, reference_out_dir), fh)

    save_loss_artifacts(rows, paths["results"], paths["plots"])

    completed = exception_text is None
    summary = {
        "attempt_index": int(attempt_idx),
        "attempt_name": attempt_name,
        "seed": int(attempt_seed),
        "attempt_dir": str(attempt_dir),
        "framework": "torch",
        "model_kind": "baseline_transformer_torch",
        "completed": bool(completed),
        "stop_reason": "completed" if completed else "exception",
        "stop_iteration": int(args.num_steps - 1 if completed else rows[-1]["step"] if rows else -1),
        "best_iteration": int(best_step),
        "best_loss": float(best_loss),
        "elapsed_s": float(elapsed_s),
        "num_train_samples": int(train_x.shape[0]),
        "num_test_samples": int(test_x.shape[0]),
        "logged_points": int(len(rows)),
        "exception": exception_text,
        "files": {
            "model_best": str(paths["checkpoints"] / "model_best.pt"),
            "model_final": str(attempt_dir / "model_final.pt"),
            "training_log": str(log_path),
            "loss_csv": str(paths["results"] / "loss_history.csv"),
            "loss_plot": str(paths["plots"] / "loss_curve.png"),
        },
    }
    (paths["results"] / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def promote_best_attempt(base_paths: dict[str, Path], best_attempt: dict[str, object]) -> None:
    attempt_dir = Path(str(best_attempt["attempt_dir"]))
    copy_if_exists(attempt_dir / "checkpoints" / "model_best.pt", base_paths["checkpoints"] / "model_best.pt")
    copy_if_exists(attempt_dir / "model_final.pt", base_paths["out_dir"] / "model_final.pt")
    copy_if_exists(attempt_dir / "results" / "training.log", base_paths["results"] / "training.log")
    copy_if_exists(attempt_dir / "results" / "loss_history.csv", base_paths["results"] / "loss_history.csv")
    copy_if_exists(attempt_dir / "plots" / "loss_curve.png", base_paths["plots"] / "loss_curve.png")
    copy_if_exists(attempt_dir / "results" / "train_summary.json", base_paths["results"] / "best_attempt_summary.json")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.num_steps = 50
        args.max_attempts = min(int(args.max_attempts), 1)
        args.d_model = 64
        args.num_layers = 1

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (LOCAL_ROOT / out_dir).resolve()
    reference_out_dir = Path(args.reference_out_dir)
    if not reference_out_dir.is_absolute():
        reference_out_dir = (LOCAL_ROOT / reference_out_dir).resolve()

    paths = ensure_dirs(out_dir)
    (out_dir / "attempts").mkdir(parents=True, exist_ok=True)

    attempts_summary, dataset_summary = resolve_reference_paths(reference_out_dir)
    cache_path = Path(dataset_summary["cache_path"])
    device = resolve_device(args.device)

    run_config = {
        "framework": "torch",
        "mode": "baseline_transformer_torch",
        "reference_out_dir": str(reference_out_dir),
        "reference_best_attempt": attempts_summary["best_attempt"]["attempt_name"],
        "dataset_summary": dataset_summary,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "args": vars(args),
        "model_config": model_config_from_args(args),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Using device: {device} | torch={torch.__version__}")
    print(f"[dataset] loading cache: {cache_path}")
    dataset = load_dataset(cache_path, device=device)
    print(
        f"[dataset] ready | train_samples={dataset['x'].shape[0]} | "
        f"test_samples={dataset['test_x'].shape[0]}"
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

        if bool(summary["completed"]):
            print("[attempt] completed full schedule. Stopping retries.")
            break
        if attempt_idx + 1 < int(args.max_attempts):
            print("[attempt] retrying with next seed.")

    total_elapsed = time.perf_counter() - total_start

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
            writer.writerow({key: attempt[key] for key in writer.fieldnames})

    overall_summary = {
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
    (paths["results"] / "attempts_summary.json").write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    if best_attempt is None:
        raise RuntimeError("No successful attempt produced a finite best loss.")

    promote_best_attempt(paths, best_attempt)
    (paths["results"] / "train_summary.json").write_text(json.dumps(best_attempt, indent=2), encoding="utf-8")

    print(f"Framework: torch")
    print(f"Device: {device}")
    print(f"Total elapsed seconds: {total_elapsed:.1f}")
    print(f"Attempts run: {len(attempts)}")
    print(f"Best attempt: {best_attempt['attempt_name']}")
    print(f"Best loss: {float(best_attempt['best_loss']):.8e}")
    print(f"Best params: {paths['checkpoints'] / 'model_best.pt'}")
    print(f"Training log: {paths['results'] / 'training.log'}")
    print(f"Loss CSV: {paths['results'] / 'loss_history.csv'}")
    print(f"Loss plot: {paths['plots'] / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
