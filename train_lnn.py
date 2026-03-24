from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from lnn.data import prepare_datasets
from lnn.dynamics import qddot_from_lagrangian
from lnn.model import LagrangianMLP
from lnn.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    ensure_output_dirs,
    save_json,
    select_device,
    set_seed,
    utc_timestamp,
)


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    val = str(v).strip().lower()
    if val in {"1", "true", "t", "yes", "y"}:
        return True
    if val in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret as bool: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Lagrangian Neural Network for double pendulum data.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="softplus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--split_by_trajectory", type=str2bool, default=False)
    parser.add_argument("--split_by_sample", action="store_true", help="Shortcut for split_by_trajectory=false")
    parser.add_argument("--normalize", type=str2bool, default=True, help="Normalize training data (z-score).")
    parser.add_argument("--normalization_file", type=str, default=None, help="Path to normalization JSON to load/save.")
    parser.add_argument("--lambda_damp", type=float, default=0.0, help="Weight decay or damping coefficient.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def _print_best_update(message_core: str) -> None:
    encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    if "utf" in encoding:
        print(f"🚀 {message_core}")
    else:
        print(f"[BEST] {message_core}")


def _piecewise_lr(base_lr: float, global_step: int, total_steps: int) -> float:
    one_third = total_steps // 3
    two_third = (2 * total_steps) // 3
    if global_step < one_third:
        return float(base_lr)
    if global_step < two_third:
        return float(base_lr / 10.0)
    return float(base_lr / 100.0)


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    base_lr: float = 1e-3,
    total_steps: int = 1,
    global_step: int = 0,
) -> tuple[float, int, int, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0
    skipped_batches = 0
    last_lr = float(base_lr)

    iterator = tqdm(loader, leave=False)
    for state_x, xdot_true in iterator:
        state_x = state_x.to(device)
        xdot_true = xdot_true.to(device)

        q = state_x[:, :2]
        qdot = state_x[:, 2:]

        if training:
            last_lr = _piecewise_lr(base_lr, global_step, total_steps)
            _set_lr(optimizer, last_lr)
            optimizer.zero_grad(set_to_none=True)

        try:
            qddot_pred = qddot_from_lagrangian(
                model=model,
                q=q,
                qdot=qdot,
                for_training=training,
            )
        except RuntimeError as exc:
            skipped_batches += 1
            warnings.warn(f"Skipping batch due to dynamics failure: {exc}")
            if training:
                global_step += 1
            continue

        xdot_pred = torch.cat([qdot, qddot_pred], dim=1)
        loss = torch.mean((xdot_pred - xdot_true) ** 2)
        if not torch.isfinite(loss):
            skipped_batches += 1
            warnings.warn("Skipping batch with non-finite loss.")
            if training:
                global_step += 1
            continue

        if training:
            loss.backward()
            optimizer.step()
            global_step += 1

        batch_size = int(state_x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    if total_count == 0:
        return math.inf, skipped_batches, global_step, last_lr
    return total_loss / total_count, skipped_batches, global_step, last_lr


def main() -> None:
    args = parse_args()
    if args.split_by_sample:
        args.split_by_trajectory = False

    set_seed(args.seed)
    device = select_device(args.device)
    torch_dtype = torch.float64 if args.double else torch.float32
    np_dtype = np.float64 if args.double else np.float32
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Using device: {device} | GPU: {gpu_name} | torch={torch.__version__} | dtype={torch_dtype}")
    else:
        print(f"Using device: {device} | torch={torch.__version__} | dtype={torch_dtype}")

    out_paths = ensure_output_dirs(Path(args.out_dir))
    out_dir = out_paths["out_dir"]
    checkpoints_dir = out_paths["checkpoints"]
    results_dir = out_paths["results"]

    data_bundle = prepare_datasets(
        data_dir=Path(args.data_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_by_trajectory=args.split_by_trajectory,
        np_dtype=np_dtype,
        torch_dtype=torch_dtype,
        normalize=bool(args.normalize),
        normalization_file=Path(args.normalization_file) if args.normalization_file else None,
        out_dir=out_dir,
    )

    summary = data_bundle["summary"]
    split_info = data_bundle["split_info"]
    print(f"Loaded files found: {summary['num_files_found']}")
    print(
        "Loaded samples: "
        f"train={summary['num_samples_train']}, "
        f"val={summary['num_samples_val']}, "
        f"test={summary['num_samples_test']}, "
        f"total={summary['num_samples_total']}"
    )
    print(
        "Loaded files by split: "
        f"train={summary['num_files_loaded_train']}, "
        f"val={summary['num_files_loaded_val']}, "
        f"test={summary['num_files_loaded_test']}"
    )

    train_loader = DataLoader(
        data_bundle["train_dataset"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        data_bundle["val_dataset"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_steps = int(args.epochs) * max(1, len(train_loader))
    if total_steps <= 0:
        raise RuntimeError("No training steps available (empty train loader).")

    model = LagrangianMLP(
        input_dim=4,
        hidden_dim=int(args.hidden_dim),
        num_hidden_layers=int(args.num_hidden_layers),
        activation=str(args.activation),
        output_dim=1,
        init_seed=int(args.seed),
    ).to(device=device, dtype=torch_dtype)

    optimizer = Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.lambda_damp))

    metrics_rows: list[dict[str, float | int]] = []
    best_val = math.inf
    best_epoch = -1
    global_step = 0

    for epoch in range(1, int(args.epochs) + 1):
        train_loss, train_skipped, global_step, lr_epoch = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            base_lr=float(args.lr),
            total_steps=total_steps,
            global_step=global_step,
        )
        val_loss, val_skipped, _, _ = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            base_lr=float(args.lr),
            total_steps=total_steps,
            global_step=global_step,
        )

        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(lr_epoch),
                "global_step_end": int(global_step),
                "train_skipped_batches": int(train_skipped),
                "val_skipped_batches": int(val_skipped),
            }
        )

        if epoch % int(args.log_every) == 0:
            print(
                f"Epoch {epoch:04d}/{int(args.epochs):04d} | "
                f"train_loss={train_loss:.8e} | "
                f"val_loss={val_loss:.8e} | "
                f"lr={lr_epoch:.3e} | "
                f"skipped(train/val)={train_skipped}/{val_skipped}"
            )

        if math.isfinite(val_loss) and val_loss < best_val:
            previous_best = best_val
            best_val = val_loss
            best_epoch = epoch
            best_payload = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "model_config": model.get_config(),
                "state_dict": model.state_dict(),
                "double": bool(args.double),
                "split_info": split_info,
            }
            torch.save(best_payload, checkpoints_dir / "model_best.pth")
            if math.isfinite(previous_best):
                improvement = previous_best - best_val
                _print_best_update(
                    f"New best model at epoch {epoch:04d} | "
                    f"val_loss={best_val:.8e} | "
                    f"improvement={improvement:.8e}"
                )
            else:
                _print_best_update(f"New best model at epoch {epoch:04d} | val_loss={best_val:.8e}")

    final_payload = {
        "epoch": int(args.epochs),
        "train_loss": float(metrics_rows[-1]["train_loss"]),
        "val_loss": float(metrics_rows[-1]["val_loss"]),
        "model_config": model.get_config(),
        "state_dict": model.state_dict(),
        "double": bool(args.double),
        "split_info": split_info,
    }
    torch.save(final_payload, out_dir / "model_final.pth")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)

    run_config = {
        "timestamp_utc": utc_timestamp(),
        "paths": {
            "data_dir": str(Path(args.data_dir)),
            "out_dir": str(out_dir),
            "model_best_path": str(checkpoints_dir / "model_best.pth"),
            "model_final_path": str(out_dir / "model_final.pth"),
            "metrics_csv": str(out_dir / "metrics.csv"),
        },
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "device": str(device),
            "double": bool(args.double),
            "num_workers": int(args.num_workers),
            "split_by_trajectory": bool(args.split_by_trajectory),
            "normalize": bool(args.normalize),
            "normalization_file": str(args.normalization_file) if args.normalization_file else None,
            "lambda_damp": float(args.lambda_damp),
            "lr_schedule": "piecewise_1x_0.1x_0.01x_by_global_step",
            "lr_schedule_step_1": int(total_steps // 3),
            "lr_schedule_step_2": int((2 * total_steps) // 3),
        },
        "model_config": model.get_config(),
        "summary": summary,
        "split": split_info,
        "normalization": data_bundle.get("normalization"),
        "final_losses": {
            "train_loss": float(metrics_rows[-1]["train_loss"]),
            "val_loss": float(metrics_rows[-1]["val_loss"]),
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "global_steps_total": int(global_step),
        },
    }
    save_json(out_dir / "run_config.json", run_config)
    save_json(results_dir / "split_info.json", split_info)

    print(f"Training complete. Final model: {out_dir / 'model_final.pth'}")
    print(f"Best model: {checkpoints_dir / 'model_best.pth'}")


if __name__ == "__main__":
    main()
