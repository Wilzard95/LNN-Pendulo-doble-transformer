from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lnn.model import LagrangianMLP
from lnn.plotting import plot_rollout
from lnn.repo_faithful_data import (
    RepoDoublePendulumPhysics,
    RepoFaithfulDataConfig,
    build_repo_faithful_dataset,
    simulate_trajectory,
)
from lnn.utils import DEFAULT_OUT_DIR, ensure_output_dirs, load_json, resolve_model_path, save_json, select_device, wrap_coords_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a repo-faithful LNN attempt on synthetic trajectories.")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR / "experiments" / "repo_faithful"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trajectory_id", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    return parser.parse_args()


def predict_delta(
    model: torch.nn.Module,
    state_batch: torch.Tensor,
    dt: float,
    n_updates: int,
) -> torch.Tensor:
    from lnn.dynamics import state_delta_from_lagrangian

    q = state_batch[:, :2]
    qdot = state_batch[:, 2:]
    return state_delta_from_lagrangian(
        model=model,
        q=q,
        qdot=qdot,
        dt=float(dt),
        n_updates=int(n_updates),
        for_training=False,
    )


def evaluate_delta_mae(
    model: torch.nn.Module,
    state_x: torch.Tensor,
    target_y: torch.Tensor,
    dt: float,
    n_updates: int,
    batch_size: int = 16,
) -> tuple[dict[str, float], int]:
    abs_parts: list[np.ndarray] = []
    sq_parts: list[np.ndarray] = []
    total = int(state_x.shape[0])
    batch_size = max(1, min(int(batch_size), total))

    while True:
        try:
            abs_parts.clear()
            sq_parts.clear()
            for start in range(0, total, int(batch_size)):
                end = min(total, start + int(batch_size))
                pred = predict_delta(model=model, state_batch=state_x[start:end], dt=dt, n_updates=n_updates)
                err = (pred - target_y[start:end]).detach().cpu().numpy()
                abs_parts.append(np.abs(err))
                sq_parts.append(err**2)

            abs_err = np.concatenate(abs_parts, axis=0)
            sq_err = np.concatenate(sq_parts, axis=0)
            return {
                "delta_mae_mean": float(np.mean(abs_err)),
                "delta_mse_mean": float(np.mean(sq_err)),
                "delta_mae_q_mean": float(np.mean(abs_err[:, :2])),
                "delta_mae_qdot_mean": float(np.mean(abs_err[:, 2:])),
                "delta_mse_q_mean": float(np.mean(sq_err[:, :2])),
                "delta_mse_qdot_mean": float(np.mean(sq_err[:, 2:])),
                "num_test_samples": int(total),
            }, int(batch_size)
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


def rollout_model(
    model: torch.nn.Module,
    initial_state: np.ndarray,
    dt: float,
    n_updates: int,
    steps: int,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    state = torch.as_tensor(initial_state, dtype=torch_dtype, device=device).unsqueeze(0)
    out = np.zeros((int(steps) + 1, 4), dtype=np.float64)
    out[0] = state.detach().cpu().numpy()[0]
    for idx in range(int(steps)):
        delta = predict_delta(model=model, state_batch=state, dt=dt, n_updates=n_updates)
        state = state + delta
        out[idx + 1] = state.detach().cpu().numpy()[0]
    return out


def main() -> None:
    args = parse_args()
    out_paths = ensure_output_dirs(Path(args.out_dir))
    out_dir = out_paths["out_dir"]
    plots_dir = out_paths["plots"]
    results_dir = out_paths["results"]

    run_config = load_json(out_dir / "run_config.json")
    model_path = resolve_model_path(args.model_path, out_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    train_preset = run_config["train_preset"]
    data_cfg = RepoFaithfulDataConfig(**run_config["data_config"])
    physics = RepoDoublePendulumPhysics(**run_config["physics"])

    ckpt = torch.load(model_path, map_location="cpu")
    model_cfg = ckpt["model_config"]
    torch_dtype = torch.float64 if bool(train_preset.get("use_double", False)) else torch.float32
    device = select_device(args.device)

    dataset = build_repo_faithful_dataset(cfg=data_cfg, physics=physics, torch_dtype=torch_dtype)
    test_x = dataset["test_x"].to(device=device, dtype=torch_dtype)
    test_y = dataset["test_y"].to(device=device, dtype=torch_dtype)

    model = LagrangianMLP(**model_cfg).to(device=device, dtype=torch_dtype)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    delta_metrics, used_eval_batch_size = evaluate_delta_mae(
        model=model,
        state_x=test_x,
        target_y=test_y,
        dt=float(train_preset["dt"]),
        n_updates=int(train_preset["n_updates"]),
        batch_size=int(args.eval_batch_size),
    )
    delta_metrics["eval_batch_size"] = int(used_eval_batch_size)
    save_json(results_dir / "eval_metrics.json", delta_metrics)

    test_traj_ids = dataset["test_trajectory_ids"]
    if len(test_traj_ids) == 0:
        raise RuntimeError("No test trajectories available for rollout evaluation.")
    if args.trajectory_id is None:
        trajectory_id = int(test_traj_ids[0])
    else:
        trajectory_id = int(args.trajectory_id)

    rollout_steps = int(args.rollout_steps)
    rollout_times = np.arange(rollout_steps + 1, dtype=np.float64) * float(train_preset["dt"])
    initial_state = dataset["all_initial_states"][trajectory_id]
    true_state = simulate_trajectory(
        initial_state=initial_state,
        times=rollout_times,
        physics=physics,
        solver=data_cfg.solver,
        rtol=data_cfg.rtol,
        atol=data_cfg.atol,
    )
    pred_state = rollout_model(
        model=model,
        initial_state=initial_state,
        dt=float(train_preset["dt"]),
        n_updates=int(train_preset["n_updates"]),
        steps=rollout_steps,
        device=device,
        torch_dtype=torch_dtype,
    )

    err = pred_state - true_state
    rollout_metrics = {
        "trajectory_id": int(trajectory_id),
        "rollout_steps": int(rollout_steps),
        "rollout_dt": float(train_preset["dt"]),
        "rollout_mae_state_mean": float(np.mean(np.abs(err))),
        "rollout_mae_q_mean": float(np.mean(np.abs(err[:, :2]))),
        "rollout_mae_qdot_mean": float(np.mean(np.abs(err[:, 2:]))),
        "rollout_final_mae_q_mean": float(np.mean(np.abs(err[-1, :2]))),
        "rollout_final_mae_qdot_mean": float(np.mean(np.abs(err[-1, 2:]))),
    }
    save_json(results_dir / "rollout_metrics.json", rollout_metrics)

    wrapped_true = true_state.copy()
    wrapped_pred = pred_state.copy()
    wrapped_true[:, :2] = wrap_coords_np(wrapped_true)[:, :2]
    wrapped_pred[:, :2] = wrap_coords_np(wrapped_pred)[:, :2]

    stem = f"rollout_traj_{trajectory_id:03d}"
    csv_path = results_dir / f"{stem}.csv"
    fig_path = plots_dir / f"{stem}.png"

    df = pd.DataFrame(
        {
            "time_s": rollout_times,
            "true_theta1_rad": wrapped_true[:, 0],
            "true_theta2_rad": wrapped_true[:, 1],
            "true_omega1_rad_s": wrapped_true[:, 2],
            "true_omega2_rad_s": wrapped_true[:, 3],
            "pred_theta1_rad": wrapped_pred[:, 0],
            "pred_theta2_rad": wrapped_pred[:, 1],
            "pred_omega1_rad_s": wrapped_pred[:, 2],
            "pred_omega2_rad_s": wrapped_pred[:, 3],
        }
    )
    df.to_csv(csv_path, index=False)
    plot_rollout(rollout_times, wrapped_true, wrapped_pred, out_path=fig_path)

    print(f"Model: {model_path}")
    print(f"Delta MAE mean: {delta_metrics['delta_mae_mean']:.8e}")
    print(f"Delta MSE mean: {delta_metrics['delta_mse_mean']:.8e}")
    print(f"Rollout q MAE mean: {rollout_metrics['rollout_mae_q_mean']:.8e}")
    print(f"Rollout qdot MAE mean: {rollout_metrics['rollout_mae_qdot_mean']:.8e}")
    print(f"Saved eval metrics: {results_dir / 'eval_metrics.json'}")
    print(f"Saved rollout CSV: {csv_path}")
    print(f"Saved rollout plot: {fig_path}")


if __name__ == "__main__":
    main()
