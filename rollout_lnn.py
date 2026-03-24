from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lnn.data import discover_simulation_files, load_full_trajectory
from lnn.dynamics import lagrangian_energy, qddot_from_lagrangian
from lnn.integrators import step_dynamics
from lnn.model import LagrangianMLP
from lnn.plotting import plot_rollout
from lnn.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    ensure_output_dirs,
    load_json,
    resolve_model_path,
    select_device,
    wrap_coords_torch,
    load_normalization_config,
    normalize_state,
    denormalize_xdot,
)


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
    parser = argparse.ArgumentParser(description="Run rollout using a trained LNN model.")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None, help="sim_data_###.txt name or absolute path")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--horizon_s", type=float, default=5.0)
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--integrator", type=str, choices=["rk4", "euler"], default="rk4")
    parser.add_argument("--k_init", type=int, default=1, help="Use sample k (1-based) as initial condition")
    parser.add_argument("--normalize", type=str2bool, default=None)
    parser.add_argument("--normalization_file", type=str, default=None)
    parser.add_argument("--double", type=str2bool, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _resolve_data_file(
    data_file_arg: str | None,
    data_dir: Path,
    run_config: dict[str, Any],
) -> Path:
    if data_file_arg:
        candidate = Path(data_file_arg)
        if not candidate.is_absolute():
            candidate = data_dir / candidate
        return candidate

    split_cfg = run_config.get("split", {})
    test_files = split_cfg.get("test_files", [])
    if test_files:
        return data_dir / test_files[0]

    all_files = discover_simulation_files(data_dir)
    return all_files[0]


def main() -> None:
    args = parse_args()
    out_paths = ensure_output_dirs(Path(args.out_dir))
    out_dir = out_paths["out_dir"]
    plots_dir = out_paths["plots"]
    results_dir = out_paths["results"]
    data_dir = Path(args.data_dir)

    run_config_path = out_dir / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {}
    hp_cfg = run_config.get("hyperparameters", {})

    # determine normalization config (same as training)
    normalize_cfg = None
    if args.normalize is None:
        normalize = bool(hp_cfg.get("normalize", False))
    else:
        normalize = bool(args.normalize)
    if normalize:
        normalize_cfg = run_config.get("normalization")
        if normalize_cfg is None:
            norm_path = Path(args.normalization_file) if args.normalization_file else (Path(args.out_dir) / "normalization.json")
            if norm_path.exists():
                normalize_cfg = load_normalization_config(norm_path)

    model_path = resolve_model_path(args.model_path, out_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    model_cfg = ckpt.get(
        "model_config",
        {"input_dim": 4, "hidden_dim": 500, "num_hidden_layers": 4, "activation": "softplus", "output_dim": 1},
    )
    ckpt_double = bool(ckpt.get("double", hp_cfg.get("double", False)))
    use_double = ckpt_double if args.double is None else bool(args.double)
    torch_dtype = torch.float64 if use_double else torch.float32
    np_dtype = np.float64 if use_double else np.float32

    device = select_device(args.device)
    model = LagrangianMLP(**model_cfg).to(device=device, dtype=torch_dtype)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_file = _resolve_data_file(args.data_file, data_dir, run_config)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    traj = load_full_trajectory(data_file, dtype=np_dtype)
    time = traj["time"]
    q = traj["q"]
    qdot = traj["qdot"]
    e_true_all = traj.get("energy")
    dt = float(traj["dt_median"])

    start_idx = max(0, int(args.k_init) - 1)
    if start_idx >= len(time) - 1:
        raise ValueError(f"k_init={args.k_init} is out of range for file with {len(time)} points.")

    max_steps = len(time) - 1 - start_idx
    if args.n_steps is not None:
        n_steps = min(int(args.n_steps), max_steps)
    else:
        n_steps = min(int(round(float(args.horizon_s) / dt)), max_steps)
    if n_steps <= 0:
        raise ValueError("Computed n_steps <= 0. Check horizon_s/n_steps and data dt.")

    q_pred = np.zeros((n_steps + 1, 2), dtype=np_dtype)
    qdot_pred = np.zeros((n_steps + 1, 2), dtype=np_dtype)
    q_pred[0] = q[start_idx]
    qdot_pred[0] = qdot[start_idx]

    q_t = torch.tensor(q_pred[0:1], dtype=torch_dtype, device=device)
    qdot_t = torch.tensor(qdot_pred[0:1], dtype=torch_dtype, device=device)
    state0 = wrap_coords_torch(torch.cat([q_t, qdot_t], dim=1))
    q_t = state0[:, :2]
    qdot_t = state0[:, 2:]

    # If model was trained with normalized state, normalize before passing through model,
    # and denormalize qddot for physical integration.
    if normalize_cfg is not None:
        eps = float(normalize_cfg.get("eps", 1e-8))
        q_mean = torch.tensor(normalize_cfg["q_mean"], dtype=torch_dtype, device=device)
        q_std = torch.tensor(normalize_cfg["q_std"], dtype=torch_dtype, device=device)
        qdot_mean = torch.tensor(normalize_cfg["qdot_mean"], dtype=torch_dtype, device=device)
        qdot_std = torch.tensor(normalize_cfg["qdot_std"], dtype=torch_dtype, device=device)
        qddot_mean = torch.tensor(normalize_cfg["qddot_mean"], dtype=torch_dtype, device=device)
        qddot_std = torch.tensor(normalize_cfg["qddot_std"], dtype=torch_dtype, device=device)

        def accel_fn(q_state: torch.Tensor, qdot_state: torch.Tensor) -> torch.Tensor:
            q_norm = (q_state - q_mean) / (q_std + eps)
            qdot_norm = (qdot_state - qdot_mean) / (qdot_std + eps)
            qddot_norm = qddot_from_lagrangian(
                model=model,
                q=q_norm,
                qdot=qdot_norm,
                for_training=False,
            )
            return qddot_norm * (qddot_std + eps) + qddot_mean

    else:
        def accel_fn(q_state: torch.Tensor, qdot_state: torch.Tensor) -> torch.Tensor:
            return qddot_from_lagrangian(
                model=model,
                q=q_state,
                qdot=qdot_state,
                for_training=False,
            )

    for i in range(n_steps):
        try:
            q_next, qdot_next = step_dynamics(
                q=q_t,
                qdot=qdot_t,
                dt=dt,
                accel_fn=accel_fn,
                integrator=args.integrator,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Rollout failed at step {i}: {exc}") from exc

        state_next = torch.cat([q_next, qdot_next], dim=1)
        state_next = wrap_coords_torch(state_next)
        q_t = state_next[:, :2].detach()
        qdot_t = state_next[:, 2:].detach()
        q_pred[i + 1] = q_t.cpu().numpy()[0]
        qdot_pred[i + 1] = qdot_t.cpu().numpy()[0]

    end_idx = start_idx + n_steps + 1
    time_slice = time[start_idx:end_idx]
    q_true = q[start_idx:end_idx]
    qdot_true = qdot[start_idx:end_idx]

    true_state = np.concatenate([q_true, qdot_true], axis=1)
    pred_state = np.concatenate([q_pred, qdot_pred], axis=1)

    # Energy computed from model depends on whether the model was trained on normalized inputs.
    if normalize_cfg is None:
        e_pred_t = lagrangian_energy(
            model=model,
            q=torch.as_tensor(q_pred, dtype=torch_dtype, device=device),
            qdot=torch.as_tensor(qdot_pred, dtype=torch_dtype, device=device),
            for_training=False,
        )
        e_pred = e_pred_t.detach().cpu().numpy()
        energy_rel_error_mean = None
        if e_true_all is not None:
            e_true = np.asarray(e_true_all[start_idx:end_idx], dtype=np.float64)
            n_e = min(len(e_true), len(e_pred))
            if n_e > 0:
                e_true = e_true[:n_e]
                e_pred_cmp = e_pred[:n_e]
                rel_err = np.abs(e_pred_cmp - e_true) / np.maximum(np.abs(e_true), 1e-8)
                energy_rel_error_mean = float(np.mean(rel_err))
    else:
        # Model expects normalized inputs; energy is not directly comparable to physical energy.
        e_pred = np.full((len(time_slice),), np.nan, dtype=np.float64)
        energy_rel_error_mean = None

    stem = data_file.stem
    csv_path = results_dir / f"rollout_{stem}.csv"
    fig_path = plots_dir / f"rollout_{stem}.png"

    df = pd.DataFrame(
        {
            "time_s": time_slice,
            "true_theta1_rad": q_true[:, 0],
            "true_theta2_rad": q_true[:, 1],
            "true_omega1_rad_s": qdot_true[:, 0],
            "true_omega2_rad_s": qdot_true[:, 1],
            "pred_theta1_rad": q_pred[:, 0],
            "pred_theta2_rad": q_pred[:, 1],
            "pred_omega1_rad_s": qdot_pred[:, 0],
            "pred_omega2_rad_s": qdot_pred[:, 1],
            "pred_energy": e_pred[: len(time_slice)],
            "true_energy": np.asarray(e_true_all[start_idx:end_idx], dtype=np_dtype)
            if e_true_all is not None
            else np.nan,
        }
    )
    df.to_csv(csv_path, index=False)

    plot_rollout(time_slice, true_state, pred_state, out_path=fig_path)

    print(f"Rollout data file: {data_file}")
    print(f"dt inferred (median): {dt:.8f} s")
    print(f"Integrator: {args.integrator}")
    print(f"n_steps: {n_steps}")
    print(f"Mean energy discrepancy (relative): {energy_rel_error_mean}")
    print(f"Saved rollout CSV: {csv_path}")
    print(f"Saved rollout plot: {fig_path}")


if __name__ == "__main__":
    main()
