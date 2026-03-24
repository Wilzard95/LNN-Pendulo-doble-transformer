from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lnn.data import load_full_trajectory, prepare_datasets
from lnn.dynamics import lagrangian_energy, qddot_from_lagrangian
from lnn.integrators import step_dynamics
from lnn.model import LagrangianMLP
from lnn.plotting import plot_error_histograms, plot_loss_curves, plot_qddot_scatter
from lnn.utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUT_DIR,
    ensure_output_dirs,
    load_json,
    resolve_model_path,
    save_json,
    select_device,
    utc_timestamp,
    wrap_coords_torch,
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
    parser = argparse.ArgumentParser(description="Evaluate LNN model on test split.")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--split_by_trajectory", type=str2bool, default=None)
    parser.add_argument("--double", type=str2bool, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--rollout_eval", type=str2bool, default=True)
    parser.add_argument("--rollout_n_traj", type=int, default=32)
    parser.add_argument("--rollout_steps", type=int, default=100)
    parser.add_argument("--integrator", type=str, choices=["rk4", "euler"], default="rk4")
    parser.add_argument("--normalize", type=str2bool, default=None)
    parser.add_argument("--normalization_file", type=str, default=None)
    return parser.parse_args()


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def _resolve_rollout_files(
    data_dir: Path,
    split_info: dict[str, Any],
    split_cfg: dict[str, Any],
) -> list[Path]:
    candidate_names = split_info.get("test_files") or split_cfg.get("test_files") or []
    files: list[Path] = []
    for name in candidate_names:
        p = Path(name)
        if not p.is_absolute():
            p = data_dir / p
        if p.exists():
            files.append(p)
    if files:
        return files
    return sorted(data_dir.glob("sim_data_*.txt"))


def _compute_vector_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"Expected [N, D], got {y_true.shape}")
    if len(labels) != y_true.shape[1]:
        raise ValueError(f"labels length {len(labels)} does not match dimension {y_true.shape[1]}")

    mse_mean = float(np.mean((y_pred - y_true) ** 2))
    mae_mean = float(np.mean(np.abs(y_pred - y_true)))
    metrics = {
        "mse_xdot_mean": mse_mean,
        "mae_xdot_mean": mae_mean,
    }
    r2_vals: list[float] = []
    for idx, label in enumerate(labels):
        err = y_pred[:, idx] - y_true[:, idx]
        mse_i = float(np.mean(err**2))
        mae_i = float(np.mean(np.abs(err)))
        yt = y_true[:, idx]
        ss_res = float(np.sum((yt - y_pred[:, idx]) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2_i = 1.0 - ss_res / (ss_tot + 1e-12)
        metrics[f"mse_{label}"] = mse_i
        metrics[f"mae_{label}"] = mae_i
        metrics[f"r2_{label}"] = float(r2_i)
        r2_vals.append(float(r2_i))
    metrics["r2_xdot_mean"] = float(np.mean(np.asarray(r2_vals, dtype=np.float64)))
    return metrics


def _energy_errors_for_states(
    model: LagrangianMLP,
    q: np.ndarray,
    qdot: np.ndarray,
    energy_true: np.ndarray | None,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray] | None:
    if energy_true is None:
        return None
    if len(energy_true) != len(q):
        n = min(len(energy_true), len(q))
        q = q[:n]
        qdot = qdot[:n]
        energy_true = energy_true[:n]
    mask = np.isfinite(energy_true)
    if not np.any(mask):
        return None

    q_t = torch.as_tensor(q[mask], dtype=torch_dtype, device=device)
    qdot_t = torch.as_tensor(qdot[mask], dtype=torch_dtype, device=device)
    with torch.set_grad_enabled(True):
        e_pred = lagrangian_energy(model, q_t, qdot_t, for_training=False).detach().cpu().numpy()
    e_true = np.asarray(energy_true[mask], dtype=np.float64)
    abs_err = np.abs(e_pred - e_true)
    rel_err = abs_err / np.maximum(np.abs(e_true), 1e-8)
    return abs_err.astype(np.float64, copy=False), rel_err.astype(np.float64, copy=False)


def _compute_rollout_metrics(
    model: LagrangianMLP,
    rollout_files: list[Path],
    device: torch.device,
    torch_dtype: torch.dtype,
    np_dtype: np.dtype,
    n_traj: int,
    horizon_steps: int,
    integrator: str = "rk4",
    norm_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if n_traj <= 0:
        raise ValueError("rollout_n_traj must be > 0")
    if horizon_steps <= 0:
        raise ValueError("rollout_steps must be > 0")

    q_sq_err_chunks: list[np.ndarray] = []
    q_abs_err_chunks: list[np.ndarray] = []
    qdot_sq_err_chunks: list[np.ndarray] = []
    qdot_abs_err_chunks: list[np.ndarray] = []
    final_q_abs_err: list[np.ndarray] = []
    final_qdot_abs_err: list[np.ndarray] = []
    dt_values: list[float] = []
    used_files: list[str] = []
    energy_rel_err_chunks: list[np.ndarray] = []
    energy_abs_err_chunks: list[np.ndarray] = []

    # Precompute normalization tensors if needed
    if norm_cfg is not None:
        eps = float(norm_cfg.get("eps", 1e-8))
        q_mean = torch.tensor(norm_cfg["q_mean"], dtype=torch_dtype, device=device)
        q_std = torch.tensor(norm_cfg["q_std"], dtype=torch_dtype, device=device)
        qdot_mean = torch.tensor(norm_cfg["qdot_mean"], dtype=torch_dtype, device=device)
        qdot_std = torch.tensor(norm_cfg["qdot_std"], dtype=torch_dtype, device=device)
        qddot_mean = torch.tensor(norm_cfg["qddot_mean"], dtype=torch_dtype, device=device)
        qddot_std = torch.tensor(norm_cfg["qddot_std"], dtype=torch_dtype, device=device)

    for file_path in rollout_files:
        if len(used_files) >= n_traj:
            break
        traj = load_full_trajectory(file_path, dtype=np_dtype)
        q_true = np.asarray(traj["q"], dtype=np_dtype)
        qdot_true = np.asarray(traj["qdot"], dtype=np_dtype)
        energy_true = traj.get("energy")
        if q_true.shape[0] < 2:
            continue
        t_eff = min(int(horizon_steps), int(q_true.shape[0] - 1))
        if t_eff <= 0:
            continue

        dt = float(traj["dt_median"])
        q_state = torch.as_tensor(q_true[0:1], dtype=torch_dtype, device=device)
        qdot_state = torch.as_tensor(qdot_true[0:1], dtype=torch_dtype, device=device)
        state0 = wrap_coords_torch(torch.cat([q_state, qdot_state], dim=1))
        q_state = state0[:, :2]
        qdot_state = state0[:, 2:]
        q_pred = np.zeros((t_eff + 1, 2), dtype=np_dtype)
        qdot_pred = np.zeros((t_eff + 1, 2), dtype=np_dtype)
        q_pred[0] = q_true[0]
        qdot_pred[0] = qdot_true[0]

        # Create acceleration function for integrator
        def _accel(q_state: torch.Tensor, qdot_state: torch.Tensor) -> torch.Tensor:
            if norm_cfg is None:
                return qddot_from_lagrangian(
                    model=model,
                    q=q_state,
                    qdot=qdot_state,
                    for_training=False,
                )
            q_norm = (q_state - q_mean) / (q_std + eps)
            qdot_norm = (qdot_state - qdot_mean) / (qdot_std + eps)
            qddot_norm = qddot_from_lagrangian(
                model=model,
                q=q_norm,
                qdot=qdot_norm,
                for_training=False,
            )
            return qddot_norm * (qddot_std + eps) + qddot_mean

        rollout_failed = False
        for t in range(t_eff):
            try:
                q_next, qdot_next = step_dynamics(
                    q=q_state,
                    qdot=qdot_state,
                    dt=dt,
                    accel_fn=_accel,
                    integrator=integrator,
                )
            except RuntimeError:
                rollout_failed = True
                break

            state_next = torch.cat([q_next, qdot_next], dim=1)
            state_next = wrap_coords_torch(state_next)
            q_state = state_next[:, :2].detach()
            qdot_state = state_next[:, 2:].detach()
            q_pred[t + 1] = q_state.cpu().numpy()[0]
            qdot_pred[t + 1] = qdot_state.cpu().numpy()[0]

        if rollout_failed:
            continue

        q_true_seg = q_true[1 : t_eff + 1]
        qdot_true_seg = qdot_true[1 : t_eff + 1]
        q_pred_seg = q_pred[1 : t_eff + 1]
        qdot_pred_seg = qdot_pred[1 : t_eff + 1]
        q_err = q_pred_seg - q_true_seg
        qdot_err = qdot_pred_seg - qdot_true_seg

        q_sq_err_chunks.append(np.square(q_err).astype(np_dtype, copy=False))
        q_abs_err_chunks.append(np.abs(q_err).astype(np_dtype, copy=False))
        qdot_sq_err_chunks.append(np.square(qdot_err).astype(np_dtype, copy=False))
        qdot_abs_err_chunks.append(np.abs(qdot_err).astype(np_dtype, copy=False))
        final_q_abs_err.append(np.abs(q_err[-1]).astype(np_dtype, copy=False))
        final_qdot_abs_err.append(np.abs(qdot_err[-1]).astype(np_dtype, copy=False))
        dt_values.append(dt)
        used_files.append(file_path.name)

        if energy_true is not None:
            energy_arr = np.asarray(energy_true, dtype=np.float64)
            n_energy = min(len(energy_arr), len(q_pred))
            energy_chunk = _energy_errors_for_states(
                model=model,
                q=q_pred[:n_energy],
                qdot=qdot_pred[:n_energy],
                energy_true=energy_arr[:n_energy],
                device=device,
                torch_dtype=torch_dtype,
            )
            if energy_chunk is not None:
                abs_err, rel_err = energy_chunk
                energy_abs_err_chunks.append(abs_err)
                energy_rel_err_chunks.append(rel_err)

    if not used_files:
        raise RuntimeError("Rollout evaluation produced no valid trajectories.")

    q_sq = np.concatenate(q_sq_err_chunks, axis=0)
    q_abs = np.concatenate(q_abs_err_chunks, axis=0)
    qdot_sq = np.concatenate(qdot_sq_err_chunks, axis=0)
    qdot_abs = np.concatenate(qdot_abs_err_chunks, axis=0)
    final_q_abs = np.stack(final_q_abs_err, axis=0)
    final_qdot_abs = np.stack(final_qdot_abs_err, axis=0)

    mse_q = np.mean(q_sq, axis=0)
    mae_q = np.mean(q_abs, axis=0)
    mse_qdot = np.mean(qdot_sq, axis=0)
    mae_qdot = np.mean(qdot_abs, axis=0)
    dt_report = float(np.median(np.asarray(dt_values, dtype=np_dtype)))

    metrics = {
        "rollout_mse_q_1": float(mse_q[0]),
        "rollout_mse_q_2": float(mse_q[1]),
        "rollout_mse_q_mean": _safe_mean(mse_q),
        "rollout_mae_q_1": float(mae_q[0]),
        "rollout_mae_q_2": float(mae_q[1]),
        "rollout_mae_q_mean": _safe_mean(mae_q),
        "rollout_mse_qdot_1": float(mse_qdot[0]),
        "rollout_mse_qdot_2": float(mse_qdot[1]),
        "rollout_mse_qdot_mean": _safe_mean(mse_qdot),
        "rollout_mae_qdot_1": float(mae_qdot[0]),
        "rollout_mae_qdot_2": float(mae_qdot[1]),
        "rollout_mae_qdot_mean": _safe_mean(mae_qdot),
        "rollout_finalstep_mae_q_mean": _safe_mean(final_q_abs),
        "rollout_finalstep_mae_qdot_mean": _safe_mean(final_qdot_abs),
        "N": int(len(used_files)),
        "N_requested": int(n_traj),
        "T": int(horizon_steps),
        "dt": dt_report,
        "device": str(device),
        "metrics_space": "physical",
        "integrator": "semi_implicit_euler",
        "trajectory_files": used_files,
    }
    if energy_rel_err_chunks and energy_abs_err_chunks:
        energy_rel = np.concatenate(energy_rel_err_chunks, axis=0)
        energy_abs = np.concatenate(energy_abs_err_chunks, axis=0)
        metrics["energy_rel_error_mean"] = float(np.mean(energy_rel))
        metrics["energy_abs_error_mean"] = float(np.mean(energy_abs))
        metrics["energy_num_samples"] = int(len(energy_rel))
    else:
        metrics["energy_rel_error_mean"] = None
        metrics["energy_abs_error_mean"] = None
        metrics["energy_num_samples"] = 0
    return metrics


def main() -> None:
    args = parse_args()
    out_paths = ensure_output_dirs(Path(args.out_dir))
    out_dir = out_paths["out_dir"]
    plots_dir = out_paths["plots"]
    results_dir = out_paths["results"]

    run_config_path = out_dir / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {}
    split_cfg = run_config.get("split", {})
    hp_cfg = run_config.get("hyperparameters", {})

    split_by_traj_cfg = split_cfg.get("split_by_trajectory", hp_cfg.get("split_by_trajectory", False))
    if args.split_by_trajectory is None:
        split_by_trajectory = bool(split_by_traj_cfg)
    else:
        split_by_trajectory = bool(args.split_by_trajectory)
        if split_cfg and split_by_trajectory != bool(split_by_traj_cfg):
            raise ValueError(
                "split_by_trajectory does not match run_config. "
                f"run_config={split_by_traj_cfg}, cli={split_by_trajectory}"
            )

    # Determine normalization config (use same as training run)
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
                from lnn.utils import load_normalization_config

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

    val_ratio = float(hp_cfg.get("val_ratio", 0.1))
    test_ratio = float(hp_cfg.get("test_ratio", 0.1))
    seed = int(hp_cfg.get("seed", 42))
    split_override = split_cfg if split_by_trajectory else None
    data_bundle = prepare_datasets(
        data_dir=Path(args.data_dir),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        split_by_trajectory=split_by_trajectory,
        np_dtype=np_dtype,
        torch_dtype=torch_dtype,
        split_override=split_override,
        normalize=normalize,
        normalization_file=Path(args.normalization_file) if args.normalization_file else None,
        out_dir=Path(args.out_dir),
    )

    summary = data_bundle["summary"]
    norm_cfg = data_bundle.get("normalization")
    if normalize and norm_cfg is None:
        raise RuntimeError("Normalization requested but normalization config not found.")

    print(f"Loaded files found: {summary['num_files_found']}")
    print(
        "Loaded samples: "
        f"train={summary['num_samples_train']}, "
        f"val={summary['num_samples_val']}, "
        f"test={summary['num_samples_test']}, "
        f"total={summary['num_samples_total']}"
    )

    test_loader = DataLoader(
        data_bundle["test_dataset"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    skipped = 0

    for batch in tqdm(test_loader, desc="Evaluating", leave=False):
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                state_x = batch[0]
                xdot_true = batch[1]
            else:
                raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)!r}")
        state_x = state_x.to(device)
        q = state_x[:, :2]
        qdot = state_x[:, 2:]
        try:
            qddot_pred = qddot_from_lagrangian(
                model=model,
                q=q,
                qdot=qdot,
                for_training=False,
            )
        except RuntimeError:
            skipped += 1
            continue
        xdot_pred = torch.cat([qdot, qddot_pred], dim=1)

        true_np = xdot_true.detach().cpu().numpy()
        pred_np = xdot_pred.detach().cpu().numpy()
        mask = np.isfinite(true_np).all(axis=1) & np.isfinite(pred_np).all(axis=1)
        if not np.any(mask):
            continue
        all_true.append(true_np[mask])
        all_pred.append(pred_np[mask])

    if not all_true:
        raise RuntimeError("No valid predictions generated on test split.")

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    if norm_cfg is not None:
        y_true_phys = denormalize_xdot(y_true, norm_cfg)
        y_pred_phys = denormalize_xdot(y_pred, norm_cfg)
    else:
        y_true_phys = y_true
        y_pred_phys = y_pred

    print(
        "[eval] variable=xdot "
        f"| y_true shape={y_true_phys.shape} min={float(np.min(y_true_phys)):.6e} max={float(np.max(y_true_phys)):.6e} "
        f"| y_pred shape={y_pred_phys.shape} min={float(np.min(y_pred_phys)):.6e} max={float(np.max(y_pred_phys)):.6e}"
    )

    metrics = _compute_vector_metrics(
        y_true=y_true_phys,
        y_pred=y_pred_phys,
        labels=["qdot_1", "qdot_2", "qddot_1", "qddot_2"],
    )
    metrics["num_test_samples_evaluated"] = int(y_true.shape[0])
    metrics["num_test_batches_skipped"] = int(skipped)

    rollout_metrics: dict[str, Any] | None = None
    if bool(args.rollout_eval):
        rollout_files = _resolve_rollout_files(
            data_dir=Path(args.data_dir),
            split_info=data_bundle.get("split_info", {}),
            split_cfg=split_cfg,
        )
        rollout_metrics = _compute_rollout_metrics(
            model=model,
            rollout_files=rollout_files,
            device=device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
            n_traj=int(args.rollout_n_traj),
            horizon_steps=int(args.rollout_steps),
            integrator=args.integrator,
            norm_cfg=norm_cfg,
        )
        save_json(results_dir / "rollout_metrics.json", rollout_metrics)
        print(
            "[rollout] "
            f"N={rollout_metrics['N']} "
            f"T={rollout_metrics['T']} "
            f"dt={rollout_metrics['dt']:.8f} "
            f"mae_q_mean={rollout_metrics['rollout_mae_q_mean']:.8e} "
            f"mae_qdot_mean={rollout_metrics['rollout_mae_qdot_mean']:.8e} "
            f"energy_rel={rollout_metrics['energy_rel_error_mean']}"
        )

    eval_payload = {
        "timestamp_utc": utc_timestamp(),
        "model_path": str(model_path),
        "split_by_trajectory": bool(split_by_trajectory),
        "metrics_variable": "xdot",
        "metrics_space": "physical_xdot",
        "metrics": metrics,
        "rollout": rollout_metrics,
    }
    save_json(results_dir / "eval_metrics.json", eval_payload)

    # Keep qddot component plots for direct visual comparability.
    plot_qddot_scatter(y_true_phys[:, 2:], y_pred_phys[:, 2:], plots_dir=plots_dir, prefix="eval")
    plot_error_histograms(y_pred_phys[:, 2:] - y_true_phys[:, 2:], out_path=plots_dir / "eval_error_hist.png")

    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = results_dir / "metrics.csv"
    if metrics_csv.exists():
        plot_loss_curves(metrics_csv, out_path=plots_dir / "train_val_loss.png")

    print(f"MSE(xdot mean): {metrics['mse_xdot_mean']:.8e}")
    print(f"MAE(xdot mean): {metrics['mae_xdot_mean']:.8e}")
    print(f"R2(xdot mean): {metrics['r2_xdot_mean']:.6f}")
    print(f"Eval metrics saved to: {results_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()
