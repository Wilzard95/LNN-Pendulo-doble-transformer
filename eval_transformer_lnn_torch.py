from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

from lnn.dynamics import state_delta_from_lagrangian, xdot_from_lagrangian
from lnn.repo_faithful_data import RepoDoublePendulumPhysics, simulate_trajectory
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Transformer-LNN with the same rollout and paper-style energy metrics.")
    parser.add_argument("--out_dir", type=str, default="experiments/compare_transformer_lnn_torch")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rollout_steps", type=int, default=200)
    parser.add_argument("--trajectory_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--paper_energy_num_ics", type=int, default=40)
    parser.add_argument("--paper_energy_steps", type=int, default=100)
    parser.add_argument("--paper_energy_seed", type=int, default=123)
    parser.add_argument("--make_plots", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=False)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        return torch.device("cpu")
    return torch.device(requested)


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    plots_dir = out_dir / "plots"
    results_dir = out_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, results_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_from_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if cache_path.is_dir():
        data, _ = load_paperlike_temporal_cache(cache_path)
        return data

    cached = np.load(cache_path)
    data = {
        "x": np.asarray(cached["x"], dtype=np.float32),
        "test_x": np.asarray(cached["test_x"], dtype=np.float32),
    }
    if "dx" in cached:
        data["dx"] = np.asarray(cached["dx"], dtype=np.float32)
        data["test_dx"] = np.asarray(cached["test_dx"], dtype=np.float32)
    if "xdot" in cached:
        data["xdot"] = np.asarray(cached["xdot"], dtype=np.float32)
        data["test_xdot"] = np.asarray(cached["test_xdot"], dtype=np.float32)
    return data


def sample_initial_states(seed: int, samples: int) -> np.ndarray:
    with jax.disable_jit():
        rng = jax.random.PRNGKey(int(seed)) + 502
        y0 = jnp.concatenate(
            [
                jax.random.uniform(rng, (int(samples), 2)) * 2.0 * np.pi,
                jax.random.uniform(rng + 1, (int(samples), 2)) * 0.1,
            ],
            axis=1,
        )
    return np.asarray(y0, dtype=np.float32)


def reconstruct_initial_states(data_seed: int, samples: int) -> np.ndarray:
    return sample_initial_states(seed=data_seed, samples=samples)


def wrap_angles(q: np.ndarray) -> np.ndarray:
    wrapped = q.copy()
    wrapped[:, :2] = (wrapped[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def wrapped_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def hamiltonian_np(
    state: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.8,
) -> np.ndarray:
    t1 = state[..., 0]
    t2 = state[..., 1]
    w1 = state[..., 2]
    w2 = state[..., 3]
    t1v = 0.5 * m1 * (l1 * w1) ** 2
    t2v = 0.5 * m2 * ((l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2))
    y1 = -l1 * np.cos(t1)
    y2 = y1 - l2 * np.cos(t2)
    v = m1 * g * y1 + m2 * g * y2
    return t1v + t2v + v


def max_potential_energy(m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.8) -> float:
    return float(m1 * g * l1 + m2 * g * (l1 + l2))


def _get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _get_plot_rollout():
    plotting_spec = importlib.util.spec_from_file_location("local_plotting", LOCAL_ROOT / "lnn" / "plotting.py")
    plotting_module = importlib.util.module_from_spec(plotting_spec)
    assert plotting_spec.loader is not None
    plotting_spec.loader.exec_module(plotting_module)
    return plotting_module.plot_rollout


def plot_energy(time_s: np.ndarray, true_energy: np.ndarray, pred_energy: np.ndarray, out_path: Path) -> Path:
    plt = _get_pyplot()
    ref = max(1e-9, float(np.mean(np.abs(true_energy))))
    rel_err = np.abs(pred_energy - true_energy) / ref
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_s, true_energy, label="true")
    axes[0].plot(time_s, pred_energy, label="pred")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[0].set_ylabel("energy")
    axes[1].plot(time_s, rel_err, color="tab:red")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel("time_s")
    axes[1].set_ylabel("relative_error")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_paper_energy_curve(time_s: np.ndarray, mean_frac: np.ndarray, median_frac: np.ndarray, p90_frac: np.ndarray, out_path: Path) -> Path:
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, mean_frac, label="mean")
    ax.plot(time_s, median_frac, label="median")
    ax.plot(time_s, p90_frac, label="p90", linestyle="--")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("time_s")
    ax.set_ylabel("abs energy discrepancy / max potential")
    ax.set_title("Paper-style energy discrepancy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> torch.nn.Module:
    cfg = ckpt["model_config"]
    architecture = str(cfg.get("architecture", "transformer_lagrangian"))
    if architecture == "structured_transformer_lagrangian_tv":
        model = StructuredTransformerLagrangian(
            d_model=int(cfg["d_model"]),
            num_heads=int(cfg["num_heads"]),
            num_layers=int(cfg["num_layers"]),
            ff_multiplier=int(cfg["ff_multiplier"]),
            dropout=float(cfg.get("dropout", 0.0)),
            mass_eps=float(cfg.get("mass_eps", 1.0e-3)),
        ).to(device)
    else:
        model = TransformerLagrangian(
            d_model=int(cfg["d_model"]),
            num_heads=int(cfg["num_heads"]),
            num_layers=int(cfg["num_layers"]),
            ff_multiplier=int(cfg["ff_multiplier"]),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_batch(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int, dt: float, n_updates: int, objective: str) -> np.ndarray:
    preds = []
    for start in range(0, x.shape[0], batch_size):
        end = min(x.shape[0], start + batch_size)
        batch_np = np.asarray(x[start:end], dtype=np.float32).copy()
        batch_np[:, :2] = (batch_np[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
        batch = torch.from_numpy(batch_np).to(device)
        if str(objective) == "xdot":
            pred = xdot_from_lagrangian(model=model, q=batch[:, :2], qdot=batch[:, 2:], for_training=False)
        else:
            pred = state_delta_from_lagrangian(
                model=model,
                q=batch[:, :2],
                qdot=batch[:, 2:],
                dt=float(dt),
                n_updates=int(n_updates),
                for_training=False,
            )
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def rk4_step(model: torch.nn.Module, state: torch.Tensor, dt: float) -> torch.Tensor:
    def fn(s: torch.Tensor) -> torch.Tensor:
        wrapped = s.clone()
        wrapped[:, :2] = torch.remainder(wrapped[:, :2] + np.pi, 2.0 * np.pi) - np.pi
        return xdot_from_lagrangian(model=model, q=wrapped[:, :2], qdot=wrapped[:, 2:], for_training=False)

    k1 = fn(state)
    k2 = fn(state + 0.5 * float(dt) * k1)
    k3 = fn(state + 0.5 * float(dt) * k2)
    k4 = fn(state + float(dt) * k3)
    return state + (float(dt) / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_model(model: torch.nn.Module, initial_state: np.ndarray, steps: int, device: torch.device, dt: float, n_updates: int, objective: str) -> np.ndarray:
    state = torch.from_numpy(initial_state.astype(np.float32))[None, :].to(device)
    out = np.zeros((int(steps) + 1, 4), dtype=np.float64)
    out[0] = state.detach().cpu().numpy()[0]
    for idx in range(int(steps)):
        if str(objective) == "xdot":
            state = rk4_step(model, state, dt=float(dt))
        else:
            model_state = state.clone()
            model_state[:, :2] = torch.remainder(model_state[:, :2] + np.pi, 2.0 * np.pi) - np.pi
            delta = state_delta_from_lagrangian(
                model=model,
                q=model_state[:, :2],
                qdot=model_state[:, 2:],
                dt=float(dt),
                n_updates=int(n_updates),
                for_training=False,
            )
            state = state + delta
        out[idx + 1] = state.detach().cpu().numpy()[0]
    return out


def rollout_model_batch(
    model: torch.nn.Module,
    initial_states: np.ndarray,
    steps: int,
    device: torch.device,
    dt: float,
    n_updates: int,
    objective: str,
    batch_size: int = 64,
) -> np.ndarray:
    all_out = []
    for start in range(0, initial_states.shape[0], batch_size):
        end = min(initial_states.shape[0], start + batch_size)
        state = torch.from_numpy(initial_states[start:end].astype(np.float32)).to(device)
        out = np.zeros((state.shape[0], int(steps) + 1, 4), dtype=np.float64)
        out[:, 0] = state.detach().cpu().numpy()
        for idx in range(int(steps)):
            if str(objective) == "xdot":
                state = rk4_step(model, state, dt=float(dt))
            else:
                model_state = state.clone()
                model_state[:, :2] = torch.remainder(model_state[:, :2] + np.pi, 2.0 * np.pi) - np.pi
                delta = state_delta_from_lagrangian(
                    model=model,
                    q=model_state[:, :2],
                    qdot=model_state[:, 2:],
                    dt=float(dt),
                    n_updates=int(n_updates),
                    for_training=False,
                )
                state = state + delta
            out[:, idx + 1] = state.detach().cpu().numpy()
        all_out.append(out)
    return np.concatenate(all_out, axis=0)


def main() -> None:
    args = parse_args()
    configure_torch_attention_backends()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (LOCAL_ROOT / out_dir).resolve()
    plots_dir, results_dir = ensure_dirs(out_dir)

    model_path = Path(args.model_path) if args.model_path is not None else out_dir / "checkpoints" / "model_best.pth"
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    ckpt = torch.load(model_path, map_location="cpu")
    device = resolve_device(args.device)
    model = build_model_from_checkpoint(ckpt, device)

    dataset_summary = ckpt["dataset_summary"]
    cache_path = Path(dataset_summary["cache_path"])
    dataset = load_dataset_from_cache(cache_path)

    train_args = ckpt["train_args"]
    dt = float(train_args["dt"])
    n_updates = int(train_args["n_updates"])
    objective = str(train_args.get("objective", dataset_summary.get("objective", "delta")))
    target_key = "test_xdot" if objective == "xdot" else "test_dx"

    pred_test_y = predict_batch(
        model,
        dataset["test_x"],
        device=device,
        batch_size=int(args.batch_size),
        dt=dt,
        n_updates=n_updates,
        objective=objective,
    )
    true_test_y = dataset[target_key]
    delta_err = pred_test_y - true_test_y
    delta_abs = np.abs(delta_err)
    delta_sq = delta_err**2
    one_step_metrics = {
        "objective": str(objective),
        "target_mae_mean": float(np.mean(delta_abs)),
        "target_mse_mean": float(np.mean(delta_sq)),
        "delta_mae_mean": float(np.mean(delta_abs)),
        "delta_mse_mean": float(np.mean(delta_sq)),
        "delta_mae_q_mean": float(np.mean(delta_abs[:, :2])),
        "delta_mae_qdot_mean": float(np.mean(delta_abs[:, 2:])),
        "delta_mse_q_mean": float(np.mean(delta_sq[:, :2])),
        "delta_mse_qdot_mean": float(np.mean(delta_sq[:, 2:])),
        "num_test_samples": int(dataset["test_x"].shape[0]),
        "batch_size": int(args.batch_size),
    }
    (results_dir / "eval_metrics.json").write_text(json.dumps(one_step_metrics, indent=2), encoding="utf-8")

    dataset_mode = str(dataset_summary.get("dataset_mode", "repo_transitions"))
    if dataset_mode == "paperlike_instantaneous":
        trajectory_id = int(args.trajectory_id) if args.trajectory_id is not None else 0
        initial_state = np.asarray(dataset["test_x"][trajectory_id], dtype=np.float32)
    else:
        reference_out_dir = Path(ckpt["reference_out_dir"])
        reference_run_config = load_json(reference_out_dir / "run_config.json")
        total_samples = int(reference_run_config["args"]["samples"])
        test_split = float(reference_run_config["args"]["test_split"])
        first_test_traj = int(total_samples * test_split)
        trajectory_id = int(args.trajectory_id) if args.trajectory_id is not None else first_test_traj
        initial_states = reconstruct_initial_states(int(dataset_summary["data_seed"]), total_samples)
        initial_state = initial_states[trajectory_id]

    rollout_steps = int(args.rollout_steps)
    rollout_times = np.arange(rollout_steps + 1, dtype=np.float64) * dt
    physics = RepoDoublePendulumPhysics()
    true_state = simulate_trajectory(initial_state=initial_state, times=rollout_times, physics=physics)
    pred_state = rollout_model(
        model,
        initial_state=initial_state,
        steps=rollout_steps,
        device=device,
        dt=dt,
        n_updates=n_updates,
        objective=objective,
    )

    wrapped_true = wrap_angles(true_state)
    wrapped_pred = wrap_angles(pred_state)
    wrapped_err = wrapped_pred - wrapped_true
    wrapped_err[:, :2] = wrapped_angle_diff(wrapped_pred[:, :2], wrapped_true[:, :2])

    true_energy = hamiltonian_np(true_state)
    pred_energy = hamiltonian_np(pred_state)
    energy_ref = np.maximum(np.abs(true_energy), 1e-9)
    energy_rel_err = np.abs(pred_energy - true_energy) / energy_ref

    rollout_metrics = {
        "trajectory_id": int(trajectory_id),
        "rollout_steps": int(rollout_steps),
        "rollout_dt": float(dt),
        "rollout_mae_state_mean": float(np.mean(np.abs(pred_state - true_state))),
        "rollout_mae_q_mean": float(np.mean(np.abs(pred_state[:, :2] - true_state[:, :2]))),
        "rollout_mae_qdot_mean": float(np.mean(np.abs(pred_state[:, 2:] - true_state[:, 2:]))),
        "rollout_wrapped_mae_theta1": float(np.mean(np.abs(wrapped_err[:, 0]))),
        "rollout_wrapped_mae_theta2": float(np.mean(np.abs(wrapped_err[:, 1]))),
        "rollout_mae_omega1": float(np.mean(np.abs(wrapped_err[:, 2]))),
        "rollout_mae_omega2": float(np.mean(np.abs(wrapped_err[:, 3]))),
        "rollout_final_wrapped_mae_theta_mean": float(np.mean(np.abs(wrapped_err[-1, :2]))),
        "energy_rel_mae_mean": float(np.mean(energy_rel_err)),
        "energy_rel_mae_final": float(energy_rel_err[-1]),
    }
    (results_dir / "rollout_metrics.json").write_text(json.dumps(rollout_metrics, indent=2), encoding="utf-8")

    paper_num_ics = int(args.paper_energy_num_ics)
    paper_steps = int(args.paper_energy_steps)
    paper_seed = int(args.paper_energy_seed)
    paper_times = np.arange(paper_steps + 1, dtype=np.float64) * dt
    paper_initial_states = sample_initial_states(seed=paper_seed, samples=paper_num_ics)
    true_batch = np.stack(
        [simulate_trajectory(initial_state=paper_initial_states[i], times=paper_times, physics=physics) for i in range(paper_num_ics)],
        axis=0,
    )
    pred_batch = rollout_model_batch(
        model,
        initial_states=paper_initial_states,
        steps=paper_steps,
        device=device,
        dt=dt,
        n_updates=n_updates,
        objective=objective,
    )

    true_energy_batch = hamiltonian_np(true_batch)
    pred_energy_batch = hamiltonian_np(pred_batch)
    energy_gap = np.abs(pred_energy_batch - true_energy_batch)
    energy_norm = max_potential_energy()
    energy_frac = energy_gap / max(energy_norm, 1e-9)
    mean_curve = np.mean(energy_frac, axis=0)
    median_curve = np.median(energy_frac, axis=0)
    p90_curve = np.quantile(energy_frac, 0.90, axis=0)
    per_traj_mean = np.mean(energy_frac, axis=1)
    per_traj_final = energy_frac[:, -1]

    paper_energy_metrics = {
        "num_initial_conditions": paper_num_ics,
        "rollout_steps": paper_steps,
        "rollout_dt": float(dt),
        "paper_energy_seed": paper_seed,
        "max_potential_energy": float(energy_norm),
        "mean_abs_energy_discrepancy_frac": float(np.mean(energy_frac)),
        "median_abs_energy_discrepancy_frac": float(np.median(energy_frac)),
        "p90_abs_energy_discrepancy_frac": float(np.quantile(energy_frac, 0.90)),
        "final_mean_abs_energy_discrepancy_frac": float(np.mean(per_traj_final)),
        "best_ic_mean_abs_energy_discrepancy_frac": float(np.min(per_traj_mean)),
        "worst_ic_mean_abs_energy_discrepancy_frac": float(np.max(per_traj_mean)),
    }
    (results_dir / "paper_energy_metrics.json").write_text(json.dumps(paper_energy_metrics, indent=2), encoding="utf-8")

    stem = f"official_rollout_traj_{trajectory_id:03d}"
    csv_path = results_dir / f"{stem}.csv"
    rollout_plot_path = plots_dir / f"{stem}.png"
    energy_plot_path = plots_dir / f"{stem}_energy.png"
    paper_curve_csv = results_dir / "paper_energy_curve.csv"
    paper_per_ic_csv = results_dir / "paper_energy_per_ic.csv"
    paper_curve_path = plots_dir / "paper_energy_curve.png"

    pd.DataFrame(
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
    ).to_csv(csv_path, index=False)
    pd.DataFrame(
        {
            "time_s": paper_times,
            "mean_abs_energy_discrepancy_frac": mean_curve,
            "median_abs_energy_discrepancy_frac": median_curve,
            "p90_abs_energy_discrepancy_frac": p90_curve,
        }
    ).to_csv(paper_curve_csv, index=False)
    pd.DataFrame(
        {
            "ic_index": np.arange(paper_num_ics, dtype=np.int64),
            "theta1_0": paper_initial_states[:, 0],
            "theta2_0": paper_initial_states[:, 1],
            "omega1_0": paper_initial_states[:, 2],
            "omega2_0": paper_initial_states[:, 3],
            "mean_abs_energy_discrepancy_frac": per_traj_mean,
            "final_abs_energy_discrepancy_frac": per_traj_final,
        }
    ).to_csv(paper_per_ic_csv, index=False)

    plot_status = "disabled"
    if bool(args.make_plots):
        try:
            plot_rollout = _get_plot_rollout()
            plot_rollout(rollout_times, wrapped_true, wrapped_pred, out_path=rollout_plot_path)
            plot_energy(rollout_times, true_energy, pred_energy, out_path=energy_plot_path)
            plot_paper_energy_curve(paper_times, mean_curve, median_curve, p90_curve, paper_curve_path)
            plot_status = "saved"
        except BaseException as exc:
            plot_status = f"skipped ({type(exc).__name__}: {exc})"

    print(f"Model: {model_path}")
    print(f"Model kind: {ckpt.get('model_kind', 'transformer_lnn_torch')}")
    print(f"Architecture: {ckpt['model_config'].get('architecture', 'transformer_lagrangian')}")
    print(f"Objective: {objective}")
    print(f"Delta MAE mean: {one_step_metrics['delta_mae_mean']:.8e}")
    print(f"Delta MAE qdot mean: {one_step_metrics['delta_mae_qdot_mean']:.8e}")
    print(f"Rollout wrapped theta1 MAE: {rollout_metrics['rollout_wrapped_mae_theta1']:.8e}")
    print(f"Rollout wrapped theta2 MAE: {rollout_metrics['rollout_wrapped_mae_theta2']:.8e}")
    print(f"Rollout qdot MAE mean: {rollout_metrics['rollout_mae_qdot_mean']:.8e}")
    print(f"Energy relative MAE mean: {rollout_metrics['energy_rel_mae_mean']:.8e}")
    print(f"Paper-style mean energy discrepancy: {paper_energy_metrics['mean_abs_energy_discrepancy_frac']:.8e}")
    print(f"Saved paper energy metrics: {results_dir / 'paper_energy_metrics.json'}")
    print(f"Plot status: {plot_status}")


if __name__ == "__main__":
    main()
