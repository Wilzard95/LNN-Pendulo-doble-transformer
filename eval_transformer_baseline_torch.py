from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch

LOCAL_ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO_ROOT = LOCAL_ROOT / "official_lagrangian_nns"

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from torch_delta_models import StateTransformerEncoder

_physics_spec = importlib.util.spec_from_file_location(
    "official_double_pendulum_physics",
    OFFICIAL_REPO_ROOT / "examples" / "double_pendulum" / "physics.py",
)
_physics_module = importlib.util.module_from_spec(_physics_spec)
assert _physics_spec.loader is not None
_physics_spec.loader.exec_module(_physics_module)
analytical_fn = _physics_module.analytical_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the PyTorch transformer baseline with the same metrics as the official-compatible LNN.")
    parser.add_argument("--out_dir", type=str, default="experiments/compare_transformer_torch")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rollout_steps", type=int, default=200)
    parser.add_argument("--trajectory_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    plots_dir = out_dir / "plots"
    results_dir = out_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, results_dir


def wrap_angles(q: np.ndarray) -> np.ndarray:
    wrapped = q.copy()
    wrapped[:, :2] = (wrapped[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def wrapped_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def load_dataset_from_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cached = np.load(cache_path)
    x = np.asarray(cached["x"], dtype=np.float32)
    dx = np.asarray(cached["dx"], dtype=np.float32)
    test_x = np.asarray(cached["test_x"], dtype=np.float32)
    test_dx = np.asarray(cached["test_dx"], dtype=np.float32)
    x[:, :2] = (x[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
    test_x[:, :2] = (test_x[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
    return x, dx, test_x, test_dx


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


def hamiltonian_np(state: np.ndarray, m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.8) -> np.ndarray:
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


@jax.jit
def get_trajectory_analytic(y0: jnp.ndarray, times: jnp.ndarray, **kwargs) -> jnp.ndarray:
    return odeint(analytical_fn, y0, t=times, rtol=1e-10, atol=1e-10, **kwargs)


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


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> nn.Module:
    cfg = ckpt["model_config"]
    model = StateTransformerEncoder(
        d_model=int(cfg["d_model"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        ff_multiplier=int(cfg["ff_multiplier"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    for start in range(0, x.shape[0], batch_size):
        end = min(x.shape[0], start + batch_size)
        batch = torch.from_numpy(x[start:end]).to(device)
        preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def rollout_model(model: nn.Module, initial_state: np.ndarray, steps: int, device: torch.device) -> np.ndarray:
    state = torch.from_numpy(initial_state.astype(np.float32))[None, :].to(device)
    out = np.zeros((int(steps) + 1, 4), dtype=np.float64)
    out[0] = state.detach().cpu().numpy()[0]
    for idx in range(int(steps)):
        wrapped = state.clone()
        wrapped[:, :2] = (wrapped[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
        delta = model(wrapped)
        state = state + delta
        out[idx + 1] = state.detach().cpu().numpy()[0]
    return out


@torch.no_grad()
def rollout_model_batch(model: nn.Module, initial_states: np.ndarray, steps: int, device: torch.device, batch_size: int = 256) -> np.ndarray:
    all_out = []
    for start in range(0, initial_states.shape[0], batch_size):
        end = min(initial_states.shape[0], start + batch_size)
        state = torch.from_numpy(initial_states[start:end].astype(np.float32)).to(device)
        out = np.zeros((state.shape[0], int(steps) + 1, 4), dtype=np.float64)
        out[:, 0] = state.detach().cpu().numpy()
        for idx in range(int(steps)):
            wrapped = state.clone()
            wrapped[:, :2] = (wrapped[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
            delta = model(wrapped)
            state = state + delta
            out[:, idx + 1] = state.detach().cpu().numpy()
        all_out.append(out)
    return np.concatenate(all_out, axis=0)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (LOCAL_ROOT / out_dir).resolve()
    plots_dir, results_dir = ensure_dirs(out_dir)

    model_path = Path(args.model_path) if args.model_path is not None else out_dir / "checkpoints" / "model_best.pt"
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()

    ckpt = torch.load(model_path, map_location="cpu")
    device = resolve_device(args.device)
    model = build_model_from_checkpoint(ckpt, device)

    dataset_summary = ckpt["dataset_summary"]
    cache_path = Path(dataset_summary["cache_path"])
    x, dx, test_x, test_dx = load_dataset_from_cache(cache_path)

    pred_test_dx = predict_batch(model, test_x, device=device, batch_size=int(args.batch_size))
    delta_err = pred_test_dx - test_dx
    delta_abs = np.abs(delta_err)
    delta_sq = delta_err**2
    one_step_metrics = {
        "delta_mae_mean": float(np.mean(delta_abs)),
        "delta_mse_mean": float(np.mean(delta_sq)),
        "delta_mae_q_mean": float(np.mean(delta_abs[:, :2])),
        "delta_mae_qdot_mean": float(np.mean(delta_abs[:, 2:])),
        "delta_mse_q_mean": float(np.mean(delta_sq[:, :2])),
        "delta_mse_qdot_mean": float(np.mean(delta_sq[:, 2:])),
        "num_test_samples": int(test_x.shape[0]),
        "batch_size": int(args.batch_size),
    }
    (results_dir / "eval_metrics.json").write_text(json.dumps(one_step_metrics, indent=2), encoding="utf-8")

    reference_out_dir = Path(ckpt["reference_out_dir"])
    reference_run_config = load_json(reference_out_dir / "run_config.json")
    total_samples = int(reference_run_config["args"]["samples"])
    test_split = float(reference_run_config["args"]["test_split"])
    first_test_traj = int(total_samples * test_split)
    trajectory_id = int(args.trajectory_id) if args.trajectory_id is not None else first_test_traj
    initial_states = reconstruct_initial_states(int(dataset_summary["data_seed"]), total_samples)
    initial_state = initial_states[trajectory_id]

    dt = float(ckpt["train_args"]["dt"])
    rollout_steps = int(args.rollout_steps)
    rollout_times = np.arange(rollout_steps + 1, dtype=np.float64) * dt
    true_state = np.asarray(get_trajectory_analytic(jnp.asarray(initial_state), jnp.asarray(rollout_times)))
    pred_state = rollout_model(model, initial_state=initial_state, steps=rollout_steps, device=device)

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
    vget = jax.jit(jax.vmap(get_trajectory_analytic, (0, None), 0), backend="cpu")
    true_batch = np.asarray(vget(jnp.asarray(paper_initial_states), jnp.asarray(paper_times)))
    pred_batch = rollout_model_batch(model, initial_states=paper_initial_states, steps=paper_steps, device=device)

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
    print("Model kind: baseline_transformer_torch")
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
