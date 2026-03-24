from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OFFICIAL_REPO_ROOT = Path(__file__).resolve().parent / "official_lagrangian_nns"
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent
if str(OFFICIAL_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_ROOT))

import jax
import jax.numpy as jnp

from examples.double_pendulum.data import get_trajectory_analytic
from examples.double_pendulum.physics import hamiltonian_fn
from examples.hyperopt import HyperparameterSearch as hyper
from lnn.utils import wrap_coords
from lnn.core import lagrangian_eom_rk4, raw_lagrangian_eom
from direct_delta_models import build_model
from paperlike_double_pendulum import load_paperlike_temporal_cache

_plotting_spec = importlib.util.spec_from_file_location("local_plotting", LOCAL_PROJECT_ROOT / "lnn" / "plotting.py")
_plotting_module = importlib.util.module_from_spec(_plotting_spec)
assert _plotting_spec.loader is not None
_plotting_spec.loader.exec_module(_plotting_module)
plot_rollout = _plotting_module.plot_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the official JAX-compatible double pendulum checkpoint.")
    parser.add_argument("--out_dir", type=str, default="experiments/official_jax_cpu_retry")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--rollout_steps", type=int, default=200)
    parser.add_argument("--trajectory_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--one_step_eval_samples", type=int, default=0)
    parser.add_argument("--paper_energy_num_ics", type=int, default=40)
    parser.add_argument("--paper_energy_steps", type=int, default=100)
    parser.add_argument("--paper_energy_seed", type=int, default=123)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    plots_dir = out_dir / "plots"
    results_dir = out_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, results_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wrap_angles(q: np.ndarray) -> np.ndarray:
    wrapped = q.copy()
    wrapped[:, :2] = (wrapped[:, :2] + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def wrapped_angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def load_dataset_from_cache(cache_path: Path) -> dict[str, jnp.ndarray]:
    if cache_path.is_dir():
        data, _ = load_paperlike_temporal_cache(cache_path)
        return data

    cached = np.load(cache_path)
    data = {
        "x": jnp.asarray(cached["x"]),
        "t": jnp.asarray(cached["t"]),
        "test_x": jnp.asarray(cached["test_x"]),
        "test_t": jnp.asarray(cached["test_t"]),
    }
    if "dx" in cached:
        data["dx"] = jnp.asarray(cached["dx"])
        data["test_dx"] = jnp.asarray(cached["test_dx"])
    if "xdot" in cached:
        data["xdot"] = jnp.asarray(cached["xdot"])
        data["test_xdot"] = jnp.asarray(cached["test_xdot"])
    return data


def reconstruct_initial_states(data_seed: int, samples: int) -> np.ndarray:
    rng = jax.random.PRNGKey(int(data_seed)) + 502
    y0 = jnp.concatenate(
        [
            jax.random.uniform(rng, (int(samples), 2)) * 2.0 * np.pi,
            jax.random.uniform(rng + 1, (int(samples), 2)) * 0.1,
        ],
        axis=1,
    )
    return np.asarray(y0)


def sample_initial_states(seed: int, samples: int) -> np.ndarray:
    rng = jax.random.PRNGKey(int(seed)) + 502
    y0 = jnp.concatenate(
        [
            jax.random.uniform(rng, (int(samples), 2)) * 2.0 * np.pi,
            jax.random.uniform(rng + 1, (int(samples), 2)) * 0.1,
        ],
        axis=1,
    )
    return np.asarray(y0)


def make_predictors(train_args: dict, params):
    args_obj = hyper.ObjectView(train_args)
    _, nn_forward_fn = hyper.extended_mlp(args_obj)
    hyper.nn_forward_fn = nn_forward_fn
    dynamics = hyper.learned_dynamics(params)
    objective = str(train_args.get("objective", "delta"))

    @jax.jit
    def predict_delta_single(state: jnp.ndarray) -> jnp.ndarray:
        if objective == "xdot":
            return raw_lagrangian_eom(dynamics, state)
        return lagrangian_eom_rk4(dynamics, state, Dt=float(train_args["dt"]), n_updates=int(train_args["n_updates"]))

    @jax.jit
    def predict_delta_batch(state_batch: jnp.ndarray) -> jnp.ndarray:
        if objective == "xdot":
            return jax.vmap(partial(raw_lagrangian_eom, dynamics))(state_batch)
        return jax.vmap(partial(lagrangian_eom_rk4, dynamics, Dt=float(train_args["dt"]), n_updates=int(train_args["n_updates"])))(state_batch)

    return predict_delta_single, predict_delta_batch


def make_direct_predictors(model_kind: str, model_config: dict[str, object], params):
    _, apply_fn = build_model(model_kind, model_config)

    @jax.jit
    def predict_delta_single(state: jnp.ndarray) -> jnp.ndarray:
        wrapped = wrap_coords(state)
        return jnp.squeeze(apply_fn(params, wrapped[None, :]), axis=0)

    @jax.jit
    def predict_delta_batch(state_batch: jnp.ndarray) -> jnp.ndarray:
        wrapped = jax.vmap(wrap_coords)(state_batch)
        return apply_fn(params, wrapped)

    return predict_delta_single, predict_delta_batch


def batch_predict_delta(predict_delta_batch, state_x: jnp.ndarray, batch_size: int) -> np.ndarray:
    total = int(state_x.shape[0])
    preds = []
    for start in range(0, total, int(batch_size)):
        end = min(total, start + int(batch_size))
        preds.append(np.asarray(predict_delta_batch(state_x[start:end])))
    return np.concatenate(preds, axis=0)


def maybe_subsample_pair(x: np.ndarray | jnp.ndarray, y: np.ndarray | jnp.ndarray, limit: int) -> tuple[np.ndarray | jnp.ndarray, np.ndarray | jnp.ndarray]:
    total = int(len(x))
    if int(limit) <= 0 or int(limit) >= total:
        return x, y
    idx = np.linspace(0, total - 1, int(limit), dtype=np.float64)
    idx = np.round(idx).astype(np.int64, copy=False)
    return x[idx], y[idx]


def rk4_step_single(predict_xdot_single, state: jnp.ndarray, dt: float) -> jnp.ndarray:
    k1 = jnp.asarray(predict_xdot_single(state))
    k2 = jnp.asarray(predict_xdot_single(state + 0.5 * dt * k1))
    k3 = jnp.asarray(predict_xdot_single(state + 0.5 * dt * k2))
    k4 = jnp.asarray(predict_xdot_single(state + dt * k3))
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_step_batch(predict_xdot_batch, state: jnp.ndarray, dt: float) -> jnp.ndarray:
    k1 = jnp.asarray(predict_xdot_batch(state))
    k2 = jnp.asarray(predict_xdot_batch(state + 0.5 * dt * k1))
    k3 = jnp.asarray(predict_xdot_batch(state + 0.5 * dt * k2))
    k4 = jnp.asarray(predict_xdot_batch(state + dt * k3))
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_model(predict_delta_single, initial_state: np.ndarray, steps: int, dt: float, objective: str) -> np.ndarray:
    state = jnp.asarray(initial_state)
    out = np.zeros((int(steps) + 1, 4), dtype=np.float64)
    out[0] = np.asarray(state)
    for idx in range(int(steps)):
        if str(objective) == "xdot":
            state = rk4_step_single(predict_delta_single, state, float(dt))
        else:
            delta = np.asarray(predict_delta_single(state))
            state = state + jnp.asarray(delta)
        out[idx + 1] = np.asarray(state)
    return out


def rollout_model_batch(predict_delta_batch, initial_states: np.ndarray, steps: int, dt: float, objective: str) -> np.ndarray:
    state = jnp.asarray(initial_states)
    out = np.zeros((int(initial_states.shape[0]), int(steps) + 1, 4), dtype=np.float64)
    out[:, 0] = np.asarray(state)
    for idx in range(int(steps)):
        if str(objective) == "xdot":
            state = rk4_step_batch(predict_delta_batch, state, float(dt))
        else:
            delta = np.asarray(predict_delta_batch(state))
            state = state + jnp.asarray(delta)
        out[:, idx + 1] = np.asarray(state)
    return out


def plot_energy(time_s: np.ndarray, true_energy: np.ndarray, pred_energy: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref = max(1e-9, float(np.mean(np.abs(true_energy))))
    rel_err = np.abs(pred_energy - true_energy) / ref

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time_s, true_energy, label="true")
    axes[0].plot(time_s, pred_energy, label="pred")
    axes[0].set_ylabel("energy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(time_s, rel_err, color="tab:red")
    axes[1].set_xlabel("time_s")
    axes[1].set_ylabel("relative_error")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Energy comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def max_potential_energy(m1: float = 1.0, m2: float = 1.0, l1: float = 1.0, l2: float = 1.0, g: float = 9.8) -> float:
    return float(m1 * g * l1 + m2 * g * (l1 + l2))


def plot_paper_energy_curve(
    time_s: np.ndarray,
    mean_frac: np.ndarray,
    median_frac: np.ndarray,
    p90_frac: np.ndarray,
    out_path: Path,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(time_s, mean_frac, label="mean")
    ax.plot(time_s, median_frac, label="median")
    ax.plot(time_s, p90_frac, label="p90", linestyle="--")
    ax.set_xlabel("time_s")
    ax.set_ylabel("abs energy discrepancy / max potential")
    ax.set_title("Paper-style energy discrepancy across random initial conditions")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
    plots_dir, results_dir = ensure_dirs(out_dir)

    run_config = load_json(out_dir / "run_config.json")
    attempts_summary_path = results_dir / "attempts_summary.json"
    if attempts_summary_path.exists():
        attempts_summary = load_json(attempts_summary_path)
        dataset_summary = attempts_summary["dataset_summary"]
        best_attempt_name = attempts_summary.get("best_attempt", {}).get("attempt_name", "unknown")
    else:
        attempts_summary = None
        dataset_summary = load_json(results_dir / "dataset_summary.json")
        best_attempt_name = "unknown"

    model_path = Path(args.model_path) if args.model_path is not None else out_dir / "checkpoints" / "model_best.pkl"
    if not model_path.is_absolute():
        model_path = (Path.cwd() / model_path).resolve()
    ckpt = pickle.load(open(model_path, "rb"))
    params = ckpt["params"]
    train_args = ckpt["train_args"]
    model_kind = str(ckpt.get("model_kind", "gln"))
    model_config = ckpt.get("model_config", {})

    dataset = load_dataset_from_cache(Path(dataset_summary["cache_path"]))
    if model_kind == "gln":
        predict_delta_single, predict_delta_batch = make_predictors(train_args, params)
    else:
        predict_delta_single, predict_delta_batch = make_direct_predictors(model_kind, model_config, params)

    objective = str(train_args.get("objective", "delta"))
    target_key = "test_xdot" if objective == "xdot" else "test_dx"
    eval_test_x, eval_test_y = maybe_subsample_pair(dataset["test_x"], dataset[target_key], int(args.one_step_eval_samples))
    pred_test_dx = batch_predict_delta(predict_delta_batch, eval_test_x, batch_size=int(args.batch_size))
    true_test_dx = np.asarray(eval_test_y)
    delta_err = pred_test_dx - true_test_dx
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
        "num_test_samples": int(np.asarray(eval_test_x).shape[0]),
        "batch_size": int(args.batch_size),
    }
    (results_dir / "eval_metrics.json").write_text(json.dumps(one_step_metrics, indent=2), encoding="utf-8")

    dataset_mode = str(dataset_summary.get("dataset_mode", "repo_transitions"))
    if dataset_mode == "paperlike_instantaneous":
        trajectory_id = int(args.trajectory_id) if args.trajectory_id is not None else 0
        initial_state = np.asarray(dataset["test_x"][trajectory_id])
    else:
        total_samples = int(run_config["args"]["samples"])
        test_split = float(run_config["args"]["test_split"])
        first_test_traj = int(total_samples * test_split)
        trajectory_id = int(args.trajectory_id) if args.trajectory_id is not None else first_test_traj
        initial_states = reconstruct_initial_states(int(dataset_summary["data_seed"]), total_samples)
        initial_state = initial_states[trajectory_id]
    rollout_steps = int(args.rollout_steps)
    rollout_times = np.arange(rollout_steps + 1, dtype=np.float64) * float(train_args["dt"])

    true_state = np.asarray(get_trajectory_analytic(jnp.asarray(initial_state), jnp.asarray(rollout_times)))
    pred_state = rollout_model(
        predict_delta_single,
        initial_state=initial_state,
        steps=rollout_steps,
        dt=float(train_args["dt"]),
        objective=str(objective),
    )

    wrapped_true = wrap_angles(true_state)
    wrapped_pred = wrap_angles(pred_state)
    wrapped_err = wrapped_pred - wrapped_true
    wrapped_err[:, :2] = wrapped_angle_diff(wrapped_pred[:, :2], wrapped_true[:, :2])

    true_energy = np.asarray(jax.vmap(lambda s: hamiltonian_fn(s[:2], s[2:]))(jnp.asarray(true_state)))
    pred_energy = np.asarray(jax.vmap(lambda s: hamiltonian_fn(s[:2], s[2:]))(jnp.asarray(pred_state)))
    energy_ref = np.maximum(np.abs(true_energy), 1e-9)
    energy_rel_err = np.abs(pred_energy - true_energy) / energy_ref

    rollout_metrics = {
        "trajectory_id": int(trajectory_id),
        "rollout_steps": int(rollout_steps),
        "rollout_dt": float(train_args["dt"]),
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
    paper_times = np.arange(paper_steps + 1, dtype=np.float64) * float(train_args["dt"])
    paper_initial_states = sample_initial_states(seed=paper_seed, samples=paper_num_ics)
    true_batch = np.asarray(hyper.vget(jnp.asarray(paper_initial_states), jnp.asarray(paper_times)))
    pred_batch = rollout_model_batch(
        predict_delta_batch,
        initial_states=paper_initial_states,
        steps=paper_steps,
        dt=float(train_args["dt"]),
        objective=str(objective),
    )

    energy_batch_fn = jax.vmap(jax.vmap(lambda s: hamiltonian_fn(s[:2], s[2:])))
    true_energy_batch = np.asarray(energy_batch_fn(jnp.asarray(true_batch)))
    pred_energy_batch = np.asarray(energy_batch_fn(jnp.asarray(pred_batch)))

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
        "rollout_dt": float(train_args["dt"]),
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
    paper_energy_curve_path = plots_dir / "paper_energy_curve.png"
    paper_energy_curve_csv = results_dir / "paper_energy_curve.csv"
    paper_energy_per_ic_csv = results_dir / "paper_energy_per_ic.csv"

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
    plot_rollout(rollout_times, wrapped_true, wrapped_pred, out_path=rollout_plot_path)
    plot_energy(rollout_times, true_energy, pred_energy, out_path=energy_plot_path)
    pd.DataFrame(
        {
            "time_s": paper_times,
            "mean_abs_energy_discrepancy_frac": mean_curve,
            "median_abs_energy_discrepancy_frac": median_curve,
            "p90_abs_energy_discrepancy_frac": p90_curve,
        }
    ).to_csv(paper_energy_curve_csv, index=False)
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
    ).to_csv(paper_energy_per_ic_csv, index=False)
    plot_paper_energy_curve(
        time_s=paper_times,
        mean_frac=mean_curve,
        median_frac=median_curve,
        p90_frac=p90_curve,
        out_path=paper_energy_curve_path,
    )

    print(f"Model: {model_path}")
    print(f"Model kind: {model_kind}")
    print(f"Objective: {objective}")
    print(f"Best attempt: {best_attempt_name}")
    print(f"Delta MAE mean: {one_step_metrics['delta_mae_mean']:.8e}")
    print(f"Delta MAE qdot mean: {one_step_metrics['delta_mae_qdot_mean']:.8e}")
    print(f"Rollout wrapped theta1 MAE: {rollout_metrics['rollout_wrapped_mae_theta1']:.8e}")
    print(f"Rollout wrapped theta2 MAE: {rollout_metrics['rollout_wrapped_mae_theta2']:.8e}")
    print(f"Rollout qdot MAE mean: {rollout_metrics['rollout_mae_qdot_mean']:.8e}")
    print(f"Energy relative MAE mean: {rollout_metrics['energy_rel_mae_mean']:.8e}")
    print(f"Paper-style mean energy discrepancy: {paper_energy_metrics['mean_abs_energy_discrepancy_frac']:.8e}")
    print(f"Paper-style final mean energy discrepancy: {paper_energy_metrics['final_mean_abs_energy_discrepancy_frac']:.8e}")
    print(f"Saved eval metrics: {results_dir / 'eval_metrics.json'}")
    print(f"Saved rollout metrics: {results_dir / 'rollout_metrics.json'}")
    print(f"Saved paper energy metrics: {results_dir / 'paper_energy_metrics.json'}")
    print(f"Saved rollout CSV: {csv_path}")
    print(f"Saved rollout plot: {rollout_plot_path}")
    print(f"Saved energy plot: {energy_plot_path}")
    print(f"Saved paper energy curve CSV: {paper_energy_curve_csv}")
    print(f"Saved paper energy per-IC CSV: {paper_energy_per_ic_csv}")
    print(f"Saved paper energy curve: {paper_energy_curve_path}")


if __name__ == "__main__":
    main()
