from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp


@dataclass
class RepoDoublePendulumPhysics:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.8


@dataclass
class RepoFaithfulDataConfig:
    seed: int = 0
    samples: int = 50
    duration_s: float = 50.0
    fps: int = 10
    lookahead: int = 1
    train_fraction: float = 0.9
    q_init_min: float = 0.0
    q_init_max: float = 2.0 * np.pi
    qdot_init_min: float = 0.0
    qdot_init_max: float = 0.1
    solver: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-10


def _analytical_xdot(state: np.ndarray, physics: RepoDoublePendulumPhysics) -> np.ndarray:
    t1, t2, w1, w2 = state
    a1 = (physics.l2 / physics.l1) * (physics.m2 / (physics.m1 + physics.m2)) * np.cos(t1 - t2)
    a2 = (physics.l1 / physics.l2) * np.cos(t1 - t2)
    f1 = (
        -(physics.l2 / physics.l1)
        * (physics.m2 / (physics.m1 + physics.m2))
        * (w2**2)
        * np.sin(t1 - t2)
        - (physics.g / physics.l1) * np.sin(t1)
    )
    f2 = (physics.l1 / physics.l2) * (w1**2) * np.sin(t1 - t2) - (physics.g / physics.l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1.0 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1.0 - a1 * a2)
    return np.asarray([w1, w2, g1, g2], dtype=np.float64)


def _make_times(cfg: RepoFaithfulDataConfig) -> np.ndarray:
    frames = int(cfg.fps * cfg.duration_s)
    if frames < 2:
        raise ValueError(f"Need at least 2 frames, got {frames}")
    return np.linspace(0.0, float(cfg.duration_s), frames, dtype=np.float64)


def _sample_initial_states(cfg: RepoFaithfulDataConfig) -> np.ndarray:
    rng = np.random.default_rng(int(cfg.seed))
    q = rng.uniform(float(cfg.q_init_min), float(cfg.q_init_max), size=(int(cfg.samples), 2))
    qdot = rng.uniform(float(cfg.qdot_init_min), float(cfg.qdot_init_max), size=(int(cfg.samples), 2))
    return np.concatenate([q, qdot], axis=1).astype(np.float64, copy=False)


def simulate_trajectory(
    initial_state: np.ndarray,
    times: np.ndarray,
    physics: RepoDoublePendulumPhysics,
    solver: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> np.ndarray:
    initial_state = np.asarray(initial_state, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)

    sol = solve_ivp(
        fun=lambda _t, y: _analytical_xdot(y, physics),
        t_span=(float(times[0]), float(times[-1])),
        y0=initial_state,
        t_eval=times,
        method=str(solver),
        rtol=float(rtol),
        atol=float(atol),
        dense_output=False,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"Trajectory integration failed: {sol.message}")
    return sol.y.T.astype(np.float64, copy=False)


def generate_trajectories(
    cfg: RepoFaithfulDataConfig,
    physics: RepoDoublePendulumPhysics | None = None,
) -> dict[str, Any]:
    physics = physics or RepoDoublePendulumPhysics()
    times = _make_times(cfg)
    initial_states = _sample_initial_states(cfg)

    trajectories = []
    for initial_state in initial_states:
        traj = simulate_trajectory(
            initial_state=initial_state,
            times=times,
            physics=physics,
            solver=cfg.solver,
            rtol=cfg.rtol,
            atol=cfg.atol,
        )
        trajectories.append(traj)

    stacked = np.stack(trajectories, axis=0).astype(np.float64, copy=False)
    return {
        "times": times,
        "initial_states": initial_states.astype(np.float64, copy=False),
        "trajectories": stacked,
        "physics": physics,
        "config": cfg,
    }


def build_repo_faithful_dataset(
    cfg: RepoFaithfulDataConfig,
    physics: RepoDoublePendulumPhysics | None = None,
    torch_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    generated = generate_trajectories(cfg=cfg, physics=physics)
    trajectories = generated["trajectories"]
    times = generated["times"]

    lookahead = int(cfg.lookahead)
    if trajectories.shape[1] <= lookahead:
        raise ValueError(f"Trajectory too short for lookahead={lookahead}")

    state = trajectories[:, :-lookahead, :]
    delta = trajectories[:, lookahead:, :] - state
    time_slice = np.tile(times[:-lookahead], int(cfg.samples))

    flat_state = state.reshape(-1, 4).astype(np.float32, copy=False)
    flat_delta = delta.reshape(-1, 4).astype(np.float32, copy=False)
    flat_time = time_slice.astype(np.float64, copy=False)

    transitions_per_trajectory = int(state.shape[1])
    trajectory_ids = np.repeat(np.arange(int(cfg.samples), dtype=np.int64), transitions_per_trajectory)

    total_samples = int(flat_state.shape[0])
    split_ix = int(total_samples * float(cfg.train_fraction))
    split_ix = max(1, min(split_ix, total_samples - 1))

    train_state = torch.from_numpy(flat_state[:split_ix]).to(dtype=torch_dtype)
    train_delta = torch.from_numpy(flat_delta[:split_ix]).to(dtype=torch_dtype)
    test_state = torch.from_numpy(flat_state[split_ix:]).to(dtype=torch_dtype)
    test_delta = torch.from_numpy(flat_delta[split_ix:]).to(dtype=torch_dtype)

    train_traj_count = split_ix // transitions_per_trajectory
    test_traj_ids = np.arange(train_traj_count, int(cfg.samples), dtype=np.int64)
    train_traj_ids = np.arange(0, train_traj_count, dtype=np.int64)
    effective_dt = float(np.median(times[lookahead:] - times[:-lookahead]))

    return {
        "train_x": train_state,
        "train_y": train_delta,
        "test_x": test_state,
        "test_y": test_delta,
        "all_times": times.astype(np.float64, copy=False),
        "all_trajectories": trajectories.astype(np.float64, copy=False),
        "all_initial_states": generated["initial_states"].astype(np.float64, copy=False),
        "train_trajectory_ids": train_traj_ids,
        "test_trajectory_ids": test_traj_ids,
        "trajectory_ids_flat": trajectory_ids,
        "time_flat": flat_time,
        "effective_dt": effective_dt,
        "lookahead": lookahead,
        "physics": generated["physics"],
        "data_config": cfg,
        "summary": {
            "samples": int(cfg.samples),
            "frames_per_trajectory": int(trajectories.shape[1]),
            "transitions_per_trajectory": transitions_per_trajectory,
            "total_samples": total_samples,
            "train_samples": int(train_state.shape[0]),
            "test_samples": int(test_state.shape[0]),
            "train_trajectories": int(len(train_traj_ids)),
            "test_trajectories": int(len(test_traj_ids)),
            "effective_dt": effective_dt,
        },
        "metadata": {
            "data_config": asdict(cfg),
            "physics": asdict(generated["physics"]),
        },
    }


def save_dataset_metadata(path: Path, dataset: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": dataset["summary"],
        "metadata": dataset["metadata"],
        "train_trajectory_ids": dataset["train_trajectory_ids"].tolist(),
        "test_trajectory_ids": dataset["test_trajectory_ids"].tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
