from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.integrate import solve_ivp


def sample_initial_states(seed: int, samples: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(int(samples), 2)).astype(np.float32)
    omega = rng.uniform(0.0, 0.1, size=(int(samples), 2)).astype(np.float32)
    return np.concatenate([theta, omega], axis=1).astype(np.float32, copy=False)


def analytical_xdot_np(
    states: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.8,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    t1 = states[:, 0]
    t2 = states[:, 1]
    w1 = states[:, 2]
    w2 = states[:, 3]

    a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
    a2 = (l1 / l2) * np.cos(t1 - t2)
    f1 = (
        -(l2 / l1)
        * (m2 / (m1 + m2))
        * (w2**2)
        * np.sin(t1 - t2)
        - (g / l1) * np.sin(t1)
    )
    f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1.0 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1.0 - a1 * a2)
    return np.stack([w1, w2, g1, g2], axis=1).astype(np.float32, copy=False)


def _analytical_xdot_single(
    state: np.ndarray,
    m1: float = 1.0,
    m2: float = 1.0,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.8,
) -> np.ndarray:
    t1, t2, w1, w2 = state
    a1 = (l2 / l1) * (m2 / (m1 + m2)) * np.cos(t1 - t2)
    a2 = (l1 / l2) * np.cos(t1 - t2)
    f1 = (
        -(l2 / l1)
        * (m2 / (m1 + m2))
        * (w2**2)
        * np.sin(t1 - t2)
        - (g / l1) * np.sin(t1)
    )
    f2 = (l1 / l2) * (w1**2) * np.sin(t1 - t2) - (g / l2) * np.sin(t2)
    g1 = (f1 - a1 * f2) / (1.0 - a1 * a2)
    g2 = (f2 - a2 * f1) / (1.0 - a1 * a2)
    return np.asarray([w1, w2, g1, g2], dtype=np.float64)


def _simulate_trajectory(initial_state: np.ndarray, times: np.ndarray) -> np.ndarray:
    initial_state = np.asarray(initial_state, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    sol = solve_ivp(
        fun=lambda _t, y: _analytical_xdot_single(y),
        t_span=(float(times[0]), float(times[-1])),
        y0=initial_state,
        t_eval=times,
        method="DOP853",
        rtol=1e-10,
        atol=1e-10,
        dense_output=False,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"Trajectory integration failed: {sol.message}")
    return sol.y.T.astype(np.float32, copy=False)


def build_paperlike_dataset(
    *,
    data_seed: int,
    samples: int,
    train_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    states = sample_initial_states(seed=int(data_seed), samples=int(samples))
    xdot = analytical_xdot_np(states)

    split_ix = int(len(states) * float(train_fraction))
    split_ix = max(1, min(split_ix, len(states) - 1))

    train_x = states[:split_ix]
    test_x = states[split_ix:]
    train_xdot = xdot[:split_ix]
    test_xdot = xdot[split_ix:]

    data = {
        "x": train_x,
        "xdot": train_xdot,
        "t": np.zeros(train_x.shape[0], dtype=np.float32),
        "test_x": test_x,
        "test_xdot": test_xdot,
        "test_t": np.zeros(test_x.shape[0], dtype=np.float32),
    }
    summary = {
        "data_seed": int(data_seed),
        "num_total_samples": int(states.shape[0]),
        "num_train_samples": int(train_x.shape[0]),
        "num_test_samples": int(test_x.shape[0]),
        "state_dim": int(train_x.shape[1]),
        "dataset_mode": "paperlike_instantaneous",
        "objective": "xdot",
    }
    return data, summary


def build_paperlike_temporal_dataset(
    *,
    data_seed: int,
    samples: int,
    trajectory_steps: int,
    dt: float,
    train_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    num_trajectories = int(samples)
    trajectory_steps = int(trajectory_steps)
    if num_trajectories < 2:
        raise ValueError("Need at least 2 trajectories")
    if trajectory_steps < 1:
        raise ValueError("Need at least 1 transition per trajectory")

    initial_states = sample_initial_states(seed=int(data_seed), samples=num_trajectories).astype(np.float64, copy=False)
    times = (np.arange(trajectory_steps + 1, dtype=np.float64) * float(dt)).astype(np.float64, copy=False)

    trajectories = []
    for initial_state in initial_states:
        traj = _simulate_trajectory(initial_state=initial_state, times=times)
        trajectories.append(traj.astype(np.float32, copy=False))
    trajectories = np.stack(trajectories, axis=0).astype(np.float32, copy=False)

    split_traj = int(num_trajectories * float(train_fraction))
    split_traj = max(1, min(split_traj, num_trajectories - 1))

    train_traj = trajectories[:split_traj]
    test_traj = trajectories[split_traj:]

    train_x = train_traj[:, :-1, :].reshape(-1, 4).astype(np.float32, copy=False)
    train_dx = (train_traj[:, 1:, :] - train_traj[:, :-1, :]).reshape(-1, 4).astype(np.float32, copy=False)
    test_x = test_traj[:, :-1, :].reshape(-1, 4).astype(np.float32, copy=False)
    test_dx = (test_traj[:, 1:, :] - test_traj[:, :-1, :]).reshape(-1, 4).astype(np.float32, copy=False)

    t_slice = np.tile(times[:-1].astype(np.float32, copy=False), split_traj)
    test_t_slice = np.tile(times[:-1].astype(np.float32, copy=False), num_trajectories - split_traj)

    data = {
        "x": train_x,
        "dx": train_dx,
        "t": t_slice,
        "test_x": test_x,
        "test_dx": test_dx,
        "test_t": test_t_slice,
    }
    summary = {
        "data_seed": int(data_seed),
        "num_total_trajectories": int(num_trajectories),
        "num_train_trajectories": int(split_traj),
        "num_test_trajectories": int(num_trajectories - split_traj),
        "num_total_samples": int(num_trajectories * trajectory_steps),
        "num_train_samples": int(train_x.shape[0]),
        "num_test_samples": int(test_x.shape[0]),
        "transitions_per_trajectory": int(trajectory_steps),
        "state_dim": 4,
        "dataset_mode": "paperlike_temporal",
        "objective": "delta",
        "dt": float(dt),
    }
    return data, summary


def build_paperlike_temporal_cache(
    *,
    cache_dir: Path,
    data_seed: int,
    samples: int,
    trajectory_steps: int,
    dt: float,
    train_fraction: float,
    chunk_trajectories: int = 128,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from examples.double_pendulum.data import get_trajectory_analytic

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    num_trajectories = int(samples)
    trajectory_steps = int(trajectory_steps)
    if num_trajectories < 2:
        raise ValueError("Need at least 2 trajectories")
    if trajectory_steps < 1:
        raise ValueError("Need at least 1 transition per trajectory")

    split_traj = int(num_trajectories * float(train_fraction))
    split_traj = max(1, min(split_traj, num_trajectories - 1))
    test_traj = num_trajectories - split_traj
    train_samples = split_traj * trajectory_steps
    test_samples = test_traj * trajectory_steps

    initial_states = sample_initial_states(seed=int(data_seed), samples=num_trajectories).astype(np.float32, copy=False)
    times = (np.arange(trajectory_steps + 1, dtype=np.float64) * float(dt)).astype(np.float64, copy=False)

    train_x = open_memmap(cache_dir / "x.npy", mode="w+", dtype=np.float32, shape=(train_samples, 4))
    train_dx = open_memmap(cache_dir / "dx.npy", mode="w+", dtype=np.float32, shape=(train_samples, 4))
    test_x = open_memmap(cache_dir / "test_x.npy", mode="w+", dtype=np.float32, shape=(test_samples, 4))
    test_dx = open_memmap(cache_dir / "test_dx.npy", mode="w+", dtype=np.float32, shape=(test_samples, 4))
    np.save(cache_dir / "times.npy", times.astype(np.float32, copy=False))

    vget_chunk = jax.jit(jax.vmap(get_trajectory_analytic, (0, None), 0), backend="cpu")

    train_offset = 0
    test_offset = 0
    for start in range(0, num_trajectories, int(chunk_trajectories)):
        end = min(num_trajectories, start + int(chunk_trajectories))
        chunk_init = initial_states[start:end]
        chunk_traj = np.asarray(vget_chunk(jnp.asarray(chunk_init), jnp.asarray(times)), dtype=np.float32)

        local_train = max(0, min(end, split_traj) - start)
        if local_train > 0:
            train_traj = chunk_traj[:local_train]
            flat_x = train_traj[:, :-1, :].reshape(-1, 4)
            flat_dx = (train_traj[:, 1:, :] - train_traj[:, :-1, :]).reshape(-1, 4)
            next_offset = train_offset + flat_x.shape[0]
            train_x[train_offset:next_offset] = flat_x
            train_dx[train_offset:next_offset] = flat_dx
            train_offset = next_offset

        local_test_start = max(0, split_traj - start)
        if local_test_start < (end - start):
            test_traj_chunk = chunk_traj[local_test_start:]
            flat_x = test_traj_chunk[:, :-1, :].reshape(-1, 4)
            flat_dx = (test_traj_chunk[:, 1:, :] - test_traj_chunk[:, :-1, :]).reshape(-1, 4)
            next_offset = test_offset + flat_x.shape[0]
            test_x[test_offset:next_offset] = flat_x
            test_dx[test_offset:next_offset] = flat_dx
            test_offset = next_offset

    train_x.flush()
    train_dx.flush()
    test_x.flush()
    test_dx.flush()

    summary = {
        "data_seed": int(data_seed),
        "num_total_trajectories": int(num_trajectories),
        "num_train_trajectories": int(split_traj),
        "num_test_trajectories": int(test_traj),
        "num_total_samples": int(num_trajectories * trajectory_steps),
        "num_train_samples": int(train_samples),
        "num_test_samples": int(test_samples),
        "transitions_per_trajectory": int(trajectory_steps),
        "state_dim": 4,
        "dataset_mode": "paperlike_temporal",
        "objective": "delta",
        "dt": float(dt),
        "cache_format": "npy_memmap_dir",
    }
    (cache_dir / "meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_paperlike_temporal_cache(cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    cache_dir = Path(cache_dir)
    summary = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    data = {
        "x": np.load(cache_dir / "x.npy", mmap_mode="r"),
        "dx": np.load(cache_dir / "dx.npy", mmap_mode="r"),
        "test_x": np.load(cache_dir / "test_x.npy", mmap_mode="r"),
        "test_dx": np.load(cache_dir / "test_dx.npy", mmap_mode="r"),
    }
    times_path = cache_dir / "times.npy"
    if times_path.exists():
        time_values = np.load(times_path, mmap_mode="r")
        data["time_values"] = time_values
    return data, summary
