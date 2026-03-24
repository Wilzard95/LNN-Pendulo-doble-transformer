from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lnn.data import discover_simulation_files, load_full_trajectory
from lnn.plotting import plot_sanity_trajectory
from lnn.utils import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, ensure_output_dirs, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check for double pendulum dataset.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--example_file", type=str, default=None, help="Optional sim_data_###.txt name or full path")
    parser.add_argument("--double", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_paths = ensure_output_dirs(Path(args.out_dir))
    plots_dir = out_paths["plots"]
    results_dir = out_paths["results"]
    data_dir = Path(args.data_dir)

    dtype = np.float64 if args.double else np.float32
    files = discover_simulation_files(data_dir)

    theta_min = np.array([np.inf, np.inf], dtype=np.float64)
    theta_max = np.array([-np.inf, -np.inf], dtype=np.float64)
    omega_min = np.array([np.inf, np.inf], dtype=np.float64)
    omega_max = np.array([-np.inf, -np.inf], dtype=np.float64)
    dt_values: list[np.ndarray] = []
    nan_files = 0
    loaded = 0

    for file_path in files:
        try:
            traj = load_full_trajectory(file_path, dtype=dtype)
        except Exception:
            nan_files += 1
            continue
        q = traj["q"]
        qdot = traj["qdot"]
        time = traj["time"]

        if not (np.isfinite(q).all() and np.isfinite(qdot).all() and np.isfinite(time).all()):
            nan_files += 1
            continue

        theta_min = np.minimum(theta_min, np.min(q, axis=0))
        theta_max = np.maximum(theta_max, np.max(q, axis=0))
        omega_min = np.minimum(omega_min, np.min(qdot, axis=0))
        omega_max = np.maximum(omega_max, np.max(qdot, axis=0))
        dt_values.append(np.diff(time))
        loaded += 1

    if loaded == 0:
        raise RuntimeError("No valid trajectories loaded in sanity check.")

    dt_cat = np.concatenate(dt_values)
    dt_cat = dt_cat[np.isfinite(dt_cat)]
    if len(dt_cat) == 0:
        raise RuntimeError("No finite dt values found.")

    if args.example_file:
        example_path = Path(args.example_file)
        if not example_path.is_absolute():
            example_path = data_dir / example_path
    else:
        example_path = files[0]

    example = load_full_trajectory(example_path, dtype=dtype)
    example_fig = plots_dir / f"sanity_example_{example_path.stem}.png"
    plot_sanity_trajectory(example["time"], example["q"], out_path=example_fig)

    summary = {
        "data_dir": str(data_dir),
        "num_files_found": len(files),
        "num_files_loaded": loaded,
        "num_files_failed_or_nonfinite": nan_files,
        "theta1_min_max": [float(theta_min[0]), float(theta_max[0])],
        "theta2_min_max": [float(theta_min[1]), float(theta_max[1])],
        "omega1_min_max": [float(omega_min[0]), float(omega_max[0])],
        "omega2_min_max": [float(omega_min[1]), float(omega_max[1])],
        "dt_min": float(np.min(dt_cat)),
        "dt_median": float(np.median(dt_cat)),
        "dt_max": float(np.max(dt_cat)),
        "example_plot": str(example_fig),
    }

    save_json(results_dir / "sanity_summary.json", summary)

    print(f"Files found: {summary['num_files_found']}")
    print(f"Files loaded: {summary['num_files_loaded']}")
    print(f"Files failed/nonfinite: {summary['num_files_failed_or_nonfinite']}")
    print(f"Theta1 range: [{summary['theta1_min_max'][0]:.6f}, {summary['theta1_min_max'][1]:.6f}]")
    print(f"Theta2 range: [{summary['theta2_min_max'][0]:.6f}, {summary['theta2_min_max'][1]:.6f}]")
    print(f"Omega1 range: [{summary['omega1_min_max'][0]:.6f}, {summary['omega1_min_max'][1]:.6f}]")
    print(f"Omega2 range: [{summary['omega2_min_max'][0]:.6f}, {summary['omega2_min_max'][1]:.6f}]")
    print(
        "dt stats: "
        f"min={summary['dt_min']:.8f}, "
        f"median={summary['dt_median']:.8f}, "
        f"max={summary['dt_max']:.8f}"
    )
    print(f"Example plot: {example_fig}")
    print(f"Summary JSON: {results_dir / 'sanity_summary.json'}")


if __name__ == "__main__":
    main()
