from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import load_normalization_config, normalize_state_xdot, wrap_coords_np

REQUIRED_COLUMNS = [
    "time_s",
    "theta1_rad",
    "omega1_rad_s",
    "theta2_rad",
    "omega2_rad_s",
]


@dataclass
class LagrangianSampleDataset(Dataset):
    state: torch.Tensor
    xdot: torch.Tensor
    energy: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.energy is None:
            return self.state[idx], self.xdot[idx], None
        return self.state[idx], self.xdot[idx], self.energy[idx]


def discover_simulation_files(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("sim_data_*.txt"))
    if not files:
        raise FileNotFoundError(f"No sim_data_*.txt files found in: {data_dir}")
    return files


def read_simulation_file(file_path: Path, dtype: np.dtype = np.float32) -> dict[str, np.ndarray]:
    file_path = Path(file_path)
    read_errors: list[str] = []
    df = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(file_path, sep="\t", comment="#", encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            read_errors.append(f"{encoding}: {exc}")
            continue
    if df is None:
        raise RuntimeError(
            f"Could not decode {file_path.name} with fallback encodings. Details: {' | '.join(read_errors)}"
        )

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path.name}: {missing}")

    time = df["time_s"].to_numpy(dtype=dtype)
    theta1 = df["theta1_rad"].to_numpy(dtype=dtype)
    omega1 = df["omega1_rad_s"].to_numpy(dtype=dtype)
    theta2 = df["theta2_rad"].to_numpy(dtype=dtype)
    omega2 = df["omega2_rad_s"].to_numpy(dtype=dtype)
    energy = None
    if "E_total_J" in df.columns:
        energy = df["E_total_J"].to_numpy(dtype=dtype)

    q = np.stack([theta1, theta2], axis=1).astype(dtype, copy=False)
    qdot = np.stack([omega1, omega2], axis=1).astype(dtype, copy=False)
    dt = np.diff(time).astype(dtype, copy=False)

    if len(time) < 3:
        raise ValueError(f"{file_path.name} has <3 rows; cannot build central differences.")

    return {
        "time": time,
        "q": q,
        "qdot": qdot,
        "dt": dt,
        "energy": energy,
    }


def build_samples_from_trajectory(raw: dict[str, np.ndarray], dtype: np.dtype = np.float32) -> dict[str, np.ndarray]:
    time = raw["time"]
    q = raw["q"]
    qdot = raw["qdot"]
    energy = raw.get("energy")

    # Central difference on omega for internal samples.
    denom = (time[2:] - time[:-2]).astype(dtype, copy=False)
    qddot = (qdot[2:] - qdot[:-2]) / denom[:, None]

    q_mid = q[1:-1]
    qdot_mid = qdot[1:-1]
    time_mid = time[1:-1]
    energy_mid = energy[1:-1] if energy is not None else None

    state = np.concatenate([q_mid, qdot_mid], axis=1).astype(dtype, copy=False)
    state = wrap_coords_np(state).astype(dtype, copy=False)
    xdot = np.concatenate([qdot_mid, qddot], axis=1).astype(dtype, copy=False)

    valid = denom != 0
    valid &= np.isfinite(time_mid)
    valid &= np.isfinite(state).all(axis=1)
    valid &= np.isfinite(xdot).all(axis=1)
    if energy_mid is not None:
        valid &= np.isfinite(energy_mid)

    state = state[valid]
    xdot = xdot[valid]
    time_mid = time_mid[valid]
    if energy_mid is not None:
        energy_mid = energy_mid[valid]

    if len(state) == 0:
        raise ValueError("No valid central-difference samples after filtering.")

    return {
        "time": time_mid.astype(dtype, copy=False),
        "state": state.astype(dtype, copy=False),
        "xdot": xdot.astype(dtype, copy=False),
        "energy": energy_mid.astype(dtype, copy=False) if energy_mid is not None else None,
    }


def load_file_as_samples(file_path: Path, dtype: np.dtype = np.float32) -> dict[str, np.ndarray]:
    raw = read_simulation_file(file_path, dtype=dtype)
    return build_samples_from_trajectory(raw, dtype=dtype)


def load_full_trajectory(file_path: Path, dtype: np.dtype = np.float32) -> dict[str, np.ndarray | float | None]:
    raw = read_simulation_file(file_path, dtype=dtype)
    dt = raw["dt"]
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(finite_dt) == 0:
        raise ValueError(f"No positive finite dt in {file_path}")
    return {
        "time": raw["time"],
        "q": raw["q"],
        "qdot": raw["qdot"],
        "energy": raw.get("energy"),
        "dt_median": float(np.median(finite_dt)),
    }


def _compute_split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError(f"Need at least 3 elements to split train/val/test, got {n}")
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0,1)")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    if val_ratio > 0 and n_val == 0:
        n_val = 1
    if test_ratio > 0 and n_test == 0:
        n_test = 1

    while n_val + n_test >= n:
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split produced empty train set.")
    return n_train, n_val, n_test


def _split_files_random(
    files: list[Path],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    n_train, n_val, n_test = _compute_split_counts(len(files), val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    shuffled = [files[i] for i in perm]
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def _files_from_names(data_dir: Path, names: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for name in names:
        p = Path(name)
        if not p.is_absolute():
            p = Path(data_dir) / p
        resolved.append(p)
    return resolved


def _load_many_files(files: list[Path], dtype: np.dtype) -> dict[str, Any]:
    state_parts: list[np.ndarray] = []
    xdot_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    file_parts: list[np.ndarray] = []
    energy_parts: list[np.ndarray] = []

    loaded_files = 0
    for file_path in files:
        try:
            samples = load_file_as_samples(file_path, dtype=dtype)
        except Exception as exc:
            warnings.warn(f"Skipping {file_path.name}: {exc}")
            continue
        n = samples["state"].shape[0]
        if n == 0:
            continue
        state_parts.append(samples["state"])
        xdot_parts.append(samples["xdot"])
        time_parts.append(samples["time"])
        file_parts.append(np.array([file_path.name] * n, dtype=object))
        if samples.get("energy") is not None:
            energy_parts.append(samples["energy"])
        else:
            energy_parts.append(np.full((n,), np.nan, dtype=dtype))
        loaded_files += 1

    if loaded_files == 0:
        raise RuntimeError("No valid files were loaded.")

    state = np.concatenate(state_parts, axis=0)
    xdot = np.concatenate(xdot_parts, axis=0)
    time = np.concatenate(time_parts, axis=0)
    file_name = np.concatenate(file_parts, axis=0)
    energy = np.concatenate(energy_parts, axis=0)
    return {
        "state": state,
        "xdot": xdot,
        "time": time,
        "file_name": file_name,
        "energy": energy,
        "loaded_files": loaded_files,
        "num_samples": int(state.shape[0]),
    }


def _compute_normalization_config(
    state: np.ndarray, xdot: np.ndarray, eps: float = 1e-8
) -> dict[str, Any]:
    """Compute z-score normalization parameters from training samples."""
    q = state[:, :2]
    qdot = state[:, 2:]
    qddot = xdot[:, 2:]

    return {
        "schema": "full_zscore_v1",
        "eps": float(eps),
        "q_mean": q.mean(axis=0).tolist(),
        "q_std": q.std(axis=0).tolist(),
        "qdot_mean": qdot.mean(axis=0).tolist(),
        "qdot_std": qdot.std(axis=0).tolist(),
        "qddot_mean": qddot.mean(axis=0).tolist(),
        "qddot_std": qddot.std(axis=0).tolist(),
    }


def _save_normalization_config(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        import json

        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _apply_normalization_to_split(
    split: dict[str, np.ndarray], norm_cfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    state, xdot = normalize_state_xdot(split["state"], split["xdot"], norm_cfg)
    out = dict(split)
    out["state"] = state
    out["xdot"] = xdot
    return out


def _to_torch_dataset(split: dict[str, np.ndarray], torch_dtype: torch.dtype) -> LagrangianSampleDataset:
    state = torch.from_numpy(split["state"]).to(dtype=torch_dtype)
    xdot = torch.from_numpy(split["xdot"]).to(dtype=torch_dtype)
    energy = None
    if "energy" in split and split["energy"] is not None:
        energy = torch.from_numpy(split["energy"]).to(dtype=torch_dtype)
    return LagrangianSampleDataset(state=state, xdot=xdot, energy=energy)


def prepare_datasets(
    data_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    split_by_trajectory: bool = False,
    np_dtype: np.dtype = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    split_override: dict[str, Any] | None = None,
    normalize: bool = False,
    normalization_file: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    all_files = discover_simulation_files(data_dir)

    if split_by_trajectory:
        if split_override and {"train_files", "val_files", "test_files"}.issubset(split_override.keys()):
            train_files = _files_from_names(data_dir, list(split_override["train_files"]))
            val_files = _files_from_names(data_dir, list(split_override["val_files"]))
            test_files = _files_from_names(data_dir, list(split_override["test_files"]))
            file_split = {"train": train_files, "val": val_files, "test": test_files}
        else:
            file_split = _split_files_random(all_files, val_ratio, test_ratio, seed)

        train_np = _load_many_files(file_split["train"], dtype=np_dtype)
        val_np = _load_many_files(file_split["val"], dtype=np_dtype)
        test_np = _load_many_files(file_split["test"], dtype=np_dtype)

        split_info = {
            "split_by_trajectory": True,
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "train_files": [p.name for p in file_split["train"]],
            "val_files": [p.name for p in file_split["val"]],
            "test_files": [p.name for p in file_split["test"]],
        }
    else:
        full = _load_many_files(all_files, dtype=np_dtype)
        n_total = full["num_samples"]
        n_train, n_val, n_test = _compute_split_counts(n_total, val_ratio, test_ratio)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_total)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val : n_train + n_val + n_test]

        def pick(idx: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "state": full["state"][idx],
                "xdot": full["xdot"][idx],
                "time": full["time"][idx],
                "file_name": full["file_name"][idx],
                "energy": full["energy"][idx],
                "num_samples": int(len(idx)),
                "loaded_files": len(np.unique(full["file_name"][idx])),
            }

        train_np = pick(train_idx)
        val_np = pick(val_idx)
        test_np = pick(test_idx)

        split_info = {
            "split_by_trajectory": False,
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "train_files": sorted(np.unique(train_np["file_name"]).tolist()),
            "val_files": sorted(np.unique(val_np["file_name"]).tolist()),
            "test_files": sorted(np.unique(test_np["file_name"]).tolist()),
        }

    normalization_cfg: dict[str, Any] | None = None
    if normalize:
        # Determine normalization config file path and load / compute
        norm_path = None
        if normalization_file is not None:
            norm_path = Path(normalization_file)
        elif out_dir is not None:
            norm_path = Path(out_dir) / "normalization.json"

        if norm_path is not None and norm_path.exists():
            normalization_cfg = load_normalization_config(norm_path)
        else:
            # compute from training samples and persist for reproducibility
            normalization_cfg = _compute_normalization_config(train_np["state"], train_np["xdot"])
            if norm_path is not None:
                _save_normalization_config(norm_path, normalization_cfg)

        train_np = _apply_normalization_to_split(train_np, normalization_cfg)
        val_np = _apply_normalization_to_split(val_np, normalization_cfg)
        test_np = _apply_normalization_to_split(test_np, normalization_cfg)

    train_ds = _to_torch_dataset(train_np, torch_dtype=torch_dtype)
    val_ds = _to_torch_dataset(val_np, torch_dtype=torch_dtype)
    test_ds = _to_torch_dataset(test_np, torch_dtype=torch_dtype)

    summary = {
        "num_files_found": len(all_files),
        "num_samples_train": len(train_ds),
        "num_samples_val": len(val_ds),
        "num_samples_test": len(test_ds),
        "num_samples_total": len(train_ds) + len(val_ds) + len(test_ds),
        "num_files_loaded_train": int(train_np["loaded_files"]),
        "num_files_loaded_val": int(val_np["loaded_files"]),
        "num_files_loaded_test": int(test_np["loaded_files"]),
    }

    return {
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "split_info": split_info,
        "summary": summary,
        "normalization": normalization_cfg,
    }
