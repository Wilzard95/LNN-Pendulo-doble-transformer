from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "double_pendulum"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    requested = device_arg.strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        raise RuntimeError(
            "CUDA was requested but is not available. "
            f"Installed torch={torch_version}, torch.version.cuda={torch_cuda}. "
            "In this venv, install a CUDA build of PyTorch, for example: "
            "`python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`."
        )
    return torch.device(device_arg)


def ensure_output_dirs(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    checkpoints = out_dir / "checkpoints"
    results = out_dir / "results"
    plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    return {
        "out_dir": out_dir,
        "checkpoints": checkpoints,
        "results": results,
        "plots": plots,
    }


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(data), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(model_path: str | None, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if model_path:
        return Path(model_path)
    best_path = out_dir / "checkpoints" / "model_best.pth"
    final_path = out_dir / "model_final.pth"
    if best_path.exists():
        return best_path
    return final_path


def wrap_q_np(q: np.ndarray) -> np.ndarray:
    q_arr = np.asarray(q)
    two_pi = 2.0 * np.pi
    return (q_arr + np.pi) % two_pi - np.pi


def wrap_q_torch(q: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    return torch.remainder(q + torch.pi, two_pi) - torch.pi


def wrap_coords_np(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state).copy()
    if arr.shape[-1] < 2:
        raise ValueError(f"Expected last dimension >= 2 for state, got {arr.shape}")
    arr[..., :2] = wrap_q_np(arr[..., :2])
    return arr


def wrap_coords_torch(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] < 2:
        raise ValueError(f"Expected last dimension >= 2 for state, got {tuple(state.shape)}")
    wrapped = state.clone()
    wrapped[..., :2] = wrap_q_torch(wrapped[..., :2])
    return wrapped


def load_normalization_config(path: Path) -> dict[str, Any]:
    """Load normalization config (z-score) from JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def _make_array(v: Any, shape: int) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full((shape,), float(arr), dtype=np.float64)
    return arr.astype(np.float64)


def normalize_state_xdot(
    state: np.ndarray,
    xdot: np.ndarray,
    norm_cfg: dict[str, Any],
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize state and xdot using a z-score normalization config."""
    # state: [N, 4] = [q1, q2, qdot1, qdot2]
    # xdot: [N, 4] = [qdot1, qdot2, qddot1, qddot2]
    q_mean = _make_array(norm_cfg.get("q_mean", 0.0), 2)
    q_std = _make_array(norm_cfg.get("q_std", 1.0), 2)
    qdot_mean = _make_array(norm_cfg.get("qdot_mean", 0.0), 2)
    qdot_std = _make_array(norm_cfg.get("qdot_std", 1.0), 2)
    qddot_mean = _make_array(norm_cfg.get("qddot_mean", 0.0), 2)
    qddot_std = _make_array(norm_cfg.get("qddot_std", 1.0), 2)

    q = state[:, :2]
    qdot = state[:, 2:]
    qddot = xdot[:, 2:]

    q_norm = (q - q_mean) / (q_std + eps)
    qdot_norm = (qdot - qdot_mean) / (qdot_std + eps)
    qddot_norm = (qddot - qddot_mean) / (qddot_std + eps)

    state_norm = np.concatenate([q_norm, qdot_norm], axis=1)
    xdot_norm = np.concatenate([qdot_norm, qddot_norm], axis=1)
    return state_norm.astype(np.float32, copy=False), xdot_norm.astype(np.float32, copy=False)


def denormalize_qddot(qddot_norm: np.ndarray, norm_cfg: dict[str, Any], eps: float = 1e-8) -> np.ndarray:
    """Convert normalized qddot back to physical units."""
    qddot_mean = _make_array(norm_cfg.get("qddot_mean", 0.0), 2)
    qddot_std = _make_array(norm_cfg.get("qddot_std", 1.0), 2)
    return (qddot_norm * (qddot_std + eps)) + qddot_mean


def denormalize_state_xdot(
    state_norm: np.ndarray, xdot_norm: np.ndarray, norm_cfg: dict[str, Any], eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized state/xdot back to physical coordinates."""
    q_mean = _make_array(norm_cfg.get("q_mean", 0.0), 2)
    q_std = _make_array(norm_cfg.get("q_std", 1.0), 2)
    qdot_mean = _make_array(norm_cfg.get("qdot_mean", 0.0), 2)
    qdot_std = _make_array(norm_cfg.get("qdot_std", 1.0), 2)
    qddot_mean = _make_array(norm_cfg.get("qddot_mean", 0.0), 2)
    qddot_std = _make_array(norm_cfg.get("qddot_std", 1.0), 2)

    q_norm = state_norm[:, :2]
    qdot_norm = state_norm[:, 2:]
    qddot_norm = xdot_norm[:, 2:]

    q = q_norm * (q_std + eps) + q_mean
    qdot = qdot_norm * (qdot_std + eps) + qdot_mean
    qddot = qddot_norm * (qddot_std + eps) + qddot_mean

    state = np.concatenate([q, qdot], axis=1)
    xdot = np.concatenate([qdot, qddot], axis=1)
    return state, xdot


def normalize_state(state: np.ndarray, norm_cfg: dict[str, Any], eps: float = 1e-8) -> np.ndarray:
    """Normalize state (q, qdot) using z-score parameters."""
    q_mean = _make_array(norm_cfg.get("q_mean", 0.0), 2)
    q_std = _make_array(norm_cfg.get("q_std", 1.0), 2)
    qdot_mean = _make_array(norm_cfg.get("qdot_mean", 0.0), 2)
    qdot_std = _make_array(norm_cfg.get("qdot_std", 1.0), 2)

    q = state[:, :2]
    qdot = state[:, 2:]
    q_norm = (q - q_mean) / (q_std + eps)
    qdot_norm = (qdot - qdot_mean) / (qdot_std + eps)
    return np.concatenate([q_norm, qdot_norm], axis=1)


def denormalize_xdot(xdot_norm: np.ndarray, norm_cfg: dict[str, Any], eps: float = 1e-8) -> np.ndarray:
    """Convert normalized xdot (qdot, qddot) back to physical units."""
    qdot_mean = _make_array(norm_cfg.get("qdot_mean", 0.0), 2)
    qdot_std = _make_array(norm_cfg.get("qdot_std", 1.0), 2)
    qddot_mean = _make_array(norm_cfg.get("qddot_mean", 0.0), 2)
    qddot_std = _make_array(norm_cfg.get("qddot_std", 1.0), 2)

    qdot_norm = xdot_norm[:, :2]
    qddot_norm = xdot_norm[:, 2:]

    qdot = qdot_norm * (qdot_std + eps) + qdot_mean
    qddot = qddot_norm * (qddot_std + eps) + qddot_mean
    return np.concatenate([qdot, qddot], axis=1)

