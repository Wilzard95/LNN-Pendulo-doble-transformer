from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OFFICIAL_REPO_ROOT = Path(__file__).resolve().parent / "official_lagrangian_nns"
if str(OFFICIAL_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_ROOT))

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from examples.hyperopt import HyperparameterSearch as hyper
from paperlike_double_pendulum import (
    build_paperlike_dataset,
    build_paperlike_temporal_cache,
    load_paperlike_temporal_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official double pendulum experiment path on CPU using the official JAX code.")
    parser.add_argument("--out_dir", type=str, default="experiments/official_jax_cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="repo_transitions",
        choices=["repo_transitions", "paperlike_instantaneous", "paperlike_temporal"],
    )
    parser.add_argument("--objective", type=str, default="auto", choices=["auto", "delta", "xdot"])
    parser.add_argument("--dataset_size", type=float, default=50.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--trajectory_steps", type=int, default=100)
    parser.add_argument("--test_split", type=float, default=0.9)
    parser.add_argument("--num_epochs", type=int, default=40000)
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--act", type=str, default="soft_relu")
    parser.add_argument("--hidden_dim", type=int, default=596)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--n_updates", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005516656601005163)
    parser.add_argument("--lr2", type=float, default=1.897157209816416e-05)
    parser.add_argument("--dt", type=float, default=0.09609870774790222)
    parser.add_argument("--batch_size", type=int, default=27)
    parser.add_argument("--l2reg", type=float, default=0.24927677946969878)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--param_check_every", type=int, default=100)
    parser.add_argument("--latest_checkpoint_every", type=int, default=5000)
    parser.add_argument("--eval_batch_size", type=int, default=8192)
    parser.add_argument("--eval_train_samples", type=int, default=100000)
    parser.add_argument("--eval_test_samples", type=int, default=100000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--warmup_eval_every", type=int, default=100)
    parser.add_argument("--warmup_eval_until", type=int, default=1000)
    parser.add_argument(
        "--final_eval_only",
        type=lambda s: str(s).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Skip all intermediate evaluations and run a single full evaluation at the end.",
    )
    parser.add_argument(
        "--eval_on_small_loss_improve",
        type=lambda s: str(s).lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--seed_stride", type=int, default=1)
    parser.add_argument(
        "--sweep_all_attempts",
        type=lambda s: str(s).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Run all seeds up to max_attempts instead of stopping after the first completed attempt.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a much smaller CPU sanity-check run.")
    return parser.parse_args()


def official_commit_hash(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head.split(" ", 1)[1]
        ref_path = repo_root / ".git" / Path(ref)
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return head


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    paths = {
        "out_dir": out_dir,
        "checkpoints": out_dir / "checkpoints",
        "plots": out_dir / "plots",
        "results": out_dir / "results",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_official_checkpoint(path: Path, params: object, train_args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params_np = tree_map(lambda x: np.asarray(x), params)
    with path.open("wb") as fh:
        pickle.dump({"params": params_np, "train_args": vars(train_args)}, fh)


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_training_log(log_path: Path) -> list[dict[str, float]]:
    pattern = re.compile(r"iteration=(\d+), train_loss=([0-9eE+\-.]+), test_loss=([0-9eE+\-.]+)")
    rows: list[dict[str, float]] = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        rows.append(
            {
                "iteration": float(match.group(1)),
                "train_loss": float(match.group(2)),
                "test_loss": float(match.group(3)),
            }
        )
    return rows


def save_loss_artifacts(rows: list[dict[str, float]], results_dir: Path, plots_dir: Path) -> None:
    csv_path = results_dir / "loss_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["iteration", "train_loss", "test_loss"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if not rows:
        return

    iterations = np.array([row["iteration"] for row in rows], dtype=np.float64)
    train_loss = np.array([row["train_loss"] for row in rows], dtype=np.float64)
    test_loss = np.array([row["test_loss"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(iterations, train_loss, label="train")
    ax.plot(iterations, test_loss, label="test")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("Official JAX double pendulum training")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def build_train_args(args: argparse.Namespace) -> hyper.ObjectView:
    objective = resolve_objective(args)
    return hyper.ObjectView(
        dict(
            num_epochs=int(args.num_epochs),
            loss=str(args.loss),
            l2reg=float(args.l2reg),
            act=str(args.act),
            hidden_dim=int(args.hidden_dim),
            output_dim=int(args.output_dim),
            dt=float(args.dt),
            layers=int(args.layers),
            lr=float(args.lr),
            lr2=float(args.lr2),
            model="gln",
            n_updates=int(args.n_updates),
            batch_size=int(args.batch_size),
            grad_clip=float(args.grad_clip),
            lr_warmup_steps=int(args.lr_warmup_steps),
            param_check_every=int(args.param_check_every),
            latest_checkpoint_every=int(args.latest_checkpoint_every),
            objective=str(objective),
            eval_batch_size=int(args.eval_batch_size),
            eval_train_samples=int(args.eval_train_samples),
            eval_test_samples=int(args.eval_test_samples),
            eval_every=int(args.eval_every),
            warmup_eval_every=int(args.warmup_eval_every),
            warmup_eval_until=int(args.warmup_eval_until),
            final_eval_only=bool(args.final_eval_only),
            eval_on_small_loss_improve=bool(args.eval_on_small_loss_improve),
        )
    )


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def dataset_cache_path(args: argparse.Namespace, data_seed: int) -> Path:
    cache_root = Path(__file__).resolve().parent / "dataset_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_mode": str(args.dataset_mode),
        "data_seed": int(data_seed),
        "dataset_size": float(args.dataset_size),
        "fps": int(args.fps),
        "samples": int(args.samples),
        "trajectory_steps": int(args.trajectory_steps),
        "test_split": float(args.test_split),
        "dt": float(args.dt),
    }
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if str(args.dataset_mode) == "paperlike_instantaneous":
        prefix = "paperlike_jax"
    elif str(args.dataset_mode) == "paperlike_temporal":
        prefix = "paperlike_traj_jax"
    else:
        prefix = "official_jax"
    if str(args.dataset_mode) == "paperlike_temporal":
        return cache_root / f"{prefix}_{key}"
    return cache_root / f"{prefix}_{key}.npz"


def resolve_objective(args: argparse.Namespace) -> str:
    if str(args.objective) != "auto":
        return str(args.objective)
    if str(args.dataset_mode) == "paperlike_instantaneous":
        return "xdot"
    if str(args.dataset_mode) == "paperlike_temporal":
        return "delta"
    return "delta"


def build_dataset(args: argparse.Namespace, data_seed: int) -> tuple[dict[str, object], dict[str, object]]:
    cache_path = dataset_cache_path(args, data_seed)
    if cache_path.exists():
        if str(args.dataset_mode) == "paperlike_temporal" and cache_path.is_dir():
            data, summary = load_paperlike_temporal_cache(cache_path)
            summary = dict(summary)
            summary["cache_path"] = str(cache_path)
            summary["cache_hit"] = True
            summary["dataset_mode"] = str(args.dataset_mode)
            summary["objective"] = resolve_objective(args)
            return data, summary
        cached = np.load(cache_path)
        data = {"x": jnp.asarray(cached["x"]), "t": jnp.asarray(cached["t"]), "test_x": jnp.asarray(cached["test_x"]), "test_t": jnp.asarray(cached["test_t"])}
        if "dx" in cached:
            data["dx"] = jnp.asarray(cached["dx"])
            data["test_dx"] = jnp.asarray(cached["test_dx"])
        if "xdot" in cached:
            data["xdot"] = jnp.asarray(cached["xdot"])
            data["test_xdot"] = jnp.asarray(cached["test_xdot"])
        summary = {
            "data_seed": int(data_seed),
            "num_train_samples": int(data["x"].shape[0]),
            "num_test_samples": int(data["test_x"].shape[0]),
            "state_dim": int(data["x"].shape[1]),
            "cache_path": str(cache_path),
            "cache_hit": True,
            "dataset_mode": str(args.dataset_mode),
            "objective": resolve_objective(args),
        }
        return data, summary

    if str(args.dataset_mode) == "paperlike_instantaneous":
        data_np, summary = build_paperlike_dataset(
            data_seed=int(data_seed),
            samples=int(args.samples),
            train_fraction=float(args.test_split),
        )
        np.savez_compressed(
            cache_path,
            x=data_np["x"],
            xdot=data_np["xdot"],
            t=data_np["t"],
            test_x=data_np["test_x"],
            test_xdot=data_np["test_xdot"],
            test_t=data_np["test_t"],
        )
        summary["cache_path"] = str(cache_path)
        summary["cache_hit"] = False
        data = {k: jnp.asarray(v) for k, v in data_np.items()}
        return data, summary

    if str(args.dataset_mode) == "paperlike_temporal":
        summary = build_paperlike_temporal_cache(
            cache_dir=cache_path,
            data_seed=int(data_seed),
            samples=int(args.samples),
            trajectory_steps=int(args.trajectory_steps),
            dt=float(args.dt),
            train_fraction=float(args.test_split),
        )
        data, _ = load_paperlike_temporal_cache(cache_path)
        summary["cache_path"] = str(cache_path)
        summary["cache_hit"] = False
        return data, summary

    rng = jax.random.PRNGKey(int(data_seed)) + 500
    data = hyper.new_get_dataset(
        rng + 2,
        t_span=[0, float(args.dataset_size)],
        fps=int(args.fps),
        samples=int(args.samples),
        test_split=float(args.test_split),
    )
    np.savez_compressed(
        cache_path,
        x=np.asarray(data["x"]),
        dx=np.asarray(data["dx"]),
        t=np.asarray(data["t"]),
        test_x=np.asarray(data["test_x"]),
        test_dx=np.asarray(data["test_dx"]),
        test_t=np.asarray(data["test_t"]),
    )
    summary = {
        "data_seed": int(data_seed),
        "num_train_samples": int(data["x"].shape[0]),
        "num_test_samples": int(data["test_x"].shape[0]),
        "state_dim": int(data["x"].shape[1]),
        "cache_path": str(cache_path),
        "cache_hit": False,
        "dataset_mode": str(args.dataset_mode),
        "objective": resolve_objective(args),
    }
    return data, summary


def run_attempt(
    base_out_dir: Path,
    args: argparse.Namespace,
    run_config: dict[str, object],
    data: dict[str, object],
    dataset_summary: dict[str, object],
    attempt_idx: int,
    attempt_seed: int,
) -> dict[str, object]:
    attempt_name = f"attempt_{attempt_idx:02d}_seed_{attempt_seed:04d}"
    attempt_dir = base_out_dir / "attempts" / attempt_name
    paths = ensure_dirs(attempt_dir)

    attempt_run_config = dict(run_config)
    attempt_run_config["attempt"] = {
        "index": int(attempt_idx),
        "seed": int(attempt_seed),
        "name": attempt_name,
    }
    attempt_run_config["dataset"] = dataset_summary
    (attempt_dir / "run_config.json").write_text(json.dumps(attempt_run_config, indent=2), encoding="utf-8")

    train_args = build_train_args(args)
    rng = jax.random.PRNGKey(int(attempt_seed)) + 500

    init_random_params, nn_forward_fn = hyper.extended_mlp(train_args)
    _, init_params = init_random_params(rng + 1, (-1, 4))
    model = (nn_forward_fn, init_params)

    log_path = paths["results"] / "training.log"
    exception_text: str | None = None
    train_losses: list[float] = []
    test_losses: list[float] = []
    best_loss = float("inf")
    params = init_params

    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_fh, contextlib.redirect_stdout(Tee(sys.stdout, log_fh)):
        print(f"[attempt] index={attempt_idx} seed={attempt_seed}")
        try:
            def persist_best_checkpoint(*, iteration: int, params: object, test_loss: float) -> None:
                save_official_checkpoint(paths["checkpoints"] / "model_best.pkl", params, train_args)
                save_official_checkpoint(base_out_dir / "checkpoints" / "model_best.pkl", params, train_args)
                print(f"[BEST] iteration={iteration} | test_loss={float(test_loss):.8e}")

            def persist_latest_checkpoint(*, iteration: int, params: object) -> None:
                save_official_checkpoint(paths["checkpoints"] / "model_latest.pkl", params, train_args)
                save_official_checkpoint(base_out_dir / "checkpoints" / "model_latest.pkl", params, train_args)
                print(f"[LATEST] iteration={int(iteration)}")

            params, train_losses, test_losses, best_loss = hyper.train(
                train_args,
                model,
                data,
                rng + 3,
                checkpoint_callback=persist_best_checkpoint,
                latest_checkpoint_callback=persist_latest_checkpoint,
            )
        except Exception:
            exception_text = traceback.format_exc()
            print(exception_text, flush=True)
    elapsed_s = time.perf_counter() - start_time

    train_meta = dict(getattr(hyper, "train_meta", {}))
    if exception_text is not None:
        train_meta = {
            "completed": False,
            "stop_reason": "exception",
            "stop_iteration": -1,
            "best_iteration": -1,
            "best_loss": float("inf"),
        }
    elif not train_meta:
        train_meta = {
            "completed": False,
            "stop_reason": "unknown",
            "stop_iteration": -1,
            "best_iteration": -1,
            "best_loss": float(best_loss),
        }

    latest_checkpoint_path = paths["checkpoints"] / "model_latest.pkl"
    if getattr(hyper, "best_params", None) is not None:
        best_params = hyper.best_params
    elif latest_checkpoint_path.exists():
        best_params = pickle.load(latest_checkpoint_path.open("rb"))["params"]
    else:
        best_params = params
    best_params_np = tree_map(lambda x: np.asarray(x), best_params)
    final_params_np = tree_map(lambda x: np.asarray(x), params)

    save_official_checkpoint(paths["checkpoints"] / "model_best.pkl", best_params_np, train_args)
    save_official_checkpoint(base_out_dir / "checkpoints" / "model_best.pkl", best_params_np, train_args)
    with (attempt_dir / "model_final.pkl").open("wb") as fh:
        pickle.dump({"params": final_params_np, "train_args": vars(train_args)}, fh)

    loss_rows = parse_training_log(log_path)
    save_loss_artifacts(loss_rows, paths["results"], paths["plots"])

    summary = {
        "attempt_index": int(attempt_idx),
        "attempt_name": attempt_name,
        "seed": int(attempt_seed),
        "attempt_dir": str(attempt_dir),
        "completed": bool(train_meta.get("completed", False)),
        "stop_reason": str(train_meta.get("stop_reason", "unknown")),
        "stop_iteration": int(train_meta.get("stop_iteration", -1)),
        "best_iteration": int(train_meta.get("best_iteration", -1)),
        "best_loss": float(train_meta.get("best_loss", best_loss)),
        "elapsed_s": float(elapsed_s),
        "num_train_samples": int(dataset_summary["num_train_samples"]),
        "num_test_samples": int(dataset_summary["num_test_samples"]),
        "logged_points": int(len(loss_rows)),
        "train_curve_points": int(len(train_losses)),
        "test_curve_points": int(len(test_losses)),
        "exception": exception_text,
        "files": {
            "model_best": str(paths["checkpoints"] / "model_best.pkl"),
            "model_latest": str(paths["checkpoints"] / "model_latest.pkl"),
            "model_final": str(attempt_dir / "model_final.pkl"),
            "training_log": str(log_path),
            "loss_csv": str(paths["results"] / "loss_history.csv"),
            "loss_plot": str(paths["plots"] / "loss_curve.png"),
        },
    }
    (paths["results"] / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def promote_best_attempt(base_paths: dict[str, Path], best_attempt: dict[str, object]) -> None:
    attempt_dir = Path(str(best_attempt["attempt_dir"]))
    copy_if_exists(attempt_dir / "checkpoints" / "model_best.pkl", base_paths["checkpoints"] / "model_best.pkl")
    copy_if_exists(attempt_dir / "model_final.pkl", base_paths["out_dir"] / "model_final.pkl")
    copy_if_exists(attempt_dir / "results" / "training.log", base_paths["results"] / "training.log")
    copy_if_exists(attempt_dir / "results" / "loss_history.csv", base_paths["results"] / "loss_history.csv")
    copy_if_exists(attempt_dir / "plots" / "loss_curve.png", base_paths["plots"] / "loss_curve.png")
    copy_if_exists(attempt_dir / "results" / "train_summary.json", base_paths["results"] / "best_attempt_summary.json")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.dataset_size = 10.0
        args.fps = 10
        args.samples = 8
        args.num_epochs = 20
        args.batch_size = 8
        args.hidden_dim = 128
        args.layers = 2
        args.n_updates = 2
        args.l2reg = 1e-3
        args.max_attempts = min(int(args.max_attempts), 2)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
    paths = ensure_dirs(out_dir)
    (out_dir / "attempts").mkdir(parents=True, exist_ok=True)

    run_config = {
        "mode": "official_jax_cpu_compat",
        "official_repo_root": str(OFFICIAL_REPO_ROOT),
        "official_repo_commit": official_commit_hash(OFFICIAL_REPO_ROOT),
        "jax_version": str(jax.__version__),
        "device": str(jax.devices()[0]),
        "args": vars(args),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"[dataset] preparing with data_seed={int(args.data_seed)}")
    dataset_start = time.perf_counter()
    data, dataset_summary = build_dataset(args, int(args.data_seed))
    dataset_elapsed_s = time.perf_counter() - dataset_start
    dataset_summary["elapsed_s"] = float(dataset_elapsed_s)
    (paths["results"] / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    print(
        f"[dataset] ready | cache_hit={dataset_summary['cache_hit']} | "
        f"train_samples={dataset_summary['num_train_samples']} | "
        f"test_samples={dataset_summary['num_test_samples']} | elapsed_s={dataset_elapsed_s:.1f}"
    )

    attempts: list[dict[str, object]] = []
    best_attempt: dict[str, object] | None = None
    total_start = time.perf_counter()

    for attempt_idx in range(max(1, int(args.max_attempts))):
        attempt_seed = int(args.seed) + attempt_idx * int(args.seed_stride)
        print(f"[attempt {attempt_idx + 1}/{max(1, int(args.max_attempts))}] seed={attempt_seed}")
        attempt_summary = run_attempt(out_dir, args, run_config, data, dataset_summary, attempt_idx, attempt_seed)
        attempts.append(attempt_summary)

        best_loss_value = float(attempt_summary["best_loss"])
        if np.isfinite(best_loss_value) and (
            best_attempt is None or best_loss_value < float(best_attempt["best_loss"])
        ):
            best_attempt = attempt_summary

        print(
            f"[attempt {attempt_idx + 1}] stop_reason={attempt_summary['stop_reason']} | "
            f"stop_iteration={attempt_summary['stop_iteration']} | "
            f"best_iteration={attempt_summary['best_iteration']} | "
            f"best_loss={best_loss_value:.8e}"
        )

        if bool(attempt_summary["completed"]) and not bool(args.sweep_all_attempts):
            print("[attempt] completed full schedule. Stopping retries.")
            break

        if attempt_idx + 1 < int(args.max_attempts):
            if bool(args.sweep_all_attempts):
                print("[attempt] moving to next scheduled seed.")
            else:
                print("[attempt] retrying with next seed.")

    total_elapsed_s = time.perf_counter() - total_start

    attempts_csv = paths["results"] / "attempts_summary.csv"
    with attempts_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "attempt_index",
                "attempt_name",
                "seed",
                "completed",
                "stop_reason",
                "stop_iteration",
                "best_iteration",
                "best_loss",
                "elapsed_s",
            ],
        )
        writer.writeheader()
        for attempt in attempts:
            writer.writerow({key: attempt[key] for key in writer.fieldnames})

    overall_summary = {
        "official_repo_commit": run_config["official_repo_commit"],
        "jax_version": run_config["jax_version"],
        "device": run_config["device"],
        "dataset_summary": dataset_summary,
        "total_elapsed_s": float(total_elapsed_s),
        "num_attempts_run": int(len(attempts)),
        "best_attempt": best_attempt,
        "attempts": attempts,
    }
    (paths["results"] / "attempts_summary.json").write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    if best_attempt is None:
        raise RuntimeError("No successful attempt produced finite metrics.")

    promote_best_attempt(paths, best_attempt)
    (paths["results"] / "train_summary.json").write_text(json.dumps(best_attempt, indent=2), encoding="utf-8")

    print(f"Official repo commit: {run_config['official_repo_commit']}")
    print(f"JAX version: {run_config['jax_version']}")
    print(f"Device: {run_config['device']}")
    print(f"Total elapsed seconds: {total_elapsed_s:.1f}")
    print(f"Attempts run: {len(attempts)}")
    print(f"Best attempt: {best_attempt['attempt_name']}")
    print(f"Best stop reason: {best_attempt['stop_reason']}")
    print(f"Best stop iteration: {best_attempt['stop_iteration']}")
    print(f"Best loss: {float(best_attempt['best_loss']):.8e}")
    print(f"Best params: {paths['checkpoints'] / 'model_best.pkl'}")
    print(f"Final params: {out_dir / 'model_final.pkl'}")
    print(f"Training log: {paths['results'] / 'training.log'}")
    print(f"Loss CSV: {paths['results'] / 'loss_history.csv'}")
    print(f"Loss plot: {paths['plots'] / 'loss_curve.png'}")
    print(f"Attempts JSON: {paths['results'] / 'attempts_summary.json'}")


if __name__ == "__main__":
    main()
