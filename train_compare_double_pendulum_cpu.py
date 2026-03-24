from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from jax.tree_util import tree_flatten, tree_map
from tqdm import tqdm

matplotlib.use("Agg")

LOCAL_ROOT = Path(__file__).resolve().parent
OFFICIAL_REPO_ROOT = LOCAL_ROOT / "official_lagrangian_nns"
if str(OFFICIAL_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO_ROOT))

from lnn.jax_compat import optimizers
from lnn.utils import wrap_coords

from direct_delta_models import build_model
from train_official_double_pendulum_cpu import (
    Tee,
    build_dataset,
    ensure_dirs,
    official_commit_hash,
    parse_training_log,
    resolve_objective,
    save_loss_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct-delta baselines for double pendulum comparison on CPU.")
    parser.add_argument("--out_dir", type=str, default="experiments/compare_baseline_mlp")
    parser.add_argument("--model_kind", type=str, default="baseline_mlp", choices=["baseline_mlp", "baseline_attn"])
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
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--lr", type=float, default=0.005516656601005163)
    parser.add_argument("--lr2", type=float, default=1.897157209816416e-05)
    parser.add_argument("--batch_size", type=int, default=27)
    parser.add_argument("--l2reg", type=float, default=0.24927677946969878)
    parser.add_argument("--dt", type=float, default=0.09609870774790222)
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
    parser.add_argument("--hidden_dim", type=int, default=596)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--activation", type=str, default="soft_relu")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--ff_multiplier", type=int, default=2)
    parser.add_argument("--disable_jit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_model_config(args: argparse.Namespace) -> dict[str, object]:
    if args.model_kind == "baseline_mlp":
        return {
            "input_dim": 4,
            "hidden_dim": int(args.hidden_dim),
            "output_dim": 4,
            "layers": int(args.layers),
            "activation": str(args.activation),
        }

    return {
        "input_dim": 4,
        "d_model": int(args.d_model),
        "output_dim": 4,
        "num_heads": int(args.num_heads),
        "num_blocks": int(args.num_blocks),
        "ff_multiplier": int(args.ff_multiplier),
        "activation": "relu",
    }


def train_direct_model(
    args: argparse.Namespace,
    model_kind: str,
    model_config: dict[str, object],
    data: dict[str, object],
    rng: jnp.ndarray,
    checkpoint_callback=None,
) -> tuple[dict[str, object], dict[str, object], list[float], list[float]]:
    init_fn, apply_fn = build_model(model_kind, model_config)
    params = init_fn(rng + 1)
    jit_enabled = not bool(args.disable_jit)
    np_rng = np.random.default_rng(int(np.asarray(rng).sum()))

    def predict_impl(params_local, state_batch):
        return apply_fn(params_local, state_batch)
    predict = jax.jit(predict_impl) if jit_enabled else predict_impl

    def loss_value(params_local, state_batch, targets, l2reg):
        preds = predict(params_local, state_batch)
        if args.loss == "l1":
            base = jnp.sum(jnp.abs(preds - targets))
        else:
            base = jnp.sum(jnp.square(preds - targets))
        leaves, _ = tree_flatten(params_local)
        l2_norm = sum(jnp.vdot(param, param) for param in leaves)
        return base + l2reg * l2_norm / float(args.batch_size)

    if jit_enabled:
        loss_value = jax.jit(loss_value)

    opt_init, opt_update, get_params = optimizers.adam(
        lambda t: jnp.select(
            [t < args.num_epochs // 2, t >= args.num_epochs // 2],
            [args.lr, args.lr2],
        )
    )
    opt_state = opt_init(params)

    def update_step_impl(i, opt_state_local, state_batch, target_batch):
        params_local = get_params(opt_state_local)
        grads = jax.grad(loss_value)(params_local, state_batch, target_batch, args.l2reg)
        next_state = opt_update(i, grads, opt_state_local)
        return next_state, params_local
    update_step = jax.jit(update_step_impl) if jit_enabled else update_step_impl

    best_params = params
    best_loss = np.inf
    best_small_loss = np.inf
    best_iteration = -1
    stop_reason = "completed"
    stop_iteration = int(args.num_epochs) - 1
    train_curve: list[float] = []
    test_curve: list[float] = []
    target_key = "xdot" if resolve_objective(args) == "xdot" else "dx"
    eval_batch_size = max(1, int(args.eval_batch_size))
    eval_train_samples = max(0, int(args.eval_train_samples))
    eval_test_samples = max(0, int(args.eval_test_samples))

    def pick_eval_indices(total: int, limit: int) -> np.ndarray | None:
        if limit <= 0 or limit >= total:
            return None
        return np_rng.choice(total, size=limit, replace=False)

    eval_train_idx = pick_eval_indices(int(len(data["x"])), eval_train_samples)
    eval_test_idx = pick_eval_indices(int(len(data["test_x"])), eval_test_samples)

    def wrap_batch(state_x):
        return jax.vmap(wrap_coords)(jnp.asarray(state_x))

    def dataset_loss_batched(params_local, state_x, targets) -> float:
        total = int(state_x.shape[0])
        total_loss_sum = 0.0
        for start in range(0, total, eval_batch_size):
            end = min(total, start + eval_batch_size)
            total_loss_sum += float(
                loss_value(
                    params_local,
                    wrap_batch(state_x[start:end]),
                    jnp.asarray(targets[start:end]),
                    0.0,
                )
            )
        return total_loss_sum / total

    with tqdm(
        total=int(args.num_epochs),
        desc="train",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    ) as progress:
        for iteration in range(int(args.num_epochs)):
            rand_idx = np_rng.integers(0, len(data["x"]), size=(int(args.batch_size),), endpoint=False)
            state_batch = wrap_batch(data["x"][rand_idx])
            target_batch = jnp.asarray(data[target_key][rand_idx])
            opt_state, params = update_step(iteration, opt_state, state_batch, target_batch)
            params = get_params(opt_state)

            small_loss = float(loss_value(params, state_batch, target_batch, 0.0))
            if not np.isfinite(small_loss):
                stop_reason = "nan_batch_loss"
                stop_iteration = iteration
                break
            new_small_loss = small_loss < best_small_loss
            if new_small_loss:
                best_small_loss = small_loss

            if iteration % 100 == 0 or iteration == int(args.num_epochs) - 1:
                progress.set_postfix(step=int(iteration), batch_loss=f"{small_loss:.4f}")

            should_eval = False
            if bool(args.final_eval_only):
                should_eval = bool(iteration == int(args.num_epochs) - 1)
            else:
                if bool(args.eval_on_small_loss_improve) and new_small_loss:
                    should_eval = True
                if iteration == 0 or iteration == int(args.num_epochs) - 1:
                    should_eval = True
                elif int(args.warmup_eval_every) > 0 and iteration < int(args.warmup_eval_until) and iteration % int(args.warmup_eval_every) == 0:
                    should_eval = True
                elif int(args.eval_every) > 0 and iteration % int(args.eval_every) == 0:
                    should_eval = True
            if should_eval:
                eval_train_x = data["x"] if eval_train_idx is None else data["x"][eval_train_idx]
                eval_train_y = data[target_key] if eval_train_idx is None else data[target_key][eval_train_idx]
                eval_test_x = data["test_x"] if eval_test_idx is None else data["test_x"][eval_test_idx]
                eval_test_y = data["test_" + target_key] if eval_test_idx is None else data["test_" + target_key][eval_test_idx]

                train_loss = dataset_loss_batched(params, eval_train_x, eval_train_y)
                test_loss = dataset_loss_batched(params, eval_test_x, eval_test_y)
                train_curve.append(train_loss)
                test_curve.append(test_loss)
                progress.set_postfix(
                    step=int(iteration),
                    train_loss=f"{float(train_loss):.4f}",
                    test_loss=f"{float(test_loss):.4f}",
                )

                if np.isfinite(test_loss) and test_loss < best_loss:
                    best_loss = test_loss
                    best_params = tree_map(lambda x: x, params)
                    best_iteration = iteration
                    if checkpoint_callback is not None:
                        checkpoint_callback(
                            iteration=int(iteration),
                            params=best_params,
                            test_loss=float(test_loss),
                        )
                    print(f"[BEST] iteration={iteration} | test_loss={test_loss:.8e}")

                if not np.isfinite(test_loss):
                    stop_reason = "nan_test_loss"
                    stop_iteration = iteration
                    break

                print(f"iteration={iteration}, train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

            progress.update(1)

    final_params = get_params(opt_state)
    train_meta = {
        "completed": bool(stop_reason == "completed"),
        "stop_reason": str(stop_reason),
        "stop_iteration": int(stop_iteration),
        "best_iteration": int(best_iteration),
        "best_loss": float(best_loss),
    }
    return tree_map(lambda x: np.asarray(x), best_params), tree_map(lambda x: np.asarray(x), final_params), train_curve, test_curve, train_meta


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def save_direct_checkpoint(path: Path, params: dict[str, object], args: argparse.Namespace, model_config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_payload = {
        "params": tree_map(lambda x: np.asarray(x), params),
        "train_args": {
            "dt": float(args.dt),
            "batch_size": int(args.batch_size),
            "loss": str(args.loss),
            "objective": str(resolve_objective(args)),
        },
        "model_kind": str(args.model_kind),
        "model_config": model_config,
    }
    with path.open("wb") as fh:
        pickle.dump(ckpt_payload, fh)


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

    model_config = build_model_config(args)
    attempt_run_config = dict(run_config)
    attempt_run_config["attempt"] = {"index": int(attempt_idx), "seed": int(attempt_seed), "name": attempt_name}
    attempt_run_config["dataset"] = dataset_summary
    attempt_run_config["model_config"] = model_config
    (attempt_dir / "run_config.json").write_text(json.dumps(attempt_run_config, indent=2), encoding="utf-8")

    log_path = paths["results"] / "training.log"
    exception_text: str | None = None
    train_meta: dict[str, object] = {}
    train_curve: list[float] = []
    test_curve: list[float] = []
    best_params: dict[str, object] | None = None
    final_params: dict[str, object] | None = None

    start_time = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_fh, contextlib.redirect_stdout(Tee(sys.stdout, log_fh)):
        print(f"[attempt] index={attempt_idx} seed={attempt_seed} model_kind={args.model_kind}")
        try:
            def persist_best_checkpoint(*, iteration: int, params: dict[str, object], test_loss: float) -> None:
                save_direct_checkpoint(paths["checkpoints"] / "model_best.pkl", params, args, model_config)
                save_direct_checkpoint(base_out_dir / "checkpoints" / "model_best.pkl", params, args, model_config)

            with jax.disable_jit(bool(args.disable_jit)):
                best_params, final_params, train_curve, test_curve, train_meta = train_direct_model(
                    args=args,
                    model_kind=args.model_kind,
                    model_config=model_config,
                    data=data,
                    rng=jax.random.PRNGKey(int(attempt_seed)) + 500,
                    checkpoint_callback=persist_best_checkpoint,
                )
        except Exception:
            exception_text = traceback.format_exc()
            print(exception_text, flush=True)
    elapsed_s = time.perf_counter() - start_time

    if exception_text is not None:
        train_meta = {
            "completed": False,
            "stop_reason": "exception",
            "stop_iteration": -1,
            "best_iteration": -1,
            "best_loss": float("inf"),
        }
        best_params = best_params or {}
        final_params = final_params or {}

    save_direct_checkpoint(paths["checkpoints"] / "model_best.pkl", best_params, args, model_config)
    save_direct_checkpoint(base_out_dir / "checkpoints" / "model_best.pkl", best_params, args, model_config)
    ckpt_payload = {
        "params": tree_map(lambda x: np.asarray(x), best_params),
        "train_args": {
            "dt": float(args.dt),
            "batch_size": int(args.batch_size),
            "loss": str(args.loss),
        },
        "model_kind": str(args.model_kind),
        "model_config": model_config,
    }
    with (attempt_dir / "model_final.pkl").open("wb") as fh:
        pickle.dump({**ckpt_payload, "params": tree_map(lambda x: np.asarray(x), final_params)}, fh)

    rows = parse_training_log(log_path)
    save_loss_artifacts(rows, paths["results"], paths["plots"])

    summary = {
        "attempt_index": int(attempt_idx),
        "attempt_name": attempt_name,
        "seed": int(attempt_seed),
        "attempt_dir": str(attempt_dir),
        "model_kind": str(args.model_kind),
        "completed": bool(train_meta.get("completed", False)),
        "stop_reason": str(train_meta.get("stop_reason", "unknown")),
        "stop_iteration": int(train_meta.get("stop_iteration", -1)),
        "best_iteration": int(train_meta.get("best_iteration", -1)),
        "best_loss": float(train_meta.get("best_loss", float("inf"))),
        "elapsed_s": float(elapsed_s),
        "num_train_samples": int(dataset_summary["num_train_samples"]),
        "num_test_samples": int(dataset_summary["num_test_samples"]),
        "logged_points": int(len(rows)),
        "train_curve_points": int(len(train_curve)),
        "test_curve_points": int(len(test_curve)),
        "exception": exception_text,
        "files": {
            "model_best": str(paths["checkpoints"] / "model_best.pkl"),
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
        args.d_model = 64
        args.num_blocks = 1
        args.max_attempts = min(int(args.max_attempts), 2)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (LOCAL_ROOT / out_dir).resolve()
    paths = ensure_dirs(out_dir)
    (out_dir / "attempts").mkdir(parents=True, exist_ok=True)

    run_config = {
        "mode": "compare_direct_delta_cpu",
        "official_repo_root": str(OFFICIAL_REPO_ROOT),
        "official_repo_commit": official_commit_hash(OFFICIAL_REPO_ROOT),
        "jax_version": str(jax.__version__),
        "device": str(jax.devices()[0]),
        "args": vars(args),
        "model_config": build_model_config(args),
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
        if np.isfinite(best_loss_value) and (best_attempt is None or best_loss_value < float(best_attempt["best_loss"])):
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
                "model_kind",
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
    print(f"Model kind: {args.model_kind}")
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
