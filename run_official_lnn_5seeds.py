from __future__ import annotations

import argparse
import subprocess
import sys

from benchmark_5seeds_config import (
    OFFICIAL_LNN_OUT_DIR,
    OFFICIAL_MODEL_ARGS,
    LOCAL_ROOT,
    OFFICIAL_STABILITY_ARGS,
    SHARED_DATASET_ARGS,
    SHARED_DYNAMICS_ARGS,
    SHARED_EVAL_ARGS_JAX,
    SHARED_SEED_SWEEP_ARGS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the 5-seed official LNN benchmark run.")
    parser.add_argument("--out_dir", type=str, default=OFFICIAL_LNN_OUT_DIR)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(LOCAL_ROOT / "train_official_double_pendulum_cpu.py"),
        "--out_dir",
        str(args.out_dir),
        *SHARED_DATASET_ARGS,
        *SHARED_DYNAMICS_ARGS,
        *OFFICIAL_MODEL_ARGS,
        *OFFICIAL_STABILITY_ARGS,
        *SHARED_SEED_SWEEP_ARGS,
        "--sweep_all_attempts",
        "true",
        *SHARED_EVAL_ARGS_JAX,
        *args.extra_args,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=LOCAL_ROOT, check=True)


if __name__ == "__main__":
    main()
