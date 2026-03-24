from __future__ import annotations

import argparse
import subprocess
import sys

from benchmark_5seeds_config import (
    OFFICIAL_LNN_OUT_DIR,
    LOCAL_ROOT,
    SHARED_SEED_SWEEP_ARGS,
    TRANSFORMER_LNN_OUT_DIR,
    TRANSFORMER_LNN_ARGS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the 5-seed Transformer-LNN benchmark run.")
    parser.add_argument("--out_dir", type=str, default=TRANSFORMER_LNN_OUT_DIR)
    parser.add_argument("--reference_out_dir", type=str, default=OFFICIAL_LNN_OUT_DIR)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(LOCAL_ROOT / "train_transformer_lnn_torch.py"),
        "--out_dir",
        str(args.out_dir),
        "--reference_out_dir",
        str(args.reference_out_dir),
        *SHARED_SEED_SWEEP_ARGS,
        "--sweep_all_attempts",
        "true",
        *TRANSFORMER_LNN_ARGS,
        *args.extra_args,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=LOCAL_ROOT, check=True)


if __name__ == "__main__":
    main()
