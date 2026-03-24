from __future__ import annotations

from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parent

EXPERIMENT_ROOT = "experiments/benchmark_5seeds_temporal"
OFFICIAL_LNN_OUT_DIR = f"{EXPERIMENT_ROOT}/official_lnn"
BASELINE_MLP_OUT_DIR = f"{EXPERIMENT_ROOT}/baseline_mlp"
TRANSFORMER_LNN_OUT_DIR = f"{EXPERIMENT_ROOT}/transformer_lnn"

# Shared temporal dataset:
# follow the user's requested temporal setup: many initial conditions and many
# sequential steps per condition, reusing the same temporal-generation scheme as
# the earlier ~25k-transition runs but at much larger scale.
PAPERLIKE_DT = "0.1"

SHARED_DATASET_ARGS = [
    "--data_seed",
    "0",
    "--dataset_mode",
    "paperlike_temporal",
    "--objective",
    "delta",
    "--samples",
    "300000",
    "--trajectory_steps",
    "250",
    # In this code path test_split still acts as the train fraction over
    # trajectories, so 0.9 means 270000 train trajectories and 30000 test
    # trajectories, each with 250 transitions.
    "--test_split",
    "0.9",
]

SHARED_DYNAMICS_ARGS = [
    "--dt",
    PAPERLIKE_DT,
    "--num_epochs",
    "100000",
    "--batch_size",
    "32",
    "--lr",
    "5e-4",
    "--lr2",
    "1e-6",
]

OFFICIAL_MODEL_ARGS = [
    "--act",
    "softplus",
    "--hidden_dim",
    "500",
    "--layers",
    "4",
    "--l2reg",
    "1e-6",
    "--lr",
    "1e-4",
    "--lr2",
    "1e-6",
]

OFFICIAL_STABILITY_ARGS = [
    "--grad_clip",
    "0.5",
    "--lr_warmup_steps",
    "5000",
    "--param_check_every",
    "25",
    "--latest_checkpoint_every",
    "100",
]

SHARED_SEED_SWEEP_ARGS = [
    "--seed",
    "0",
    "--max_attempts",
    "5",
    "--seed_stride",
    "1",
]

SHARED_EVAL_ARGS_JAX = [
    "--eval_batch_size",
    "8192",
    "--eval_train_samples",
    "100000",
    "--eval_test_samples",
    "100000",
    "--final_eval_only",
    "true",
    "--eval_every",
    "0",
    "--warmup_eval_every",
    "0",
    "--warmup_eval_until",
    "0",
    "--eval_on_small_loss_improve",
    "false",
]

TRANSFORMER_LNN_ARGS = [
    "--device",
    "cuda",
    "--objective",
    "delta",
    "--num_steps",
    "100000",
    "--batch_size",
    "32",
    "--lr",
    "1e-3",
    "--lr2",
    "1e-5",
    "--l2reg",
    "1e-2",
    "--dt",
    PAPERLIKE_DT,
    "--n_updates",
    "1",
    "--d_model",
    "64",
    "--num_heads",
    "4",
    "--num_layers",
    "1",
    "--ff_multiplier",
    "2",
    "--lagrangian_form",
    "structured_tv",
    "--mass_eps",
    "1e-3",
    "--grad_clip",
    "1.0",
    "--loss_abort_threshold",
    "10000",
    "--eval_batch_size",
    "256",
    "--eval_train_samples",
    "100000",
    "--eval_test_samples",
    "100000",
    "--final_eval_only",
    "true",
    "--eval_every",
    "0",
    "--quick_eval_every",
    "0",
    "--quick_eval_until",
    "0",
    "--final_eval_mode",
    "full",
]
