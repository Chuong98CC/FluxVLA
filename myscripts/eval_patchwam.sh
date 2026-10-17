#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
export OUTPUT_DIR=runs/evaluation/my_run
export PATCHWAM_DATASET_STATS='work_dirs/patchwam_robotwin_released_conversion/dataset_statistics.json'
export TASK_NAMES="${1:-}"
export TASK_SUITE_NAME="${2:-clean random}"

CONFIG=configs/patchwam/robotwin/patchwam_robotwin.py
CKPT=work_dirs/patchwam_robotwin_released_conversion/step_140000_native.safetensors

exec bash scripts/eval_robotwin_manager.sh "$CONFIG" "$CKPT"
