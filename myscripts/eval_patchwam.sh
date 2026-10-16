#!/usr/bin/env bash
# Usage:
# ./myscripts/eval_patchwam.sh [task_names] [task_suite_name]
# (task_names and task_suite_name are optional, default to all tasks in clean random suite)
# Example:
# bash myscripts/eval_patchwam.sh /path/to/checkpoint.pt "" clean --> will evaluate all tasks in the clean suite
# bash myscripts/eval_patchwam.sh "turn_switch rotate_qrcode" clean --> will evaluate only the specified tasks in the clean suite
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-100}"
export OUTPUT_DIR=runs/evaluation/my_run
export PATCHWAM_DATASET_STATS='work_dirs/patchwam_robotwin_released_conversion/dataset_statistics.json'
export TASK_NAMES="${1:-}"
export TASK_SUITE_NAME="${2:-clean random}"


CONFIG=configs/patchwam/robotwin/patchwam_robotwin_vlm_full_finetune.py
# CKPT=checkpoints/Pretrained_models/patchwam-robotwin/step_140000.pt
CKPT=work_dirs/patchwam_robotwin_released_conversion/step_140000_native.safetensors

exec bash scripts/eval_robotwin_manager.sh "$CONFIG" "$CKPT"
