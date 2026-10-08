#!/usr/bin/env bash
# Copyright 2026 Limx Dynamics
#
# Native FluxVLA evaluation of the released PatchWAM RoboTwin 2.0 C2R
# checkpoint. The policy runs in the FluxVLA environment; the RoboTwin
# simulator runs in the interpreter selected by PATCHWAM_SIM_PYTHON.
#
#   bash scripts/eval_patchwam_robotwin.sh                 # one-task smoke
#   PATCHWAM_EPISODES=5 bash scripts/eval_patchwam_robotwin.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_patchwam_robotwin.sh \
#       --cfg-options eval.task_name=click_alarmclock \
#                     eval.task_config=demo_randomized
#
# Environment overrides consumed by configs/patchwam/patchwam_robotwin_c2r_eval.py:
#   PATCHWAM_CHECKPOINT, PATCHWAM_ASSETS_ROOT, PATCHWAM_FLUX2_ROOT,
#   PATCHWAM_FLUX2_SRC, PATCHWAM_TRANSFORMER, PATCHWAM_AE,
#   PATCHWAM_TEXT_ENCODER, PATCHWAM_TEXT_TOKENIZER, PATCHWAM_DATASET_STATS,
#   PATCHWAM_ROBOTWIN_ROOT, PATCHWAM_SIM_PYTHON
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PATCHWAM_EVAL_CONFIG:-configs/patchwam/patchwam_robotwin_c2r_eval.py}"
PYTHON="${PATCHWAM_EVAL_PYTHON:-python}"

cd "${REPO_ROOT}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    PATCHWAM_EPISODES="${PATCHWAM_EPISODES:-1}" \
    "${PYTHON}" scripts/eval.py --config "${CONFIG}" "$@"
