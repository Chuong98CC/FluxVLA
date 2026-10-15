#!/usr/bin/env bash
# PatchWAM RoboTwin 2.0 official benchmark launcher.
#
# Thin convenience wrapper over the shared RoboTwin manager
# (`scripts/eval_robotwin_manager.sh`), which owns the task queue, one
# single-task eval worker per GPU and the summary aggregation. Evaluation is
# the shared FluxVLA `RobotwinEvalRunner`; this wrapper only maps a few flags
# onto the manager's environment interface.
#
# bash my_scripts/eval_patchwam.sh \
#   --suite-root runs/evaluation/patchwam_quick_check \
#   --task rotate_qrcode \
#   --condition clean \
#   --episodes 10
#   --ckpt work_dirs/patchwam_robotwin_released_conversion/step_140000_native.safetensors \
#
# Anything after `--` is forwarded verbatim to the manager (for example
# `-- --cfg-options eval.eval_chunk_size=10`).
# `--suite-root` sets OUTPUT_DIR; legacy evaluation results are not resumed.
#
# Defaults target the released PatchWAM C2R checkpoint. Its published
# `step_140000.pt` is an ImageWAM-lineage payload that the shared evaluator
# cannot strict-load directly, so the default checkpoint is the native
# conversion produced by:
#
#   python tools/convert_patchwam_robotwin_released.py
#
# (Converted files live under work_dirs and are never checked in; the released
# originals are left untouched.)
#
# Environment:
#   PATCHWAM_EVAL_CONFIG   eval config (default
#                          configs/patchwam/patchwam_robotwin_released_eval.py)
#   PATCHWAM_CHECKPOINT    checkpoint when --ckpt is omitted
#   CUDA_VISIBLE_DEVICES, NUM_GPUS, MAX_TASKS_PER_GPU, OUTPUT_DIR
#                          forwarded to the manager
#   PATCHWAM_*             asset/checkpoint overrides read by the config
#
# Defaults are the released config plus the converted checkpoint: two tasks
# (`adjust_bottle`, `rotate_qrcode`), the `clean` condition, one trial each.
# Override any part with the flags above or the PATCHWAM_* variables. There is
# no plan-only mode: the manager starts real workers for every task whose
# summary is not already complete.
#
# The released 50-task protocol is the same wrapper with the released task list,
# both conditions and 100 trials:
#   bash my_scripts/eval_patchwam.sh \
#     --suite-root runs/evaluation/patchwam_released_official \
#     --task "$(python -c "import json;print(' '.join(json.load(open( \
#       'configs/patchwam/patchwam_robotwin_official_benchmark_tasks.json')) \
#       ['tasks']))")" \
#     --condition "clean random" --episodes 100
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${REPO_ROOT}"

CONFIG="${PATCHWAM_EVAL_CONFIG:-configs/patchwam/patchwam_robotwin_released_eval.py}"
CKPT="${PATCHWAM_CHECKPOINT:-work_dirs/patchwam_robotwin_released_conversion/step_140000_native.safetensors}"
TASK_NAMES_ARG=""
SUITE_ARG=""
TRIALS_ARG=""
OUTPUT_DIR_ARG=""
FORWARDED=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)
            CKPT="${2:?--ckpt needs a checkpoint path}"
            shift 2
            ;;
        --config)
            CONFIG="${2:?--config needs a config path}"
            shift 2
            ;;
        --task)
            TASK_NAMES_ARG+="${TASK_NAMES_ARG:+ }${2:?--task needs a task name}"
            shift 2
            ;;
        --suite-root)
            OUTPUT_DIR_ARG="${2:?--suite-root needs an output directory}"
            shift 2
            ;;
        --condition)
            SUITE_ARG="${2:?--condition needs clean or random}"
            shift 2
            ;;
        --episodes)
            TRIALS_ARG="${2:?--episodes needs a count}"
            shift 2
            ;;
        --smoke)
            TASK_NAMES_ARG="adjust_bottle"
            SUITE_ARG="clean"
            TRIALS_ARG="1"
            shift
            ;;
        --)
            shift
            FORWARDED=("$@")
            break
            ;;
        -h|--help)
            sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "eval_patchwam.sh: unknown argument: $1" >&2
            echo "Pass manager arguments after -- (see --help)." >&2
            exit 2
            ;;
    esac
done

if [[ -z "${CKPT}" ]]; then
    echo "eval_patchwam.sh: pass --ckpt <checkpoint> or set PATCHWAM_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "eval_patchwam.sh: config not found: ${CONFIG}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ -n "${TASK_NAMES_ARG}" ]] && export TASK_NAMES="${TASK_NAMES_ARG}"
[[ -n "${SUITE_ARG}" ]] && export TASK_SUITE_NAME="${SUITE_ARG}"
[[ -n "${TRIALS_ARG}" ]] && export NUM_TRIALS_PER_TASK="${TRIALS_ARG}"
[[ -n "${OUTPUT_DIR_ARG}" ]] && export OUTPUT_DIR="${OUTPUT_DIR_ARG}"

echo "eval_patchwam.sh: manager config=${CONFIG} ckpt=${CKPT}" \
     "cuda=${CUDA_VISIBLE_DEVICES}" >&2

exec bash scripts/eval_robotwin_manager.sh "${CONFIG}" "${CKPT}" \
    ${FORWARDED[@]+"${FORWARDED[@]}"}
