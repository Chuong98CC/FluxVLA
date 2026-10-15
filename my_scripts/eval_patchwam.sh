#!/usr/bin/env bash
# PatchWAM RoboTwin 2.0 official benchmark launcher.
#
# Thin convenience wrapper over the shared RoboTwin manager
# (`scripts/eval_robotwin_manager.sh`), which owns the task queue, one
# single-task eval worker per GPU and the summary aggregation. Evaluation is
# the shared FluxVLA `RobotwinEvalRunner`; this wrapper only maps a few flags
# onto the manager's environment interface.
#
#   bash my_scripts/eval_patchwam.sh --ckpt <checkpoint>
#   bash my_scripts/eval_patchwam.sh --ckpt <ckpt> --smoke
#   bash my_scripts/eval_patchwam.sh --ckpt <ckpt> --task adjust_bottle
#   bash my_scripts/eval_patchwam.sh --ckpt <ckpt> --condition random
#   bash my_scripts/eval_patchwam.sh --ckpt <ckpt> --episodes 4
#   bash my_scripts/eval_patchwam.sh --config <cfg> --ckpt <ckpt>
#
# Anything after `--` is forwarded verbatim to the manager (for example
# `-- --cfg-options eval.eval_chunk_size=10`).
#
# Environment:
#   PATCHWAM_EVAL_CONFIG   eval config (default
#                          configs/patchwam/patchwam_robotwin_official_benchmark.py)
#   PATCHWAM_CHECKPOINT    checkpoint when --ckpt is omitted
#   CUDA_VISIBLE_DEVICES, NUM_GPUS, MAX_TASKS_PER_GPU, OUTPUT_DIR
#                          forwarded to the manager
#   PATCHWAM_*             asset/checkpoint overrides read by the config
#
# The whole benchmark is 50 tasks x (clean, random) x 100 trials. There is no
# plan-only mode: the manager always starts real workers when the suite is not
# already complete.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${REPO_ROOT}"

CONFIG="${PATCHWAM_EVAL_CONFIG:-configs/patchwam/patchwam_robotwin_official_benchmark.py}"
CKPT="${PATCHWAM_CHECKPOINT:-}"
TASK_NAMES_ARG=""
SUITE_ARG=""
TRIALS_ARG=""
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
            TASK_NAMES_ARG="${2:?--task needs a task name}"
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
            sed -n '2,30p' "${BASH_SOURCE[0]}"
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

echo "eval_patchwam.sh: manager config=${CONFIG} ckpt=${CKPT}" \
     "cuda=${CUDA_VISIBLE_DEVICES}" >&2

exec bash scripts/eval_robotwin_manager.sh "${CONFIG}" "${CKPT}" \
    ${FORWARDED[@]+"${FORWARDED[@]}"}
