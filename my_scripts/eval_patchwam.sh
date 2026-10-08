#!/usr/bin/env bash
# Native FluxVLA PatchWAM RoboTwin 2.0 official benchmark launcher.
#
#   bash my_scripts/eval_patchwam.sh                 # full 50 x 2 x 100 suite
#   bash my_scripts/eval_patchwam.sh --dry-run       # plan only, no model/sim
#   bash my_scripts/eval_patchwam.sh --list
#   bash my_scripts/eval_patchwam.sh --smoke         # 1 task, clean, 1 episode
#   bash my_scripts/eval_patchwam.sh --task adjust_bottle --condition clean
#   bash my_scripts/eval_patchwam.sh --episodes 4 --task adjust_bottle
#
# The whole benchmark is 50 tasks x (clean, random) x 100 episodes = 10,000
# episodes on one GPU, one case after another. The launcher resolves the
# repository root from its own location, picks the policy interpreter without
# relying on `python` being on PATH, and forwards every extra argument to
# tools/patchwam_official_benchmark.py (eval overrides go through
# `--cfg-options`, which must be the last argument).
#
# Environment:
#   PATCHWAM_EVAL_PYTHON   policy interpreter override
#   PATCHWAM_EVAL_CONFIG   eval config (default configs/patchwam/patchwam_robotwin_official_benchmark.py)
#   CUDA_VISIBLE_DEVICES   GPU selection (default 0)
#   PATCHWAM_*             asset/checkpoint/simulator overrides read by the config
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${REPO_ROOT}"

CONFIG="${PATCHWAM_EVAL_CONFIG:-configs/patchwam/patchwam_robotwin_official_benchmark.py}"

if [[ -n "${PATCHWAM_EVAL_PYTHON:-}" ]]; then
    PYTHON="${PATCHWAM_EVAL_PYTHON}"
elif [[ -x /home/chuong/miniconda3/envs/fluxvla/bin/python ]]; then
    PYTHON=/home/chuong/miniconda3/envs/fluxvla/bin/python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
else
    echo "eval_patchwam.sh: no policy interpreter found;" \
         "set PATCHWAM_EVAL_PYTHON" >&2
    exit 127
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "eval_patchwam.sh: config not found: ${CONFIG}" >&2
    exit 2
fi
if [[ ! -f tools/patchwam_official_benchmark.py ]]; then
    echo "eval_patchwam.sh: tools/patchwam_official_benchmark.py not found" \
         "under ${REPO_ROOT}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "eval_patchwam.sh: benchmark launcher python=${PYTHON} config=${CONFIG}" \
     "cuda=${CUDA_VISIBLE_DEVICES}" >&2

exec "${PYTHON}" tools/patchwam_official_benchmark.py \
    --python "${PYTHON}" --config "${CONFIG}" "$@"
