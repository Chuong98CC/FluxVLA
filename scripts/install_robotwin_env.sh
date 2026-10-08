#!/usr/bin/env bash
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portable installer for the dedicated RoboTwin 2.0 simulator environment used
# by the PatchWAM RoboTwin evaluation bridge.
#
# The script creates (or reuses) a Conda environment, installs the pinned
# simulator dependency closure and PyTorch from the selected CUDA index, applies
# the RoboTwin SAPIEN/mplib source fixes and the Open Image Denoise 2.3.3
# upgrade, builds the cuRobo CUDA extensions against the selected Torch, and
# prints the PATCHWAM_SIM_PYTHON / PATCHWAM_ROBOTWIN_ROOT exports the evaluation
# config consumes. It never touches the policy environment, an existing
# RoboTwin checkout's local changes, or system drivers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${PATCHWAM_ROBOTWIN_ENV_NAME:-Robotwin}"
PREFIX="${PATCHWAM_ROBOTWIN_PREFIX:-}"
ROBOTWIN_ROOT="${PATCHWAM_ROBOTWIN_ROOT:-}"
PROFILE="${PATCHWAM_ROBOTWIN_PROFILE:-auto}"
DRY_RUN=0
WITH_ASSETS=0
SKIP_VERIFY=0
CUDA_HOME_OVERRIDE=""
LOCAL_OIDN_ARCHIVE=""

ROBOTWIN_REPO="${PATCHWAM_ROBOTWIN_REPO:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_REF="${PATCHWAM_ROBOTWIN_REF:-bf44be51cf5717a5595ce59447f2cf5263d2aa95}"
ROBOTWIN_SRC_ROOT="${PATCHWAM_ROBOTWIN_SRC_ROOT:-${PROJECT_ROOT}/src}"
CUROBO_REPO="${PATCHWAM_CUROBO_REPO:-https://github.com/NVlabs/curobo.git}"
CUROBO_REF="${PATCHWAM_CUROBO_REF:-v0.7.8}"
CUROBO_SRC_ROOT="${PATCHWAM_CUROBO_SRC_ROOT:-${PROJECT_ROOT}/src}"
OIDN_URL="${PATCHWAM_OIDN_URL:-https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz}"
OIDN_SHA256="${PATCHWAM_OIDN_SHA256:-3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f}"
OIDN_CACHE="${PATCHWAM_OIDN_CACHE:-${XDG_CACHE_HOME:-${HOME}/.cache}/fluxvla/oidn}"
REQUIREMENTS_FILE="${PATCHWAM_ROBOTWIN_REQUIREMENTS:-${SCRIPT_DIR}/requirements-robotwin-sim.txt}"

PIP_RETRIES="${PATCHWAM_PIP_RETRIES:-3}"
PIP_TIMEOUT="${PATCHWAM_PIP_TIMEOUT:-900}"
GIT_RETRIES="${PATCHWAM_GIT_RETRIES:-3}"
GIT_CLONE_TIMEOUT="${PATCHWAM_GIT_CLONE_TIMEOUT:-1800}"
CONDA_INSTALL_TIMEOUT="${PATCHWAM_CONDA_INSTALL_TIMEOUT:-3600}"
CUROBO_BUILD_JOBS="${PATCHWAM_CUROBO_BUILD_JOBS:-8}"

SETUPTOOLS_PIN="setuptools==69.5.1"

PYTHON_VERSION=""
TORCH_VERSION=""
TORCHVISION_VERSION=""
TORCH_INDEX_URL=""
EXPECTED_CUDA=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/install_robotwin_env.sh [options]

Create (or reuse) the dedicated RoboTwin 2.0 simulator Conda environment and
print the PATCHWAM_SIM_PYTHON / PATCHWAM_ROBOTWIN_ROOT exports the PatchWAM
RoboTwin evaluation config consumes.

Options:
  --env-name NAME        Conda environment name when --prefix is not given.
                         Default: Robotwin.
  --prefix PATH          Explicit Conda environment prefix. Overrides
                         --env-name and never removes an existing environment.
  --robotwin-root PATH   Existing RoboTwin checkout to use. When omitted the
                         pinned upstream revision is cloned into
                         <repo>/src/RoboTwin (or PATCHWAM_ROBOTWIN_SRC_ROOT).
  --profile PROFILE      CUDA/Torch profile: auto, cu128 or cu121.
                         auto prefers cu128 when the local toolkit/driver
                         supports CUDA >= 12.8 and falls back to cu121.
                         Default: auto.
  --cuda-home PATH       CUDA toolkit used to build the cuRobo extensions.
                         The selected toolkit is authoritative: its bin/nvcc
                         decides the `auto` profile and the build, even when a
                         different nvcc is on PATH. Default: $CUDA_HOME, else
                         the directory of nvcc, else /usr/local/cuda.
  --oidn-archive PATH    Use a local oidn-2.3.3.x86_64.linux.tar.gz instead of
                         downloading it. Its SHA256 is still verified.
  --with-assets          Run the upstream RoboTwin asset downloader after the
                         environment is installed (large Hugging Face download).
                         If the downloader or the asset layout cannot be
                         prepared, the installer exits non-zero and reports
                         that the environment itself is installed.
  --skip-verify          Skip the post-install import/CUDA verification.
  --dry-run              Print every command without creating the environment,
                         downloading, building or writing anything. Works
                         without Conda, network access or an existing checkout.
  -h, --help             Show this help.

Profiles:
  cu128  Python 3.11.15, torch 2.7.1+cu128, CUDA toolkit >= 12.8 (Blackwell).
         Matches the verified RoboTwin environment used for PatchWAM.
  cu121  Python 3.11.15, torch 2.4.1+cu121, CUDA toolkit >= 12.1. The Torch
         pin follows the upstream RoboTwin 2.0 documentation for pre-Blackwell
         GPUs, with Python 3.11 instead of upstream's 3.10 because the pinned
         closure (scipy 1.17.1, zarr 3.1.5, matplotlib 3.11.2, networkx 3.6.1,
         scikit-image 0.26.0) declares Requires-Python >= 3.11.

Requirements: Linux x86_64 with an NVIDIA GPU, a CUDA toolkit matching the
selected profile, and git, curl, tar, a C++ compiler, cmake and ninja on PATH.
Missing tools are reported with an actionable message; the installer never runs
apt/sudo or installs drivers.

Environment variables:
  PATCHWAM_ROBOTWIN_ENV_NAME, PATCHWAM_ROBOTWIN_PREFIX
                      Defaults for --env-name / --prefix.
  PATCHWAM_ROBOTWIN_ROOT
                      Default for --robotwin-root.
  PATCHWAM_ROBOTWIN_PROFILE
                      Default for --profile.
  PATCHWAM_ROBOTWIN_SRC_ROOT
                      Directory that receives the pinned upstream clone.
                      Default: <repo>/src.
  PATCHWAM_CUROBO_SRC_ROOT
                      Directory that receives the pinned cuRobo checkout.
                      Default: <repo>/src. cuRobo is built from a separate
                      managed checkout so a user-supplied RoboTwin root is
                      never modified.
  PATCHWAM_ROBOTWIN_REPO / PATCHWAM_ROBOTWIN_REF
                      RoboTwin repository and pinned revision.
  PATCHWAM_CUROBO_REPO / PATCHWAM_CUROBO_REF
                      cuRobo repository and pinned tag. Default: NVlabs/curobo
                      v0.7.8.
  PATCHWAM_OIDN_URL / PATCHWAM_OIDN_SHA256 / PATCHWAM_OIDN_CACHE
                      Open Image Denoise 2.3.3 archive URL, SHA256 and download
                      cache. Default: official RenderKit release.
  PATCHWAM_ROBOTWIN_REQUIREMENTS
                      Pinned pip requirement manifest.
  PATCHWAM_CONDA_ENVS_DIR
                      Directory holding named Conda environments. Default:
                      `conda info --base`/envs.
  PATCHWAM_PIP_RETRIES, PATCHWAM_PIP_TIMEOUT
                      pip --retries / --timeout values.
  PATCHWAM_CUROBO_BUILD_JOBS
                      MAX_JOBS/CMAKE_BUILD_PARALLEL_LEVEL for the cuRobo build.
                      Default: 8.
  PIP_INDEX_URL       Forwarded to the pinned requirement install so mirrors
                      and proxies keep working exactly as in the policy env.
  PATCHWAM_CUROBO_EDITABLE=1
                      Install cuRobo in editable mode (`pip install -e .`)
                      instead of the portable non-editable build.

Examples:
  bash scripts/install_robotwin_env.sh --dry-run
  bash scripts/install_robotwin_env.sh
  bash scripts/install_robotwin_env.sh --env-name Robotwin --profile cu128
  bash scripts/install_robotwin_env.sh --robotwin-root /data/RoboTwin
  bash scripts/install_robotwin_env.sh --with-assets
EOF
}

log() {
  printf '%s\n' "$*"
}

warn() {
  printf 'Warning: %s\n' "$*" >&2
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

run() {
  printf '+ %s\n' "$*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

run_with_timeout() {
  local timeout_seconds="$1"
  shift
  printf '+ %s\n' "$*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_seconds}" "$@"
  else
    "$@"
  fi
}

run_in_dir() {
  local dir="$1"
  shift
  printf '+ (cd %s && %s)\n' "${dir}" "$*"
  if [[ "${DRY_RUN}" == "0" ]]; then
    ( cd "${dir}" && "$@" )
  fi
}

sha256_of() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
  else
    shasum -a 256 "${file}" | awk '{print $1}'
  fi
}

abspath() {
  local path="$1"
  if [[ "${path}" == /* ]]; then
    printf '%s' "${path}"
  else
    printf '%s/%s' "${PWD}" "${path}"
  fi
}

find_conda() {
  if [[ -n "${PATCHWAM_CONDA_BIN:-}" ]]; then
    printf '%s' "${PATCHWAM_CONDA_BIN}"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf '%s' "${CONDA_EXE}"
    return 0
  fi
  local name
  for name in conda mamba micromamba; do
    if command -v "${name}" >/dev/null 2>&1; then
      command -v "${name}"
      return 0
    fi
  done
  return 1
}

detect_nvcc_version() {
  # Prefer the explicitly selected toolkit, then the nvcc found on PATH.
  local nvcc_bin="${1:-}"
  if [[ -z "${nvcc_bin}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
      nvcc_bin="$(command -v nvcc)"
    else
      return 0
    fi
  fi
  if [[ -x "${nvcc_bin}" ]]; then
    "${nvcc_bin}" --version 2>/dev/null \
      | sed -n 's/.*release[[:space:]]*\([0-9][0-9.]*\),.*/\1/p' | head -n 1
  fi
}

detect_driver_cuda_version() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>/dev/null \
      | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -n 1
  fi
}

version_ge() {
  local version="$1" minimum="$2"
  [[ -n "${version}" ]] || return 1
  awk -v version="${version}" -v minimum="${minimum}" 'BEGIN {
    split(version, v, "."); split(minimum, m, ".")
    vmaj = v[1] + 0; vmin = v[2] + 0; mmaj = m[1] + 0; mmin = m[2] + 0
    exit !((vmaj > mmaj) || (vmaj == mmaj && vmin >= mmin))
  }'
}

resolve_profile() {
  local requested="$1" nvcc_version="${2:-}" driver_version
  if [[ "${requested}" != "auto" ]]; then
    printf '%s' "${requested}"
    return 0
  fi
  driver_version="$(detect_driver_cuda_version)"
  printf 'CUDA detection: nvcc=%s driver=%s\n' \
    "${nvcc_version:-unknown}" "${driver_version:-unknown}" >&2
  if version_ge "${nvcc_version}" "12.8"; then
    printf 'cu128'
    return 0
  fi
  if [[ -n "${nvcc_version}" ]]; then
    # The CUDA toolkit selected for the cuRobo build is authoritative.
    printf 'cu121'
    return 0
  fi
  if version_ge "${driver_version}" "12.8"; then
    printf 'cu128'
  else
    printf 'cu121'
  fi
}

apply_profile() {
  case "$1" in
    cu128)
      PYTHON_VERSION="3.11.15"
      TORCH_VERSION="2.7.1"
      TORCHVISION_VERSION="0.22.1"
      TORCH_INDEX_URL="${PATCHWAM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
      EXPECTED_CUDA="12.8"
      ;;
    cu121)
      # Python 3.11 for both profiles: the pinned closure (scipy 1.17.1,
      # zarr 3.1.5, matplotlib 3.11.2, networkx 3.6.1, scikit-image 0.26.0)
      # declares Requires-Python >= 3.11, so upstream's Python 3.10 recipe
      # cannot install these pins. torch 2.4.1+cu121 has cp311 wheels.
      PYTHON_VERSION="3.11.15"
      TORCH_VERSION="2.4.1"
      TORCHVISION_VERSION="0.19.1"
      TORCH_INDEX_URL="${PATCHWAM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
      EXPECTED_CUDA="12.1"
      ;;
    *)
      die "--profile must be one of: auto, cu128, cu121 (got '${1}')"
      ;;
  esac
  PYTHON_MAJOR_MINOR="${PYTHON_VERSION%.*}"
}

resolve_default_prefix() {
  local base=""
  if [[ -n "${PATCHWAM_CONDA_ENVS_DIR:-}" ]]; then
    printf '%s/%s' "${PATCHWAM_CONDA_ENVS_DIR%/}" "${ENV_NAME}"
    return 0
  fi
  if [[ -n "${CONDA_BIN:-}" ]]; then
    base="$("${CONDA_BIN}" info --base 2>/dev/null || true)"
  fi
  if [[ -z "${base}" ]]; then
    base="${HOME}/miniconda3"
  fi
  printf '%s/envs/%s' "${base%/}" "${ENV_NAME}"
}

resolve_cuda_home() {
  if [[ -n "${CUDA_HOME_OVERRIDE}" ]]; then
    printf '%s' "${CUDA_HOME_OVERRIDE}"
    return 0
  fi
  if [[ -n "${CUDA_HOME:-}" ]]; then
    printf '%s' "${CUDA_HOME}"
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    printf '%s' "$(dirname "$(dirname "${nvcc_path}")")"
    return 0
  fi
  if [[ -d /usr/local/cuda ]]; then
    printf '%s' "/usr/local/cuda"
  fi
}

pip_install() {
  run "${ENV_PYTHON}" -m pip install \
    --retries "${PIP_RETRIES}" --timeout "${PIP_TIMEOUT}" "$@"
}

retry_command() {
  local attempts="$1"
  shift
  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi
    if (( attempt >= attempts )); then
      return 1
    fi
    warn "command failed (attempt ${attempt}/${attempts}); retrying: $*"
    sleep 5
    attempt=$(( attempt + 1 ))
  done
}

checkout_repo() {
  local label="$1" repo="$2" ref="$3" dest="$4"
  if [[ -d "${dest}/.git" ]]; then
    local current status
    current="$(git -C "${dest}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${current}" == "${ref}" ]]; then
      log "${label}: ${dest} already at ${ref}"
      return 0
    fi
    status="$(git -C "${dest}" status --porcelain 2>/dev/null || true)"
    if [[ -n "${status}" ]]; then
      die "${label} at ${dest} has local changes and is not at ${ref}. Commit/stash them, pass --robotwin-root to another checkout, or reset manually; this installer never resets a user checkout."
    fi
    log "${label}: updating ${dest} to ${ref}"
    retry_command "${GIT_RETRIES}" run_with_timeout "${GIT_CLONE_TIMEOUT}" \
      git -C "${dest}" fetch --tags origin "${ref}" \
      || die "could not fetch ${ref} for ${label} in ${dest}"
    run git -C "${dest}" checkout --detach FETCH_HEAD
  else
    if [[ -e "${dest}" && -n "$(ls -A "${dest}" 2>/dev/null || true)" ]]; then
      die "${label} destination ${dest} exists and is not a git checkout; move it aside or pass --robotwin-root"
    fi
    run mkdir -p "$(dirname "${dest}")"
    retry_command "${GIT_RETRIES}" run_with_timeout "${GIT_CLONE_TIMEOUT}" \
      git clone --filter=blob:none --no-checkout "${repo}" "${dest}" \
      || die "could not clone ${label} from ${repo}"
    retry_command "${GIT_RETRIES}" run_with_timeout "${GIT_CLONE_TIMEOUT}" \
      git -C "${dest}" fetch --tags origin "${ref}" \
      || die "could not fetch ${ref} for ${label} in ${dest}"
    run git -C "${dest}" checkout --detach FETCH_HEAD
  fi
  if [[ -f "${dest}/.gitmodules" ]]; then
    run git -C "${dest}" submodule update --init --recursive
  fi
}

download_oidn_archive() {
  local archive=""
  if [[ -n "${LOCAL_OIDN_ARCHIVE}" ]]; then
    archive="${LOCAL_OIDN_ARCHIVE}"
    [[ -f "${archive}" ]] || die "--oidn-archive not found: ${archive}"
    printf '%s' "${archive}"
    return 0
  fi
  archive="${OIDN_CACHE}/$(basename "${OIDN_URL}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '+ mkdir -p %s\n' "${OIDN_CACHE}" >&2
    printf '+ curl -L --fail --retry 5 --connect-timeout 30 -C - -o %s %s\n' \
      "${archive}" "${OIDN_URL}" >&2
    printf '%s' "${archive}"
    return 0
  fi
  mkdir -p "${OIDN_CACHE}"
  if [[ -f "${archive}" ]] && [[ "$(sha256_of "${archive}")" == "${OIDN_SHA256}" ]]; then
    printf 'OIDN archive already cached: %s\n' "${archive}" >&2
    printf '%s' "${archive}"
    return 0
  fi
  local attempt=1
  while true; do
    printf '+ curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 -C - -o %s %s\n' \
      "${archive}" "${OIDN_URL}" >&2
    # Everything the downloader writes goes to stderr: stdout is the returned
    # archive path and nothing else.
    if curl -L --fail --retry 5 --retry-delay 2 --connect-timeout 30 \
        -C - -o "${archive}" "${OIDN_URL}" 1>&2; then
      break
    fi
    if (( attempt >= 3 )); then
      die "could not download ${OIDN_URL}; pass --oidn-archive with a local oidn-2.3.3.x86_64.linux.tar.gz"
    fi
    printf 'download attempt %d/3 failed; retrying\n' "${attempt}" >&2
    attempt=$(( attempt + 1 ))
    sleep 5
  done
  if [[ "$(sha256_of "${archive}")" != "${OIDN_SHA256}" ]]; then
    die "downloaded ${archive} does not match the expected SHA256 ${OIDN_SHA256}; re-download it or set PATCHWAM_OIDN_SHA256 to the published value"
  fi
  printf '%s' "${archive}"
}

find_assets_downloader() {
  local candidate
  for candidate in \
      "${ROBOTWIN_ROOT}/scripts/_download_assets.sh" \
      "${ROBOTWIN_ROOT}/script/_download_assets.sh"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  return 1
}

print_exports() {
  log ""
  log "Evaluation exports (also written to ${PREFIX}/etc/conda/activate.d/patchwam-robotwin.sh):"
  log "  export PATCHWAM_SIM_PYTHON=$(printf '%q' "${PREFIX}/bin/python")"
  log "  export PATCHWAM_ROBOTWIN_ROOT=$(printf '%q' "${ROBOTWIN_ROOT}")"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die "--env-name requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || die "--prefix requires a value"
      PREFIX="$2"
      shift 2
      ;;
    --robotwin-root)
      [[ $# -ge 2 ]] || die "--robotwin-root requires a value"
      ROBOTWIN_ROOT="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      PROFILE="$2"
      shift 2
      ;;
    --cuda-home)
      [[ $# -ge 2 ]] || die "--cuda-home requires a value"
      CUDA_HOME_OVERRIDE="$2"
      shift 2
      ;;
    --oidn-archive)
      [[ $# -ge 2 ]] || die "--oidn-archive requires a value"
      LOCAL_OIDN_ARCHIVE="$2"
      shift 2
      ;;
    --with-assets)
      WITH_ASSETS=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${PROFILE}" in
  auto|cu128|cu121) ;;
  *)
    die "--profile must be one of: auto, cu128, cu121 (got '${PROFILE}')"
    ;;
esac

HOST_OS="$(uname -s 2>/dev/null || echo unknown)"
HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
if [[ "${HOST_OS}" != "Linux" || "${HOST_ARCH}" != "x86_64" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    warn "host ${HOST_OS}/${HOST_ARCH} is not Linux x86_64; this installer only supports Linux x86_64 NVIDIA systems"
  else
    die "unsupported host ${HOST_OS}/${HOST_ARCH}: this installer supports Linux x86_64 NVIDIA systems only (the PyTorch wheels and the Open Image Denoise archive are x86_64)"
  fi
fi

REQUIREMENTS_FILE="$(abspath "${REQUIREMENTS_FILE}")"
if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  die "requirements manifest not found: ${REQUIREMENTS_FILE}"
fi

if [[ -z "${ROBOTWIN_ROOT}" ]]; then
  ROBOTWIN_ROOT="${ROBOTWIN_SRC_ROOT%/}/RoboTwin"
  MANAGED_ROBOTWIN_ROOT=1
else
  MANAGED_ROBOTWIN_ROOT=0
fi
ROBOTWIN_ROOT="$(abspath "${ROBOTWIN_ROOT}")"
CUROBO_DIR="$(abspath "${CUROBO_SRC_ROOT%/}/curobo")"
OIDN_CACHE="$(abspath "${OIDN_CACHE}")"
if [[ -n "${LOCAL_OIDN_ARCHIVE}" ]]; then
  LOCAL_OIDN_ARCHIVE="$(abspath "${LOCAL_OIDN_ARCHIVE}")"
fi

CONDA_BIN="$(find_conda || true)"
if [[ -z "${CONDA_BIN}" && "${DRY_RUN}" == "0" ]]; then
  die "no Conda/mamba found. Install Miniconda or Miniforge first (this installer never runs apt/sudo)."
fi
CONDA_BIN_RESOLVED="${CONDA_BIN}"
CONDA_BIN="${CONDA_BIN:-conda}"

if [[ -z "${PREFIX}" ]]; then
  PREFIX="$(resolve_default_prefix)"
fi
PREFIX="$(abspath "${PREFIX}")"

ENV_PYTHON="${PREFIX}/bin/python"

# CUDA toolkit selection happens before the profile is resolved so that an
# explicit --cuda-home always wins over whatever nvcc happens to be on PATH.
CUDA_HOME_RESOLVED="$(resolve_cuda_home)"
if [[ -n "${CUDA_HOME_RESOLVED}" ]]; then
  CUDA_HOME_RESOLVED="$(abspath "${CUDA_HOME_RESOLVED}")"
fi
if [[ -n "${CUDA_HOME_OVERRIDE}" ]]; then
  if [[ -x "${CUDA_HOME_RESOLVED}/bin/nvcc" ]]; then
    CUDA_NVCC_VERSION="$(detect_nvcc_version "${CUDA_HOME_RESOLVED}/bin/nvcc")"
  else
    CUDA_NVCC_VERSION=""
    if [[ "${DRY_RUN}" == "1" ]]; then
      warn "--cuda-home ${CUDA_HOME_RESOLVED} has no bin/nvcc yet (dry run: not checked)"
    else
      die "--cuda-home ${CUDA_HOME_RESOLVED} does not contain bin/nvcc. Point it at a CUDA toolkit root that provides nvcc."
    fi
  fi
else
  if [[ -n "${CUDA_HOME_RESOLVED}" && -x "${CUDA_HOME_RESOLVED}/bin/nvcc" ]]; then
    CUDA_NVCC_VERSION="$(detect_nvcc_version "${CUDA_HOME_RESOLVED}/bin/nvcc")"
  else
    CUDA_NVCC_VERSION="$(detect_nvcc_version)"
  fi
fi

PROFILE="$(resolve_profile "${PROFILE}" "${CUDA_NVCC_VERSION}")"
apply_profile "${PROFILE}"

# Gate the toolkit before any environment creation or multi-GB download.
if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -n "${CUDA_HOME_RESOLVED}" ]] || die "no CUDA toolkit found. Install a CUDA ${EXPECTED_CUDA}+ toolkit matching torch ${TORCH_VERSION}+${PROFILE}, or pass --cuda-home (this installer never runs apt/sudo)."
  [[ -x "${CUDA_HOME_RESOLVED}/bin/nvcc" ]] || die "no nvcc under ${CUDA_HOME_RESOLVED}; pass --cuda-home pointing at the matching CUDA toolkit"
  if [[ "${CUDA_NVCC_VERSION%%.*}" != "${EXPECTED_CUDA%%.*}" ]]; then
    die "profile ${PROFILE} needs a CUDA ${EXPECTED_CUDA%%.*}.x toolkit matching its PyTorch wheel, but the selected nvcc reports ${CUDA_NVCC_VERSION:-unknown}. Pass --cuda-home for a compatible CUDA ${EXPECTED_CUDA} toolkit."
  fi
  if ! version_ge "${CUDA_NVCC_VERSION}" "${EXPECTED_CUDA}"; then
    die "profile ${PROFILE} needs CUDA >= ${EXPECTED_CUDA} (torch ${TORCH_VERSION}+${PROFILE}), but the selected nvcc reports ${CUDA_NVCC_VERSION:-unknown}. Install the matching toolkit, pass --cuda-home, or rerun with --profile cu121."
  fi
fi

REQUIRED_TOOLS=(git tar)
if [[ -z "${LOCAL_OIDN_ARCHIVE}" ]]; then
  REQUIRED_TOOLS+=(curl)
fi
REQUIRED_TOOLS+=(cmake ninja)
MISSING_TOOLS=()
for tool in "${REQUIRED_TOOLS[@]}"; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    MISSING_TOOLS+=("${tool}")
  fi
done
CXX_COMPILER=""
for candidate in g++ c++ gcc cc; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    CXX_COMPILER="$(command -v "${candidate}")"
    break
  fi
done
if [[ -z "${CXX_COMPILER}" ]]; then
  MISSING_TOOLS+=("g++/c++ (C++ compiler)")
fi
if [[ "${#MISSING_TOOLS[@]}" -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    warn "missing build tool(s): ${MISSING_TOOLS[*]} (dry run: not checked)"
  else
    die "missing required build tool(s): ${MISSING_TOOLS[*]}. Install them with your system/package manager (git, curl, tar, a C++ compiler, cmake and ninja); this installer never runs apt/sudo."
  fi
fi

log "RoboTwin simulator environment installer"
log "  repository root : ${PROJECT_ROOT}"
log "  host            : ${HOST_OS}/${HOST_ARCH}"
log "  env prefix      : ${PREFIX}"
if [[ "${MANAGED_ROBOTWIN_ROOT}" == "1" ]]; then
  log "  RoboTwin root   : ${ROBOTWIN_ROOT} (pinned clone at ${ROBOTWIN_REF})"
else
  log "  RoboTwin root   : ${ROBOTWIN_ROOT} (user-supplied; used as-is)"
fi
log "  cuRobo checkout : ${CUROBO_DIR} (${CUROBO_REF})"
log "  profile         : ${PROFILE} (python ${PYTHON_VERSION}, torch ${TORCH_VERSION}+${PROFILE})"
log "  torch index     : ${TORCH_INDEX_URL}"
log "  requirements    : ${REQUIREMENTS_FILE}"
log "  CUDA toolkit    : ${CUDA_HOME_RESOLVED:-<not found>} (nvcc ${CUDA_NVCC_VERSION:-unknown})"
if [[ -n "${CONDA_BIN_RESOLVED}" ]]; then
  log "  conda           : ${CONDA_BIN_RESOLVED}"
else
  log "  conda           : not found in PATH (dry run only)"
fi
log "  dry run         : ${DRY_RUN}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi is not available; GPU/CUDA verification will be limited"
fi

# 1. RoboTwin checkout ------------------------------------------------------
if [[ "${MANAGED_ROBOTWIN_ROOT}" == "1" ]]; then
  if [[ "${DRY_RUN}" == "1" && ! -d "${ROBOTWIN_ROOT}/.git" ]]; then
    log ""
    log "Dry run: the RoboTwin checkout would be created at ${ROBOTWIN_ROOT}"
    printf '+ git clone --filter=blob:none --no-checkout %s %s\n' \
      "${ROBOTWIN_REPO}" "${ROBOTWIN_ROOT}"
    printf '+ git -C %s fetch --tags origin %s\n' "${ROBOTWIN_ROOT}" "${ROBOTWIN_REF}"
    printf '+ git -C %s checkout --detach FETCH_HEAD\n' "${ROBOTWIN_ROOT}"
  else
    checkout_repo "RoboTwin" "${ROBOTWIN_REPO}" "${ROBOTWIN_REF}" "${ROBOTWIN_ROOT}"
  fi
elif [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    warn "--robotwin-root does not exist yet: ${ROBOTWIN_ROOT} (dry run: not checked)"
  else
    die "--robotwin-root does not exist: ${ROBOTWIN_ROOT}"
  fi
fi

if [[ "${DRY_RUN}" == "0" && ! -f "${ROBOTWIN_ROOT}/envs/_GLOBAL_CONFIGS.py" ]]; then
  warn "${ROBOTWIN_ROOT} does not look like a RoboTwin 2.0 checkout (envs/_GLOBAL_CONFIGS.py missing)"
fi

# 2. Conda environment ------------------------------------------------------
if [[ -x "${ENV_PYTHON}" ]]; then
  log ""
  log "Reusing existing environment: ${PREFIX}"
  if [[ "${DRY_RUN}" == "0" ]]; then
    existing_python="$("${ENV_PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [[ "${existing_python}" != "${PYTHON_MAJOR_MINOR}" ]]; then
      die "existing environment ${PREFIX} has Python ${existing_python:-unknown}, but profile ${PROFILE} needs Python ${PYTHON_MAJOR_MINOR}. Remove/rename that environment or pass --prefix pointing at a fresh path; this installer never deletes an environment."
    fi
  fi
else
  log ""
  log "Creating Conda environment: ${PREFIX}"
  run_with_timeout "${CONDA_INSTALL_TIMEOUT}" "${CONDA_BIN}" create -y -p "${PREFIX}" \
    "python=${PYTHON_VERSION}" pip
fi

if [[ "${DRY_RUN}" == "0" && ! -x "${ENV_PYTHON}" ]]; then
  die "expected interpreter ${ENV_PYTHON} was not created; check the Conda output above"
fi

# 3. Torch + pinned simulator requirements ----------------------------------
log ""
log "Installing PyTorch ${TORCH_VERSION}+${PROFILE}"
pip_install --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

log ""
log "Installing pinned simulator requirements"
pip_install -r "${REQUIREMENTS_FILE}"
pip_install "${SETUPTOOLS_PIN}"

# 4. SAPIEN / mplib source fixes --------------------------------------------
log ""
log "Applying RoboTwin SAPIEN/mplib source fixes"
run "${ENV_PYTHON}" "${SCRIPT_DIR}/patch_robotwin_runtime.py" patch --prefix "${PREFIX}"

# 5. Open Image Denoise 2.3.3 ----------------------------------------------
log ""
log "Installing Open Image Denoise 2.3.3 into SAPIEN"
OIDN_ARCHIVE="$(download_oidn_archive)"
run "${ENV_PYTHON}" "${SCRIPT_DIR}/patch_sapien_oidn.py" patch \
  --prefix "${PREFIX}" --archive "${OIDN_ARCHIVE}" \
  --archive-sha256 "${OIDN_SHA256}"

# 6. cuRobo CUDA extensions -------------------------------------------------
log ""
checkout_repo "cuRobo" "${CUROBO_REPO}" "${CUROBO_REF}" "${CUROBO_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "Dry run: cuRobo would build with CUDA_HOME=${CUDA_HOME_RESOLVED:-<unresolved>}"
else
  log "cuRobo build: CUDA_HOME=${CUDA_HOME_RESOLVED} nvcc=${CUDA_NVCC_VERSION}"
fi

if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  log "cuRobo build: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (from environment)"
elif command -v nvidia-smi >/dev/null 2>&1; then
  detected_caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | awk 'NF' | sort -u || true)"
  if [[ -n "${detected_caps}" ]]; then
    arch_list=""
    while IFS= read -r cap; do
      [[ -n "${cap}" ]] || continue
      arch_list="${arch_list:+${arch_list};}${cap}"
    done <<< "${detected_caps}"
    TORCH_CUDA_ARCH_LIST="${arch_list}+PTX"
    log "cuRobo build: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (detected GPUs)"
  else
    warn "could not detect GPU compute capabilities; cuRobo will use the default arch list (slower build)"
  fi
else
  warn "nvidia-smi unavailable; cuRobo will use the default arch list (slower build)"
fi

curobo_target=(.)
if [[ "${PATCHWAM_CUROBO_EDITABLE:-0}" == "1" ]]; then
  curobo_target=(-e .)
  log "cuRobo install mode: editable"
else
  log "cuRobo install mode: non-editable (portable site-packages build)"
fi

build_env=(env
  "CUDA_HOME=${CUDA_HOME_RESOLVED}"
  "MAX_JOBS=${CUROBO_BUILD_JOBS}"
  "CMAKE_BUILD_PARALLEL_LEVEL=${CUROBO_BUILD_JOBS}")
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  build_env+=("TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")
fi
run_in_dir "${CUROBO_DIR}" "${build_env[@]}" "${ENV_PYTHON}" -m pip install \
  "${curobo_target[@]}" --no-build-isolation --no-deps

# 7. Exports for the evaluation config --------------------------------------
log ""
log "Writing environment exports"
ACTIVATE_DIR="${PREFIX}/etc/conda/activate.d"
EXPORT_FILE="${PREFIX}/patchwam_robotwin_env.sh"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '+ write %s (PATCHWAM_SIM_PYTHON, PATCHWAM_ROBOTWIN_ROOT)\n' "${EXPORT_FILE}"
  printf '+ write %s/patchwam-robotwin.sh (PATCHWAM_SIM_PYTHON, PATCHWAM_ROBOTWIN_ROOT)\n' "${ACTIVATE_DIR}"
else
  mkdir -p "${ACTIVATE_DIR}"
  {
    printf '# Source this file before running the PatchWAM RoboTwin evaluation, or\n'
    printf '# activate the %s Conda environment.\n' "${ENV_NAME}"
    printf 'export PATCHWAM_SIM_PYTHON=%q\n' "${PREFIX}/bin/python"
    printf 'export PATCHWAM_ROBOTWIN_ROOT=%q\n' "${ROBOTWIN_ROOT}"
  } > "${EXPORT_FILE}"
  {
    printf 'export PATCHWAM_SIM_PYTHON=%q\n' "${PREFIX}/bin/python"
    printf 'export PATCHWAM_ROBOTWIN_ROOT=%q\n' "${ROBOTWIN_ROOT}"
  } > "${ACTIVATE_DIR}/patchwam-robotwin.sh"
  log "  ${EXPORT_FILE}"
  log "  ${ACTIVATE_DIR}/patchwam-robotwin.sh"
fi

# 8. Verification -----------------------------------------------------------
if [[ "${SKIP_VERIFY}" == "1" ]]; then
  log ""
  log "Skipping post-install verification (--skip-verify)"
else
  log ""
  log "Verifying the RoboTwin environment"
  run "${ENV_PYTHON}" "${SCRIPT_DIR}/patch_sapien_oidn.py" verify --prefix "${PREFIX}"
  run "${ENV_PYTHON}" "${SCRIPT_DIR}/patch_robotwin_runtime.py" check --prefix "${PREFIX}"
  run "${ENV_PYTHON}" "${SCRIPT_DIR}/robotwin_env_checks.py" \
    --prefix "${PREFIX}" --profile "${PROFILE}" \
    --robotwin-root "${ROBOTWIN_ROOT}" \
    --requirements-manifest "${REQUIREMENTS_FILE}"
fi

# 9. Optional upstream assets ----------------------------------------------
ASSETS_STATUS="not-requested"
if [[ "${WITH_ASSETS}" == "1" ]]; then
  log ""
  downloader="$(find_assets_downloader || true)"
  if [[ -z "${downloader}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      downloader="${ROBOTWIN_ROOT}/scripts/_download_assets.sh"
      ASSETS_STATUS="dry-run"
      log "Dry run: RoboTwin assets would be fetched with ${downloader}"
    else
      ASSETS_STATUS="missing-downloader"
      warn "no upstream asset downloader found under ${ROBOTWIN_ROOT} (expected scripts/_download_assets.sh or script/_download_assets.sh)"
    fi
  else
    if ! command -v unzip >/dev/null 2>&1; then
      warn "unzip is not on PATH; RoboTwin's asset downloader unpacks its archives with unzip"
    fi
    log "Downloading RoboTwin assets with ${downloader}"
    ASSETS_STATUS="downloaded"
    if ! run_in_dir "${ROBOTWIN_ROOT}" \
        env "PATH=${PREFIX}/bin:${PATH}" bash "${downloader}" < /dev/null; then
      ASSETS_STATUS="failed"
      warn "the upstream asset downloader failed; assets are incomplete"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
      ASSETS_STATUS="dry-run"
    fi
  fi

  if [[ "${DRY_RUN}" == "0" && "${ASSETS_STATUS}" == "downloaded" ]]; then
    missing_assets=()
    for asset_dir in assets/embodiments assets/background_texture assets/objects; do
      [[ -e "${ROBOTWIN_ROOT}/${asset_dir}" ]] || missing_assets+=("${asset_dir}")
    done
    if [[ "${#missing_assets[@]}" -gt 0 ]]; then
      ASSETS_STATUS="failed"
      warn "asset path(s) still missing after the download: ${missing_assets[*]}"
    fi
  fi

  # RoboTwin rewrites its embodiment templates to absolute asset paths; the
  # upstream script prompts only when assets/embodiments is absent, so this is
  # non-interactive once the archives are unpacked. Re-running it is idempotent.
  if [[ "${DRY_RUN}" == "0" && "${ASSETS_STATUS}" == "downloaded" ]]; then
    config_script=""
    for candidate in \
        "${ROBOTWIN_ROOT}/script/update_embodiment_config_path.py" \
        "${ROBOTWIN_ROOT}/scripts/update_embodiment_config_path.py"; do
      if [[ -f "${candidate}" ]]; then
        config_script="${candidate}"
        break
      fi
    done
    if [[ -n "${config_script}" ]]; then
      log "Configuring embodiment asset paths with ${config_script}"
      if ! run_in_dir "${ROBOTWIN_ROOT}" "${ENV_PYTHON}" "${config_script}" < /dev/null; then
        ASSETS_STATUS="failed"
        warn "embodiment asset path configuration failed; run ${config_script} from ${ROBOTWIN_ROOT} manually"
      fi
    fi
  fi
fi

log ""
if [[ "${DRY_RUN}" == "1" ]]; then
  log "Dry run complete: no environment was created, downloaded or modified."
elif [[ "${ASSETS_STATUS}" == "failed" || "${ASSETS_STATUS}" == "missing-downloader" ]]; then
  log "Simulator environment installed at ${PREFIX}, but the requested RoboTwin assets are incomplete."
else
  log "RoboTwin simulator environment ready."
fi
print_exports
log ""
log "Next steps:"
log "  1. Policy environment and checkpoints stay separate: install FluxVLA in its own env (scripts/install_env.sh) and keep PATCHWAM_CHECKPOINT/PATCHWAM_ASSETS_ROOT pointing at the released assets."
log "  2. Run the evaluation with the FluxVLA interpreter, e.g.:"
log "     PATCHWAM_EVAL_PYTHON=<fluxvla env>/bin/python bash scripts/eval_patchwam_robotwin.sh"
log "  3. Simulator diagnostics: source ${EXPORT_FILE} or run ${PREFIX}/bin/python directly."
if [[ "${SKIP_VERIFY}" == "1" ]]; then
  warn "verification was skipped; rerun without --skip-verify before using this environment for evaluation"
fi
if [[ "${WITH_ASSETS}" == "1" && "${DRY_RUN}" == "0" \
    && "${ASSETS_STATUS}" != "downloaded" ]]; then
  printf 'Error: --with-assets did not complete (%s). The simulator environment itself remains installed at %s; rerun the upstream downloader from %s or install the assets manually, then rerun this installer.\n' \
    "${ASSETS_STATUS}" "${PREFIX}" "${ROBOTWIN_ROOT}" >&2
  exit 1
fi
