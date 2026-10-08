# Copyright 2026 Limx Dynamics
"""Targeted tests for scripts/install_robotwin_env.sh.

The installer is exercised with stub conda/git/nvidia-smi commands and a
synthetic Open Image Denoise archive. Nothing touches a real Conda
environment, the network, or a user's miniconda installation.
"""

import hashlib
import io
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / 'scripts' / 'install_robotwin_env.sh'
REAL_GIT = shutil.which('git') or '/usr/bin/git'
REAL_PYTHON = sys.executable

SAPIEN_LOADER = textwrap.dedent(
    '''\
    import ctypes
    import os

    if True:
        try:
            oidn_dll = ctypes.CDLL(
                os.path.join(
                    os.path.dirname(__file__),
                    "oidn_library/libOpenImageDenoise_core.so.2.0.1",
                ),
                ctypes.RTLD_LOCAL,
            )
            oidn_dll = ctypes.CDLL(
                os.path.join(
                    os.path.dirname(__file__),
                    "oidn_library/libOpenImageDenoise.so.2.0.1",
                ),
                ctypes.RTLD_LOCAL,
            )
        except Exception:
            pass
    '''
)

SAPIEN_URDF_LOADER = textwrap.dedent(
    '''\
    import os


    def load(urdf_file):
        with open(urdf_file, "r") as f:
            return f.read()
    '''
)

MPLIB_PLANNER = textwrap.dedent(
    '''\
    import numpy as np


    def screw_plan(delta_twist, collide, within_joint_limit):
        if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
            return {"status": "screw plan failed"}
        return {"status": "ok"}
    '''
)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_executable(path: Path, body: str) -> Path:
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + textwrap.dedent(body))
    path.chmod(0o755)
    return path


CONDA_STUB = '''
log="${STUB_LOG}"
if [ "${1:-}" = "info" ]; then
  echo "${STUB_CONDA_BASE:-/tmp/stub-conda-base}"
  exit 0
fi
echo "conda $*" >> "${log}"
if [ "${1:-}" = "create" ]; then
  if [ "${STUB_CONDA_FAIL:-0}" = "1" ]; then
    echo "stub conda create failed" >&2
    exit 1
  fi
  prefix=""
  while [ $# -gt 0 ]; do
    if [ "$1" = "-p" ]; then prefix="$2"; fi
    shift
  done
  if [ -z "${prefix}" ]; then
    echo "stub conda: create without -p" >&2
    exit 2
  fi
  mkdir -p "${prefix}/bin"
  mkdir -p "${prefix}/lib/python3.11/site-packages/sapien/wrapper"
  mkdir -p "${prefix}/lib/python3.11/site-packages/sapien/oidn_library"
  mkdir -p "${prefix}/lib/python3.11/site-packages/mplib"
  cp "${STUB_DIR}/fake_python" "${prefix}/bin/python"
  chmod +x "${prefix}/bin/python"
  cp "${STUB_DIR}/loader_template.py" "${prefix}/lib/python3.11/site-packages/sapien/_oidn_tricks.py"
  cp "${STUB_DIR}/urdf_loader_template.py" "${prefix}/lib/python3.11/site-packages/sapien/wrapper/urdf_loader.py"
  cp "${STUB_DIR}/planner_template.py" "${prefix}/lib/python3.11/site-packages/mplib/planner.py"
fi
exit 0
'''

GIT_STUB = '''
echo "git $*" >> "${STUB_LOG}"
if [ "${1:-}" = "-C" ] && [ "${3:-}" = "rev-parse" ]; then
  exec "${STUB_REAL_GIT}" "$@"
fi
if [ "${1:-}" = "-C" ] && [ "${3:-}" = "status" ]; then
  exec "${STUB_REAL_GIT}" "$@"
fi
if [ "${1:-}" = "clone" ]; then
  for dest in "$@"; do :; done
  mkdir -p "${dest}/.git"
fi
exit 0
'''

NVIDIA_SMI_STUB = '''
if [ "${1:-}" = "--query-gpu=compute_cap" ]; then
  echo "12.0"
  exit 0
fi
if [ "${1:-}" = "--query-gpu=name" ]; then
  echo "Stub GPU"
  exit 0
fi
echo "| NVIDIA-SMI 999.99  Driver Version: 999.99  CUDA Version: 13.2 |"
'''

CURL_STUB = '''
echo "curl $*" >> "${STUB_LOG}"
out=""
prev=""
for arg in "$@"; do
  if [ "${prev}" = "-o" ]; then out="${arg}"; fi
  prev="${arg}"
done
if [ -n "${STUB_CURL_SOURCE:-}" ] && [ -n "${out}" ]; then
  mkdir -p "$(dirname "${out}")"
  cp "${STUB_CURL_SOURCE}" "${out}"
fi
exit 0
'''

FAKE_PYTHON = '''
echo "PYTHON self=$0 cwd=$PWD args=$*" >> "${STUB_LOG}"
if [ "${1:-}" = "-c" ] && printf '%s' "${2:-}" | grep -q 'sys.version_info'; then
  echo "${STUB_PYTHON_VERSION:-3.11}"
  exit 0
fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pip" ]; then
  echo "PIP self=$0 cwd=$PWD args=$*" >> "${STUB_LOG}"
  exit 0
fi
exec "${STUB_REAL_PYTHON}" "$@"
'''

NVCC_STUB = '''
echo "Cuda compilation tools, release ${STUB_NVCC_VERSION:-12.8}, V${STUB_NVCC_VERSION:-12.8}.61"
'''

NOOP_TOOL_STUB = '''
exit 0
'''


@pytest.fixture()
def stub_env(tmp_path):
    """Stub command directory with fake conda/git/nvidia-smi/curl/nvcc."""
    stub_dir = tmp_path / 'stub-bin'
    stub_dir.mkdir()
    log = tmp_path / 'stub.log'
    log.write_text('')

    _write_executable(stub_dir / 'conda', CONDA_STUB)
    _write_executable(stub_dir / 'git', GIT_STUB)
    _write_executable(stub_dir / 'nvidia-smi', NVIDIA_SMI_STUB)
    _write_executable(stub_dir / 'curl', CURL_STUB)
    _write_executable(stub_dir / 'cmake', NOOP_TOOL_STUB)
    _write_executable(stub_dir / 'ninja', NOOP_TOOL_STUB)
    _write_executable(stub_dir / 'fake_python', FAKE_PYTHON)
    (stub_dir / 'loader_template.py').write_text(SAPIEN_LOADER)
    (stub_dir / 'urdf_loader_template.py').write_text(SAPIEN_URDF_LOADER)
    (stub_dir / 'planner_template.py').write_text(MPLIB_PLANNER)

    cuda_home = tmp_path / 'cuda'
    (cuda_home / 'bin').mkdir(parents=True)
    _write_executable(cuda_home / 'bin' / 'nvcc', NVCC_STUB)

    robotwin_root = tmp_path / 'RoboTwin'
    (robotwin_root / 'envs').mkdir(parents=True)
    (robotwin_root / 'envs' / '_GLOBAL_CONFIGS.py').write_text('ROOT_PATH = ""\n')
    (robotwin_root / 'task_config').mkdir()

    return {
        'stub_dir': stub_dir,
        'log': log,
        'cuda_home': cuda_home,
        'robotwin_root': robotwin_root,
        'tmp': tmp_path,
    }


def _installer_env(stub, tmp_path, **overrides):
    env = dict(os.environ)
    cuda_bin = stub['cuda_home'] / 'bin'
    env.update(
        {
            'PATH': (
                f"{stub['stub_dir']}:{cuda_bin}:"
                f"{os.environ.get('PATH', '/usr/bin:/bin')}"
            ),
            'HOME': str(tmp_path / 'home'),
            'STUB_LOG': str(stub['log']),
            'STUB_DIR': str(stub['stub_dir']),
            'STUB_REAL_PYTHON': REAL_PYTHON,
            'STUB_REAL_GIT': REAL_GIT,
            # Stub conda must win over an inherited CONDA_EXE from the test
            # process so the flow is fully mocked.
            'PATCHWAM_CONDA_BIN': str(stub['stub_dir'] / 'conda'),
            'CONDA_EXE': '',
            'PATCHWAM_CONDA_ENVS_DIR': str(tmp_path / 'conda-envs'),
            'PATCHWAM_CUROBO_SRC_ROOT': str(tmp_path / 'src'),
            'PATCHWAM_OIDN_CACHE': str(tmp_path / 'oidn-cache'),
        }
    )
    env.update(overrides)
    return env


def _run(args, env, cwd=None):
    return subprocess.run(
        ['bash', str(INSTALLER), *args],
        env=env,
        cwd=cwd or str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def _sourced_exports(path: Path):
    script = (
        'set -e\n'
        f'source {shlex.quote(str(path))}\n'
        'printf "%s\\n" "$PATCHWAM_SIM_PYTHON" "$PATCHWAM_ROBOTWIN_ROOT"\n'
    )
    result = subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 2, result.stdout
    return {'python': lines[0], 'root': lines[1]}


def _make_oidn_archive(tmp_path: Path) -> Path:
    archive = tmp_path / 'oidn-2.3.3.x86_64.linux.tar.gz'
    root = 'oidn-2.3.3.x86_64.linux'
    names = (
        'libOpenImageDenoise_core.so.2.3.3',
        'libOpenImageDenoise.so.2.3.3',
        'libOpenImageDenoise_device_cuda.so.2.3.3',
    )
    with tarfile.open(archive, 'w:gz') as handle:
        for name in names:
            payload = b'stub-octet-stream-' + name.encode()
            info = tarfile.TarInfo(f'{root}/lib/{name}')
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return archive


def test_script_syntax():
    result = subprocess.run(
        ['bash', '-n', str(INSTALLER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_help_lists_documented_options():
    result = subprocess.run(
        ['bash', str(INSTALLER), '--help'], capture_output=True, text=True
    )
    assert result.returncode == 0
    for token in (
        '--env-name',
        '--prefix',
        '--robotwin-root',
        '--profile',
        '--dry-run',
        '--with-assets',
        '--skip-verify',
        'PATCHWAM_SIM_PYTHON',
        'PATCHWAM_ROBOTWIN_ROOT',
    ):
        assert token in result.stdout


def test_unknown_argument_is_rejected():
    result = subprocess.run(
        ['bash', str(INSTALLER), '--nope'], capture_output=True, text=True
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'Unknown argument' in result.stderr


def test_dry_run_needs_no_conda_or_checkout_and_writes_nothing(tmp_path):
    home = tmp_path / 'home'
    prefix = tmp_path / 'env dir' / 'Robotwin'
    env = {
        'PATH': '/usr/bin:/bin',
        'HOME': str(home),
        'PATCHWAM_CONDA_ENVS_DIR': str(tmp_path / 'conda-envs'),
    }
    result = _run(
        [
            '--dry-run',
            '--prefix',
            str(prefix),
            '--robotwin-root',
            str(tmp_path / 'missing RoboTwin'),
        ],
        env,
    )
    assert result.returncode == 0, result.stderr
    assert 'no environment was created' in result.stdout
    assert str(prefix) in result.stdout
    assert 'python=3.11.15' in result.stdout
    assert not prefix.exists()
    assert not home.exists()
    assert not (tmp_path / 'conda-envs').exists()


def test_dry_run_reports_missing_conda_without_failing(tmp_path):
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path / 'home')}
    result = _run(['--dry-run', '--env-name', 'Robotwin'], env)
    assert result.returncode == 0, result.stderr
    assert 'conda           : not found in PATH (dry run only)' in result.stdout
    assert 'envs/Robotwin' in result.stdout


def test_real_run_without_conda_fails_with_actionable_message(tmp_path):
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path / 'home')}
    result = _run(['--env-name', 'Robotwin'], env)
    assert result.returncode != 0
    assert 'no Conda/mamba found' in result.stderr
    assert 'never runs apt/sudo' in result.stderr


def test_requested_low_profile_is_used_in_dry_run(tmp_path):
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path / 'home')}
    result = _run(
        ['--dry-run', '--profile', 'cu121', '--env-name', 'Robotwin'], env
    )
    assert result.returncode == 0, result.stderr
    # Python 3.11 for both profiles: the pinned closure requires >= 3.11.
    assert 'python=3.11.15' in result.stdout
    assert 'torch==2.4.1' in result.stdout
    assert 'https://download.pytorch.org/whl/cu121' in result.stdout


def test_dirty_managed_robotwin_checkout_is_refused(stub_env, tmp_path):
    src_root = tmp_path / 'managed-src'
    checkout = src_root / 'RoboTwin'
    (checkout / 'envs').mkdir(parents=True)
    (checkout / 'envs' / '_GLOBAL_CONFIGS.py').write_text('ROOT_PATH = ""\n')
    subprocess.run(['git', 'init', '-q'], cwd=checkout, check=True)
    subprocess.run(['git', 'add', '.'], cwd=checkout, check=True)
    subprocess.run(
        ['git', '-c', 'user.email=t@e.st', '-c', 'user.name=t', 'commit', '-qm', 'x'],
        cwd=checkout,
        check=True,
    )
    (checkout / 'envs' / '_GLOBAL_CONFIGS.py').write_text('ROOT_PATH = "changed"\n')

    env = _installer_env(
        stub_env, tmp_path, PATCHWAM_ROBOTWIN_SRC_ROOT=str(src_root)
    )
    result = _run(['--prefix', str(tmp_path / 'env')], env)
    assert result.returncode != 0
    assert 'has local changes' in result.stderr


def test_mocked_full_install_flow_and_rerun(stub_env, tmp_path):
    archive = _make_oidn_archive(tmp_path)
    prefix = tmp_path / 'env with space' / 'Robotwin'
    env = _installer_env(
        stub_env, tmp_path, PATCHWAM_OIDN_SHA256=_hash_file(archive)
    )
    args = [
        '--prefix',
        str(prefix),
        '--robotwin-root',
        str(stub_env['robotwin_root']),
        '--cuda-home',
        str(stub_env['cuda_home']),
        '--oidn-archive',
        str(archive),
        '--skip-verify',
    ]

    first = _run(args, env)
    assert first.returncode == 0, first.stderr + first.stdout
    log = stub_env['log'].read_text()

    assert f'conda create -y -p {prefix} python=3.11.15 pip' in log
    python_lines = [line for line in log.splitlines() if line.startswith('PYTHON ')]
    assert python_lines
    expected_python = f'{prefix}/bin/python'
    for line in python_lines:
        assert f'self={expected_python} ' in line, line

    pip_lines = [line for line in log.splitlines() if line.startswith('PIP ')]
    assert pip_lines
    assert any('torch==2.7.1' in line for line in pip_lines)
    assert any(
        str(REPO_ROOT / 'scripts' / 'requirements-robotwin-sim.txt') in line
        for line in pip_lines
    )
    curobo_dir = tmp_path / 'src' / 'curobo'
    assert any(
        f'cwd={curobo_dir}' in line
        and '--no-build-isolation' in line
        and '--no-deps' in line
        for line in pip_lines
    ), log

    site_packages = prefix / 'lib' / 'python3.11' / 'site-packages'
    library_dir = site_packages / 'sapien' / 'oidn_library'
    for name in (
        'libOpenImageDenoise.so.2.3.3',
        'libOpenImageDenoise_core.so.2.3.3',
        'libOpenImageDenoise_device_cuda.so.2.3.3',
    ):
        assert (library_dir / name).is_file()
    assert '2.3.3' in (site_packages / 'sapien' / '_oidn_tricks.py').read_text()
    loader_backup = site_packages / 'sapien' / '_oidn_tricks.py.patchwam-backup'
    assert '2.0.1' in loader_backup.read_text()

    urdf_loader = site_packages / 'sapien' / 'wrapper' / 'urdf_loader.py'
    assert 'open(urdf_file, "r", encoding="utf-8") as f:' in urdf_loader.read_text()
    planner = site_packages / 'mplib' / 'planner.py'
    planner_text = planner.read_text()
    assert 'or not within_joint_limit:' in planner_text
    assert 'or collide' not in planner_text
    assert (site_packages / 'mplib' / 'planner.py.patchwam-backup').is_file()

    export_file = prefix / 'patchwam_robotwin_env.sh'
    activate = prefix / 'etc' / 'conda' / 'activate.d' / 'patchwam-robotwin.sh'
    for path in (export_file, activate):
        exported = _sourced_exports(path)
        assert exported['python'] == expected_python
        assert exported['root'] == str(stub_env['robotwin_root'])

    second = _run(args, env)
    assert second.returncode == 0, second.stderr + second.stdout
    log_after = stub_env['log'].read_text()
    assert log_after.count('conda create') == 1
    assert 'already patched' in second.stdout
    assert urdf_loader.read_text().count('encoding="utf-8"') == 1
    assert list((site_packages / 'sapien').glob('*.patchwam-backup')) == [
        loader_backup
    ]


def test_conda_create_failure_is_reported(stub_env, tmp_path):
    env = _installer_env(stub_env, tmp_path, STUB_CONDA_FAIL='1')
    result = _run(
        [
            '--prefix',
            str(tmp_path / 'env'),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode != 0
    assert 'stub conda create failed' in result.stderr
    assert not (tmp_path / 'env' / 'patchwam_robotwin_env.sh').exists()


def test_mismatched_cuda_toolkit_is_reported(stub_env, tmp_path):
    archive = _make_oidn_archive(tmp_path)
    env = _installer_env(
        stub_env,
        tmp_path,
        STUB_NVCC_VERSION='12.4',
        PATCHWAM_OIDN_SHA256=_hash_file(archive),
    )
    result = _run(
        [
            '--prefix',
            str(tmp_path / 'env'),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--cuda-home',
            str(stub_env['cuda_home']),
            '--oidn-archive',
            str(archive),
            '--profile',
            'cu128',
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'needs CUDA >= 12.8' in result.stderr
    assert 'nvcc reports 12.4' in result.stderr


def test_with_assets_dry_run_uses_upstream_downloader(stub_env, tmp_path):
    downloader = stub_env['robotwin_root'] / 'scripts' / '_download_assets.sh'
    downloader.parent.mkdir()
    downloader.write_text('echo upstream\n')
    prefix = tmp_path / 'env'
    env = _installer_env(stub_env, tmp_path)
    result = _run(
        [
            '--dry-run',
            '--prefix',
            str(prefix),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--with-assets',
        ],
        env,
    )
    assert result.returncode == 0, result.stderr
    assert str(downloader) in result.stdout
    assert not prefix.exists()


def _assets_args(stub_env, tmp_path, archive):
    return [
        '--prefix',
        str(tmp_path / 'env'),
        '--robotwin-root',
        str(stub_env['robotwin_root']),
        '--cuda-home',
        str(stub_env['cuda_home']),
        '--oidn-archive',
        str(archive),
        '--skip-verify',
        '--with-assets',
    ]


def test_with_assets_success_runs_downloader_and_config(stub_env, tmp_path):
    archive = _make_oidn_archive(tmp_path)
    root = stub_env['robotwin_root']
    (root / 'scripts').mkdir()
    (root / 'scripts' / '_download_assets.sh').write_text(
        'mkdir -p assets/embodiments assets/background_texture assets/objects\n'
        'echo downloaded\n'
    )
    (root / 'script').mkdir()
    (root / 'script' / 'update_embodiment_config_path.py').write_text(
        'print("configured")\n'
    )
    env = _installer_env(
        stub_env, tmp_path, PATCHWAM_OIDN_SHA256=_hash_file(archive)
    )
    result = _run(_assets_args(stub_env, tmp_path, archive), env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RoboTwin simulator environment ready.' in result.stdout
    log = stub_env['log'].read_text()
    config_lines = [
        line for line in log.splitlines()
        if 'update_embodiment_config_path.py' in line
    ]
    assert config_lines
    assert f'cwd={root}' in config_lines[0]


def test_with_assets_missing_downloader_returns_nonzero(stub_env, tmp_path):
    archive = _make_oidn_archive(tmp_path)
    env = _installer_env(
        stub_env, tmp_path, PATCHWAM_OIDN_SHA256=_hash_file(archive)
    )
    result = _run(_assets_args(stub_env, tmp_path, archive), env)
    assert result.returncode != 0
    assert 'no upstream asset downloader found' in result.stderr
    assert 'assets are incomplete' in result.stdout
    assert '--with-assets did not complete (missing-downloader)' in result.stderr
    # The environment itself is still installed and usable.
    assert (tmp_path / 'env' / 'bin' / 'python').exists()


def test_with_assets_downloader_failure_returns_nonzero(stub_env, tmp_path):
    archive = _make_oidn_archive(tmp_path)
    root = stub_env['robotwin_root']
    (root / 'scripts').mkdir()
    (root / 'scripts' / '_download_assets.sh').write_text(
        'echo "hf download failed" >&2\nexit 1\n'
    )
    env = _installer_env(
        stub_env, tmp_path, PATCHWAM_OIDN_SHA256=_hash_file(archive)
    )
    result = _run(_assets_args(stub_env, tmp_path, archive), env)
    assert result.returncode != 0
    assert 'upstream asset downloader failed' in result.stderr
    assert '--with-assets did not complete (failed)' in result.stderr


def test_non_linux_x86_64_host_is_refused(stub_env, tmp_path):
    fake_uname = stub_env['stub_dir'] / 'uname'
    _write_executable(
        fake_uname,
        '''
        if [ "${1:-}" = "-s" ]; then
          echo "Darwin"
        else
          echo "arm64"
        fi
        ''',
    )
    env = _installer_env(stub_env, tmp_path)
    env['PATH'] = f"{stub_env['stub_dir']}:{env['PATH']}"
    result = _run(
        [
            '--prefix',
            str(tmp_path / 'env'),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode != 0
    assert 'unsupported host Darwin/arm64' in result.stderr


def test_fresh_uncached_oidn_download_returns_only_path(stub_env, tmp_path):
    """The curl path must not leak command text into the captured archive path."""
    archive = _make_oidn_archive(tmp_path)
    prefix = tmp_path / 'env'
    env = _installer_env(
        stub_env,
        tmp_path,
        STUB_CURL_SOURCE=str(archive),
        PATCHWAM_OIDN_SHA256=_hash_file(archive),
    )
    result = _run(
        [
            '--prefix',
            str(prefix),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--cuda-home',
            str(stub_env['cuda_home']),
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = stub_env['log'].read_text()
    assert 'curl' in log
    cached = tmp_path / 'oidn-cache' / archive.name
    assert cached.is_file()
    assert _hash_file(cached) == _hash_file(archive)
    site_packages = prefix / 'lib' / 'python3.11' / 'site-packages'
    assert (site_packages / 'sapien' / 'oidn_library' /
            'libOpenImageDenoise.so.2.3.3').is_file()
    # The patch step must receive the bare path, not a '+ curl ...' fragment.
    patch_lines = [
        line for line in log.splitlines()
        if line.startswith('PYTHON ') and 'patch_sapien_oidn.py patch' in line
    ]
    assert patch_lines
    assert '--archive ' in patch_lines[0]
    assert 'curl' not in patch_lines[0]


def test_cuda_home_override_beats_path_nvcc(stub_env, tmp_path):
    """--cuda-home must drive both the profile and the cuRobo build."""
    path_nvcc = stub_env['stub_dir'] / 'nvcc'
    _write_executable(path_nvcc, NVCC_STUB)
    new_toolkit = tmp_path / 'cuda-new'
    (new_toolkit / 'bin').mkdir(parents=True)
    _write_executable(
        new_toolkit / 'bin' / 'nvcc',
        '''
        echo "Cuda compilation tools, release 12.8, V12.8.61"
        ''',
    )
    archive = _make_oidn_archive(tmp_path)

    env = _installer_env(
        stub_env,
        tmp_path,
        STUB_NVCC_VERSION='12.4',
        PATCHWAM_OIDN_SHA256=_hash_file(archive),
    )
    result = _run(
        [
            '--prefix',
            str(tmp_path / 'env'),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--cuda-home',
            str(new_toolkit),
            '--oidn-archive',
            str(archive),
            '--skip-verify',
        ],
        env,
    )
    # PATH nvcc reports 12.4 but the selected toolkit reports 12.8, so auto
    # must pick cu128 and the cuRobo build must succeed.
    assert result.returncode == 0, result.stdout + result.stderr
    log = stub_env['log'].read_text()
    assert any('torch==2.7.1' in line for line in log.splitlines())
    assert 'cuRobo build: CUDA_HOME=' + str(new_toolkit) in result.stdout
    assert 'nvcc=12.8' in result.stdout


def test_cuda_home_override_below_profile_is_reported(stub_env, tmp_path):
    """A mismatched override must fail even when PATH nvcc looks modern."""
    path_nvcc = stub_env['stub_dir'] / 'nvcc'
    _write_executable(path_nvcc, NVCC_STUB)
    old_toolkit = tmp_path / 'old-cuda'
    (old_toolkit / 'bin').mkdir(parents=True)
    _write_executable(
        old_toolkit / 'bin' / 'nvcc',
        '''
        echo "Cuda compilation tools, release 12.4, V12.4.61"
        ''',
    )
    archive = _make_oidn_archive(tmp_path)
    env = _installer_env(
        stub_env, tmp_path, PATCHWAM_OIDN_SHA256=_hash_file(archive)
    )
    result = _run(
        [
            '--prefix',
            str(tmp_path / 'env'),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--cuda-home',
            str(old_toolkit),
            '--oidn-archive',
            str(archive),
            '--profile',
            'cu128',
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode != 0
    assert 'the selected nvcc reports 12.4' in result.stderr


def test_missing_build_tools_are_reported(tmp_path):
    tools_dir = tmp_path / 'tools'
    tools_dir.mkdir()
    for name in ('conda', 'git', 'tar', 'curl', 'cmake', 'g++', 'uname'):
        _write_executable(
            tools_dir / name,
            'if [ "${1:-}" = "-s" ]; then echo Linux; else echo x86_64; fi\n',
        )
    env = {
        'PATH': f"{tools_dir}:/usr/bin:/bin",
        'HOME': str(tmp_path / 'home'),
        'PATCHWAM_CONDA_ENVS_DIR': str(tmp_path / 'conda-envs'),
    }
    result = _run(
        [
            '--prefix',
            str(tmp_path / 'env'),
            '--robotwin-root',
            str(tmp_path),
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode != 0
    assert 'missing required build tool(s)' in result.stderr
    assert 'ninja' in result.stderr


def test_existing_env_with_wrong_python_fails_before_pip(stub_env, tmp_path):
    prefix = tmp_path / 'env'
    (prefix / 'bin').mkdir(parents=True)
    _write_executable(
        prefix / 'bin' / 'python',
        'echo "3.10"\n',
    )
    env = _installer_env(stub_env, tmp_path)
    result = _run(
        [
            '--prefix',
            str(prefix),
            '--robotwin-root',
            str(stub_env['robotwin_root']),
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode != 0
    assert 'has Python 3.10' in result.stderr
    assert 'needs Python 3.11' in result.stderr
    assert 'PIP ' not in stub_env['log'].read_text()


def test_relative_paths_are_normalized_in_dry_run(tmp_path):
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path / 'home')}
    result = _run(
        [
            '--dry-run',
            '--prefix',
            'rel env/Robotwin',
            '--robotwin-root',
            'rel RoboTwin',
        ],
        env,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert str(tmp_path / 'rel env' / 'Robotwin') in result.stdout
    assert str(tmp_path / 'rel RoboTwin') in result.stdout
    assert not (tmp_path / 'rel env').exists()


def test_export_file_is_sourceable_for_hostile_paths(stub_env, tmp_path):
    tricky_root = tmp_path / "Robo'twin $weird"
    (tricky_root / 'envs').mkdir(parents=True)
    (tricky_root / 'envs' / '_GLOBAL_CONFIGS.py').write_text('ROOT_PATH = ""\n')
    (tricky_root / 'task_config').mkdir()
    prefix = tmp_path / "env with 'quote' $dollar"
    archive = _make_oidn_archive(tmp_path)
    env = _installer_env(
        stub_env, tmp_path, PATCHWAM_OIDN_SHA256=_hash_file(archive)
    )
    result = _run(
        [
            '--prefix',
            str(prefix),
            '--robotwin-root',
            str(tricky_root),
            '--cuda-home',
            str(stub_env['cuda_home']),
            '--oidn-archive',
            str(archive),
            '--skip-verify',
        ],
        env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for path in (
        prefix / 'patchwam_robotwin_env.sh',
        prefix / 'etc' / 'conda' / 'activate.d' / 'patchwam-robotwin.sh',
    ):
        exported = _sourced_exports(path)
        assert exported['root'] == str(tricky_root)
        assert exported['python'] == str(prefix / 'bin' / 'python')


def test_cuda_major_mismatch_fails_before_environment_creation(stub_env, tmp_path):
    archive = _make_oidn_archive(tmp_path)
    env = _installer_env(
        stub_env, tmp_path, STUB_NVCC_VERSION='13.0',
        PATCHWAM_OIDN_SHA256=_hash_file(archive),
    )
    prefix = tmp_path / 'env'
    result = _run(
        ['--prefix', str(prefix), '--robotwin-root',
         str(stub_env['robotwin_root']), '--cuda-home',
         str(stub_env['cuda_home']), '--oidn-archive', str(archive),
         '--profile', 'cu128', '--skip-verify'], env,
    )
    assert result.returncode != 0
    assert 'needs a CUDA 12.x toolkit' in result.stderr
    assert 'nvcc reports 13.0' in result.stderr
    assert not prefix.exists()
