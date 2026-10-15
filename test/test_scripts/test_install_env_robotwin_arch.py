"""RoboTwin cuRobo TORCH_CUDA_ARCH_LIST resolution in ``install_env.sh``.

An architecture-specific build for the wrong GPU fails at runtime with
"no kernel image is available for execution on the device", so the installer
must not keep a hardcoded default. These tests drive the real installer in
dry-run mode with a fake ``nvidia-smi`` and assert the value it would use.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / 'scripts' / 'install_env.sh'


@pytest.fixture
def installer_env(tmp_path):
    """PATH with a fake nvidia-smi ahead of the interpreter's directory."""
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake = fake_bin / 'nvidia-smi'
    fake.write_text(
        '#!/bin/sh\n'
        'case "$*" in\n'
        '  *compute_cap*) printf "%s\\n" "${FAKE_COMPUTE_CAP-12.0}" ;;\n'
        '  *query-gpu=name*) printf "%s\\n" "NVIDIA GeForce RTX 5090" ;;\n'
        '  *driver_version*) printf "%s\\n" "595.71.05" ;;\n'
        '  *) printf "%s\\n" "CUDA Version: 12.8" ;;\n'
        'esac\n')
    fake.chmod(0o755)
    env = {
        name: os.environ[name]
        for name in ('HOME', 'LANG', 'TMPDIR', 'CUDA_HOME', 'PATH')
        if name in os.environ
    }
    env['PYTHON'] = sys.executable
    env['PATH'] = os.pathsep.join(
        [str(fake_bin), os.path.dirname(sys.executable),
         env.get('PATH', '')])
    env.pop('FLUXVLA_ROBOTWIN_CUDA_ARCH_LIST', None)
    return env


def _resolved_arch(env):
    result = subprocess.run(
        ['bash', str(INSTALLER), '--dry-run', '--with-robotwin',
         '--profile', 'cu128'],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        timeout=900, check=False)
    for line in result.stdout.splitlines():
        if 'TORCH_CUDA_ARCH_LIST:' in line:
            return line.split('TORCH_CUDA_ARCH_LIST:', 1)[1].strip()
    return None


def test_arch_list_follows_the_visible_gpu(installer_env):
    resolved = _resolved_arch(installer_env)
    assert resolved is not None, 'installer never reported the arch list'
    assert resolved.startswith('12.0+PTX')
    assert 'explicit=none' in resolved


def test_explicit_arch_list_override_is_preserved(installer_env):
    env = dict(installer_env)
    env['FLUXVLA_ROBOTWIN_CUDA_ARCH_LIST'] = '8.0,12.0'
    resolved = _resolved_arch(env)
    assert resolved == '8.0,12.0 (explicit=8.0,12.0)'


def test_missing_gpu_signal_uses_the_reported_fallback(installer_env):
    env = dict(installer_env)
    env['FAKE_COMPUTE_CAP'] = ''
    resolved = _resolved_arch(env)
    assert resolved is not None
    assert resolved.startswith('8.0')
    assert 'explicit=none' in resolved


def test_multiple_gpu_architectures_use_pytorch_syntax(installer_env):
    env = dict(installer_env)
    env['FAKE_COMPUTE_CAP'] = '12.0\n8.0\n12.0'
    resolved = _resolved_arch(env)
    assert resolved == '8.0;12.0+PTX (explicit=none)'


def test_installer_stays_syntax_valid():
    assert shutil.which('bash') is not None
    result = subprocess.run(['bash', '-n', str(INSTALLER)], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stderr
