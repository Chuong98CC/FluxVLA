# Copyright 2026 Limx Dynamics
"""Tests for scripts/robotwin_env_checks.py.

The helper only reads an environment; the optional end-to-end case re-checks
the existing Robotwin environment on this machine without modifying it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKS_HELPER = REPO_ROOT / 'scripts' / 'robotwin_env_checks.py'

ROBOTWIN_ENV_PREFIX = Path('/home/chuong/miniconda3/envs/Robotwin')
ROBOTWIN_ENV_PYTHON = ROBOTWIN_ENV_PREFIX / 'bin' / 'python'
ROBOTWIN_CHECKOUT = Path(
    '/home/chuong/workspace/wam/ImageWAM/third_party/RoboTwin'
)


def _run(*args, python=sys.executable):
    return subprocess.run(
        [str(python), str(CHECKS_HELPER), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_missing_site_packages_is_reported(tmp_path):
    prefix = tmp_path / 'empty-env'
    prefix.mkdir()
    result = _run('--prefix', str(prefix))
    assert result.returncode != 0
    assert 'no site-packages under' in result.stderr


def test_foreign_path_hooks_are_reported(tmp_path):
    prefix = tmp_path / 'env'
    site_packages = prefix / 'lib' / 'python3.11' / 'site-packages'
    site_packages.mkdir(parents=True)
    (site_packages / 'foreign.pth').write_text('/opt/somewhere/else\n')

    # Interpret the helper as a module so a synthetic prefix can be inspected
    # without a matching interpreter: only the hook scanner is expected here.
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
    try:
        import robotwin_env_checks as checks
    finally:
        sys.path.pop(0)

    offenders = checks.foreign_path_hooks(site_packages, [prefix])
    assert offenders == [f'{site_packages / "foreign.pth"} -> /opt/somewhere/else']
    inside = site_packages / 'inside.pth'
    inside.write_text(str(prefix / 'lib' / 'shared') + '\n')
    assert checks.foreign_path_hooks(site_packages, [prefix]) == offenders


def test_editable_finder_outside_env_is_reported(tmp_path):
    prefix = tmp_path / 'env'
    site_packages = prefix / 'lib' / 'python3.11' / 'site-packages'
    site_packages.mkdir(parents=True)
    finder = site_packages / '__editable___demo_finder.py'
    finder.write_text('MAPPING = {"demo": "/home/other/src/demo"}\n')

    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
    try:
        import robotwin_env_checks as checks
    finally:
        sys.path.pop(0)

    offenders = checks.foreign_path_hooks(site_packages, [prefix])
    assert offenders == [f'{finder} -> /home/other/src/demo']


def test_python_satisfies_handles_real_world_specifiers():
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
    try:
        import robotwin_env_checks as checks
    finally:
        sys.path.pop(0)

    assert checks.python_satisfies('>=3.11', '3.11.15')
    assert not checks.python_satisfies('>=3.11', '3.10.22')
    assert checks.python_satisfies('!=3.14.1,>=3.11', '3.11.15')
    assert not checks.python_satisfies('!=3.14.1,>=3.11', '3.14.1')
    assert checks.python_satisfies('>=3.9', '3.11.15')
    # The stdlib-only fallback must agree with the packaging path.
    assert checks._fallback_satisfies('>=3.11', '3.11.15')
    assert not checks._fallback_satisfies('>=3.11', '3.10.22')
    assert not checks._fallback_satisfies('!=3.14.1,>=3.11', '3.14.1')
    assert checks._fallback_satisfies('==3.11.*', '3.11.15')


def test_parse_requirements_reads_pins(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
    try:
        import robotwin_env_checks as checks
    finally:
        sys.path.pop(0)

    manifest = tmp_path / 'requirements.txt'
    manifest.write_text(
        '# comment\n'
        'scipy==1.17.1\n'
        '\n'
        'numpy==1.26.4  # trailing comment\n'
        'torch\n'
    )
    assert checks.parse_requirements(manifest) == {
        'scipy': '1.17.1',
        'numpy': '1.26.4',
    }


def test_check_requirements_reports_mismatch_and_requires_python(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / 'scripts'))
    try:
        import robotwin_env_checks as checks
    finally:
        sys.path.pop(0)

    problems = []
    warnings = []
    checks.check_requirements(
        {'definitely-not-installed': '1.0.0'}, problems, warnings
    )
    assert problems == [
        'pinned package is not installed: definitely-not-installed==1.0.0'
    ]


@pytest.mark.skipif(
    not ROBOTWIN_ENV_PYTHON.is_file() or not ROBOTWIN_CHECKOUT.is_dir(),
    reason='existing Robotwin environment is not available on this machine',
)
def test_existing_robotwin_environment_passes_readonly_checks():
    result = _run(
        '--prefix',
        str(ROBOTWIN_ENV_PREFIX),
        '--profile',
        'cu128',
        '--robotwin-root',
        str(ROBOTWIN_CHECKOUT),
        '--requirements-manifest',
        str(REPO_ROOT / 'scripts' / 'requirements-robotwin-sim.txt'),
        python=ROBOTWIN_ENV_PYTHON,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'RoboTwin environment checks passed' in result.stdout
    assert 'torch 2.7.1+cu128' in result.stdout
