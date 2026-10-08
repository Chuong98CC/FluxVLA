# Copyright 2026 Limx Dynamics
"""Unit tests for the RoboTwin installer helper scripts.

The OIDN patch tests use synthetic archives and a synthetic environment layout.
One optional check reuses the real RenderKit archive when it is already present
on the machine, which exercises the copy/loader/CUDA-device path against real
2.3.3 libraries without touching any installed environment.
"""

import hashlib
import io
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OIDN_HELPER = REPO_ROOT / 'scripts' / 'patch_sapien_oidn.py'
RUNTIME_HELPER = REPO_ROOT / 'scripts' / 'patch_robotwin_runtime.py'
CHECKS_HELPER = REPO_ROOT / 'scripts' / 'robotwin_env_checks.py'

REAL_OIDN_ARCHIVE = Path(
    '/tmp/fluxvla-oidn-2.3.3/oidn-2.3.3.x86_64.linux.tar.gz'
)
REAL_OIDN_SHA256 = (
    '3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f'
)

LOADER_TEMPLATE = textwrap.dedent(
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

URDF_LOADER_TEMPLATE = textwrap.dedent(
    '''\
    import os


    def load(urdf_file):
        with open(urdf_file, "r") as f:
            return f.read()
    '''
)

PLANNER_TEMPLATE = textwrap.dedent(
    '''\
    import numpy as np


    def screw_plan(delta_twist, collide, within_joint_limit):
        if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
            return {"status": "screw plan failed"}
        return {"status": "ok"}
    '''
)


def _fake_env_prefix(tmp_path: Path, python_version: str = '3.11') -> Path:
    prefix = tmp_path / 'env'
    sapien = prefix / 'lib' / f'python{python_version}' / 'site-packages' / 'sapien'
    (sapien / 'wrapper').mkdir(parents=True)
    (sapien / 'oidn_library').mkdir()
    mplib = prefix / 'lib' / f'python{python_version}' / 'site-packages' / 'mplib'
    mplib.mkdir()
    (sapien / '_oidn_tricks.py').write_text(LOADER_TEMPLATE)
    (sapien / 'wrapper' / 'urdf_loader.py').write_text(URDF_LOADER_TEMPLATE)
    (mplib / 'planner.py').write_text(PLANNER_TEMPLATE)
    return prefix


def _make_oidn_archive(tmp_path: Path, name: str = 'oidn-2.3.3.x86_64.linux.tar.gz'):
    archive = tmp_path / name
    root = 'oidn-2.3.3.x86_64.linux'
    libraries = (
        'libOpenImageDenoise_core.so.2.3.3',
        'libOpenImageDenoise.so.2.3.3',
        'libOpenImageDenoise_device_cuda.so.2.3.3',
    )
    with tarfile.open(archive, 'w:gz') as handle:
        for library in libraries:
            payload = b'synthetic-library-' + library.encode()
            info = tarfile.TarInfo(f'{root}/lib/{library}')
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return archive


def _run(script: Path, *args, python: str = sys.executable):
    return subprocess.run(
        [python, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_runtime_patch_applies_and_is_idempotent(tmp_path):
    prefix = _fake_env_prefix(tmp_path)
    site_packages = prefix / 'lib' / 'python3.11' / 'site-packages'

    before = _run(RUNTIME_HELPER, 'check', '--prefix', str(prefix))
    assert before.returncode != 0
    assert 'unpatched' in before.stderr

    patched = _run(RUNTIME_HELPER, 'patch', '--prefix', str(prefix))
    assert patched.returncode == 0, patched.stderr
    urdf_loader = site_packages / 'sapien' / 'wrapper' / 'urdf_loader.py'
    assert 'open(urdf_file, "r", encoding="utf-8") as f:' in urdf_loader.read_text()
    planner_text = (site_packages / 'mplib' / 'planner.py').read_text()
    assert 'or not within_joint_limit:' in planner_text
    assert 'or collide' not in planner_text
    assert (urdf_loader.parent / 'urdf_loader.py.patchwam-backup').is_file()
    assert (site_packages / 'mplib' / 'planner.py.patchwam-backup').is_file()

    again = _run(RUNTIME_HELPER, 'patch', '--prefix', str(prefix))
    assert again.returncode == 0, again.stderr
    assert 'already patched' in again.stdout
    assert urdf_loader.read_text().count('encoding="utf-8"') == 1
    backup = urdf_loader.parent / 'urdf_loader.py.patchwam-backup'
    assert backup.read_text() == URDF_LOADER_TEMPLATE

    check = _run(RUNTIME_HELPER, 'check', '--prefix', str(prefix))
    assert check.returncode == 0, check.stderr


def test_runtime_patch_rejects_unknown_layout(tmp_path):
    prefix = _fake_env_prefix(tmp_path)
    site_packages = prefix / 'lib' / 'python3.11' / 'site-packages'
    (site_packages / 'mplib' / 'planner.py').write_text('def unrelated():\n    pass\n')
    result = _run(RUNTIME_HELPER, 'patch', '--prefix', str(prefix))
    assert result.returncode != 0
    assert 'unrecognized mplib planner layout' in result.stderr


def test_runtime_patch_requires_targets(tmp_path):
    prefix = tmp_path / 'empty'
    (prefix / 'lib' / 'python3.11' / 'site-packages').mkdir(parents=True)
    result = _run(RUNTIME_HELPER, 'patch', '--prefix', str(prefix))
    assert result.returncode != 0
    assert 'mplib planner not found' in result.stderr


def test_oidn_check_archive_rejects_wrong_hash(tmp_path):
    archive = _make_oidn_archive(tmp_path)
    ok = _run(OIDN_HELPER, 'check-archive', '--archive', str(archive))
    assert ok.returncode == 0, ok.stderr
    assert 'OIDN archive OK' in ok.stdout

    bad = _run(
        OIDN_HELPER,
        'check-archive',
        '--archive',
        str(archive),
        '--archive-sha256',
        '0' * 64,
    )
    assert bad.returncode != 0
    assert 'checksum mismatch' in bad.stderr


def test_oidn_patch_copies_libraries_and_backs_up_loader(tmp_path):
    archive = _make_oidn_archive(tmp_path)
    prefix = _fake_env_prefix(tmp_path)
    result = _run(
        OIDN_HELPER, 'patch', '--prefix', str(prefix), '--archive', str(archive)
    )
    assert result.returncode == 0, result.stderr

    sapien = prefix / 'lib' / 'python3.11' / 'site-packages' / 'sapien'
    library_dir = sapien / 'oidn_library'
    for name in (
        'libOpenImageDenoise.so.2.3.3',
        'libOpenImageDenoise_core.so.2.3.3',
        'libOpenImageDenoise_device_cuda.so.2.3.3',
    ):
        assert (library_dir / name).is_file()
    loader = (sapien / '_oidn_tricks.py').read_text()
    assert '2.3.3' in loader
    assert '2.0.1' not in loader
    backup = sapien / '_oidn_tricks.py.patchwam-backup'
    assert backup.read_text() == LOADER_TEMPLATE

    manifest = library_dir / 'patchwam_oidn_manifest.json'
    assert manifest.is_file()
    assert '2.3.3' in manifest.read_text()

    again = _run(
        OIDN_HELPER, 'patch', '--prefix', str(prefix), '--archive', str(archive)
    )
    assert again.returncode == 0, again.stderr
    assert 'already present' in again.stdout
    assert 'already points at 2.3.3' in again.stdout
    assert (sapien / '_oidn_tricks.py').read_text() == loader


def test_oidn_patch_rejects_unknown_loader(tmp_path):
    archive = _make_oidn_archive(tmp_path)
    prefix = _fake_env_prefix(tmp_path)
    sapien = prefix / 'lib' / 'python3.11' / 'site-packages' / 'sapien'
    (sapien / '_oidn_tricks.py').write_text('print("unrelated")\n')
    result = _run(
        OIDN_HELPER, 'patch', '--prefix', str(prefix), '--archive', str(archive)
    )
    assert result.returncode != 0
    assert 'unrecognized SAPIEN _oidn_tricks.py layout' in result.stderr


def test_oidn_patch_rejects_archive_without_libraries(tmp_path):
    archive = tmp_path / 'empty.tar.gz'
    with tarfile.open(archive, 'w:gz') as handle:
        info = tarfile.TarInfo('oidn-2.3.3.x86_64.linux/README')
        info.size = 4
        handle.addfile(info, io.BytesIO(b'none'))
    prefix = _fake_env_prefix(tmp_path)
    result = _run(
        OIDN_HELPER, 'patch', '--prefix', str(prefix), '--archive', str(archive)
    )
    assert result.returncode != 0
    assert 'does not contain the expected OIDN 2.3.3 libraries' in result.stderr


@pytest.mark.skipif(
    not REAL_OIDN_ARCHIVE.is_file(),
    reason='verified RenderKit 2.3.3 archive is not present on this machine',
)
def test_oidn_patch_and_verify_with_real_archive(tmp_path):
    """Exercise the real 2.3.3 libraries: copy, loader rewrite, CUDA device."""
    assert (
        hashlib.sha256(REAL_OIDN_ARCHIVE.read_bytes()).hexdigest()
        == REAL_OIDN_SHA256
    )
    prefix = _fake_env_prefix(tmp_path)
    patch = _run(
        OIDN_HELPER,
        'patch',
        '--prefix',
        str(prefix),
        '--archive',
        str(REAL_OIDN_ARCHIVE),
        '--archive-sha256',
        REAL_OIDN_SHA256,
    )
    assert patch.returncode == 0, patch.stderr
    verify = _run(
        OIDN_HELPER,
        'verify',
        '--prefix',
        str(prefix),
        '--allow-no-cuda',
    )
    assert verify.returncode == 0, verify.stderr
    assert 'OIDN 2.3.3 verified' in verify.stdout


@pytest.mark.skipif(
    not REAL_OIDN_ARCHIVE.is_file(),
    reason='verified RenderKit 2.3.3 archive is not present on this machine',
)
def test_oidn_verify_without_manifest_uses_known_digests(tmp_path):
    """Copied environments without a manifest are still checked by digest."""
    prefix = _fake_env_prefix(tmp_path)
    patch = _run(
        OIDN_HELPER,
        'patch',
        '--prefix',
        str(prefix),
        '--archive',
        str(REAL_OIDN_ARCHIVE),
        '--archive-sha256',
        REAL_OIDN_SHA256,
    )
    assert patch.returncode == 0, patch.stderr
    sapien = prefix / 'lib' / 'python3.11' / 'site-packages' / 'sapien'
    (sapien / 'oidn_library' / 'patchwam_oidn_manifest.json').unlink()
    assert not (sapien / 'oidn_library' / 'patchwam_oidn_manifest.json').exists()

    verify = _run(
        OIDN_HELPER, 'verify', '--prefix', str(prefix), '--allow-no-cuda'
    )
    assert verify.returncode == 0, verify.stderr
    assert 'OIDN 2.3.3 verified' in verify.stdout

    tampered = sapien / 'oidn_library' / 'libOpenImageDenoise.so.2.3.3'
    with tampered.open('ab') as handle:
        handle.write(b'tampered')
    broken = _run(
        OIDN_HELPER, 'verify', '--prefix', str(prefix), '--allow-no-cuda'
    )
    assert broken.returncode != 0
    assert 'unexpected SHA256' in broken.stderr
