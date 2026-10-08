#!/usr/bin/env python3
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
"""Install the official Open Image Denoise 2.3.3 libraries into SAPIEN.

SAPIEN 3.0.0b1 ships Open Image Denoise 2.0.1.  On newer NVIDIA GPUs
(`RoboTwin issue #52 <https://github.com/RoboTwin-Platform/RoboTwin/issues/52>`_)
that build cannot create a CUDA denoising device, so this helper copies the
official RenderKit 2.3.3 shared libraries into ``sapien/oidn_library`` and
rewrites the ``_oidn_tricks.py`` preload stub to point at them.

Safety properties:

* the original ``_oidn_tricks.py`` is copied to ``_oidn_tricks.py.patchwam-backup``
  before it is replaced (the backup is never overwritten afterwards);
* the existing 2.0.1 libraries stay in place, only 2.3.3 files are added;
* the loader is replaced with an atomic rename, so a crash cannot leave a
  half-written Python module in ``site-packages``;
* every write is skipped when the environment already matches, so reruns are
  idempotent.

Subcommands
-----------
``check-archive``
    Validate the downloaded RenderKit archive (SHA256 + expected members).
``patch``
    Copy the 2.3.3 libraries and switch the SAPIEN loader to them.
``verify``
    Validate the installed libraries and create + commit a CUDA denoising
    device through the documented Open Image Denoise C API.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

OIDN_VERSION = "2.3.3"
OIDN_PREVIOUS_VERSION = "2.0.1"
DEFAULT_ARCHIVE_SHA256 = (
    "3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f"
)
DEFAULT_ARCHIVE_URL = (
    "https://github.com/RenderKit/oidn/releases/download/"
    f"v{OIDN_VERSION}/oidn-{OIDN_VERSION}.x86_64.linux.tar.gz"
)

# Shared libraries SAPIEN preloads; the names carry the versioned SONAME that
# the 2.3.3 objects link against, so no symlinks are required.
LIBRARY_FILES: Sequence[str] = (
    f"libOpenImageDenoise_core.so.{OIDN_VERSION}",
    f"libOpenImageDenoise.so.{OIDN_VERSION}",
    f"libOpenImageDenoise_device_cuda.so.{OIDN_VERSION}",
)

# SHA256 of the three shared libraries inside the verified 2.3.3 archive.
# They are used to confirm an archive that matches DEFAULT_ARCHIVE_SHA256.
VERIFIED_LIBRARY_SHA256: Dict[str, str] = {
    f"libOpenImageDenoise.so.{OIDN_VERSION}": (
        "9ac9dcd18318cae879efb01c14cff72c2c4275428a15e5ab6e94c8ff8a43c4dc"
    ),
    f"libOpenImageDenoise_core.so.{OIDN_VERSION}": (
        "eadb2437d6ac4a88e5746980495ba1091f7d14745e3000b82a9382093d22f351"
    ),
    f"libOpenImageDenoise_device_cuda.so.{OIDN_VERSION}": (
        "12b3ff87cfa4b6a90481b757f5a56ee69380a45e3dcaeadf8dd22cb73d4f714a"
    ),
}

MANIFEST_NAME = "patchwam_oidn_manifest.json"
LOADER_BACKUP_SUFFIX = ".patchwam-backup"

# Open Image Denoise C API constants (include/OpenImageDenoise/oidn.h, v2.3.3).
OIDN_DEVICE_TYPE_CUDA = 3
OIDN_ERROR_NONE = 0

# CUDA device creation is the only way to prove the new libraries are usable.
FALLBACK_CUDA_MISSING = (
    "OIDN 2.3.3 CUDA device creation failed. Check that an NVIDIA driver "
    "(>= 520) and a CUDA-capable GPU are visible in this environment, or pass "
    "--allow-no-cuda to verify only the library/loader installation."
)


class OidnPatchError(RuntimeError):
    """Raised for actionable installation or verification failures."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_sapien_dir(prefix: Path) -> Path:
    """Return ``<prefix>/lib/python3.*/site-packages/sapien``.

    Conda environments also create short ``lib/python3.1`` style aliases, so
    candidates are de-duplicated by resolved path before anything is rejected.
    """
    unique: Dict[Path, Path] = {}
    expected = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for candidate in sorted(
        prefix.glob("lib/python3*/site-packages/sapien"),
        key=lambda path: (0 if expected in str(path) else 1, str(path)),
    ):
        unique.setdefault(candidate.resolve(), candidate)
    candidates = list(unique.values())
    if not candidates:
        raise OidnPatchError(
            f"no sapien package found under {prefix}/lib/python3*/site-packages. "
            "Install the simulator requirements before patching OIDN."
        )
    if len(candidates) > 1:
        preferred = [path for path in candidates if expected in str(path)]
        if len(preferred) == 1:
            return preferred[0]
        raise OidnPatchError(
            "multiple sapien installs found under "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def archive_members(archive: Path) -> Dict[str, bytes]:
    """Read the three OIDN shared libraries straight out of the tarball."""
    found: Dict[str, bytes] = {}
    try:
        handle = tarfile.open(archive, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise OidnPatchError(f"cannot read archive {archive}: {exc}") from exc
    with handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name not in LIBRARY_FILES or "/lib/" not in member.name:
                continue
            extracted = handle.extractfile(member)
            if extracted is None:  # pragma: no cover - regular files always read
                continue
            found[name] = extracted.read()
    missing = [name for name in LIBRARY_FILES if name not in found]
    if missing:
        raise OidnPatchError(
            f"archive {archive} does not contain the expected OIDN "
            f"{OIDN_VERSION} libraries: {', '.join(missing)}"
        )
    return found


def check_archive(archive: Path, expected_sha256: Optional[str]) -> Dict[str, str]:
    if not archive.is_file():
        raise OidnPatchError(f"OIDN archive not found: {archive}")
    actual = sha256_file(archive)
    if expected_sha256 and actual != expected_sha256:
        raise OidnPatchError(
            "OIDN archive checksum mismatch\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}\n"
            f"  file:     {archive}\n"
            "Re-download the archive or pass --archive-sha256 with the value "
            "published for your download."
        )
    members = archive_members(archive)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    if expected_sha256 == DEFAULT_ARCHIVE_SHA256:
        for name, expected in VERIFIED_LIBRARY_SHA256.items():
            if hashes.get(name) != expected:
                raise OidnPatchError(
                    f"{name} in {archive} does not match the verified "
                    f"{OIDN_VERSION} digest: expected {expected}, got {hashes.get(name)}"
                )
    return hashes


def loader_script(existing: str) -> str:
    """Rewrite the SAPIEN OIDN preload stub to the 2.3.3 library names."""
    if OIDN_VERSION in existing and OIDN_PREVIOUS_VERSION not in existing:
        return existing
    if OIDN_PREVIOUS_VERSION not in existing:
        raise OidnPatchError(
            "unrecognized SAPIEN _oidn_tricks.py layout: neither "
            f"{OIDN_PREVIOUS_VERSION} nor {OIDN_VERSION} library names were "
            "found. Patch the loader manually or reinstall a matching SAPIEN "
            "build."
        )
    return existing.replace(OIDN_PREVIOUS_VERSION, OIDN_VERSION)


def atomic_write_text(path: Path, text: str, mode: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_copy_bytes(path: Path, data: bytes, mode: int = 0o755) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def patch(prefix: Path, archive: Path, expected_sha256: Optional[str]) -> int:
    hashes = check_archive(archive, expected_sha256)
    members = archive_members(archive)
    sapien_dir = find_sapien_dir(prefix)
    library_dir = sapien_dir / "oidn_library"
    loader = sapien_dir / "_oidn_tricks.py"
    if not loader.is_file():
        raise OidnPatchError(
            f"{loader} is missing; this SAPIEN build does not use the OIDN "
            "preload stub, so the 2.3.3 fix cannot be applied."
        )

    copied: List[str] = []
    for name in LIBRARY_FILES:
        target = library_dir / name
        if target.is_file() and sha256_file(target) == hashes[name]:
            continue
        atomic_copy_bytes(target, members[name])
        copied.append(str(target))

    backup = Path(str(loader) + LOADER_BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(loader, backup)

    original = loader.read_text(encoding="utf-8")
    updated = loader_script(original)
    loader_updated = updated != original
    if loader_updated:
        atomic_write_text(loader, updated, mode=loader.stat().st_mode & 0o777)

    manifest = {
        "oidn_version": OIDN_VERSION,
        "previous_version": OIDN_PREVIOUS_VERSION,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "sapien_dir": str(sapien_dir),
        "loader": str(loader),
        "loader_backup": str(backup),
        "libraries": {name: hashes[name] for name in LIBRARY_FILES},
    }
    atomic_write_text(
        library_dir / MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n"
    )

    print(f"SAPIEN OIDN {OIDN_VERSION} patch applied to {sapien_dir}")
    for name in LIBRARY_FILES:
        state = "copied" if str(library_dir / name) in copied else "already present"
        print(f"  {name}: {state}")
    print(
        "  _oidn_tricks.py: "
        + ("updated" if loader_updated else "already points at 2.3.3")
        + f" (backup: {backup})"
    )
    print(f"  manifest: {library_dir / MANIFEST_NAME}")
    return 0


def _load_oidn(library_dir: Path):
    core = library_dir / LIBRARY_FILES[0]
    public = library_dir / LIBRARY_FILES[1]
    # Mirror SAPIEN's preload order and RTLD_LOCAL mode.
    core_handle = ctypes.CDLL(str(core), mode=ctypes.RTLD_LOCAL)
    public_handle = ctypes.CDLL(str(public), mode=ctypes.RTLD_LOCAL)

    public_handle.oidnGetNumPhysicalDevices.restype = ctypes.c_int
    public_handle.oidnIsCUDADeviceSupported.restype = ctypes.c_bool
    public_handle.oidnIsCUDADeviceSupported.argtypes = [ctypes.c_int]
    public_handle.oidnGetPhysicalDeviceString.restype = ctypes.c_char_p
    public_handle.oidnGetPhysicalDeviceString.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
    ]
    public_handle.oidnNewDevice.restype = ctypes.c_void_p
    public_handle.oidnNewDevice.argtypes = [ctypes.c_int]
    public_handle.oidnCommitDevice.restype = None
    public_handle.oidnCommitDevice.argtypes = [ctypes.c_void_p]
    public_handle.oidnReleaseDevice.restype = None
    public_handle.oidnReleaseDevice.argtypes = [ctypes.c_void_p]
    public_handle.oidnGetDeviceError.restype = ctypes.c_int
    public_handle.oidnGetDeviceError.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    return core_handle, public_handle


def verify(prefix: Path, allow_no_cuda: bool, require_device: bool = True) -> int:
    sapien_dir = find_sapien_dir(prefix)
    library_dir = sapien_dir / "oidn_library"
    loader = sapien_dir / "_oidn_tricks.py"

    manifest_path = library_dir / MANIFEST_NAME
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems: List[str] = []
    for name in LIBRARY_FILES:
        path = library_dir / name
        if not path.is_file():
            problems.append(f"missing library: {path}")
            continue
        digest = sha256_file(path)
        expected = None
        if manifest and name in manifest.get("libraries", {}):
            expected = manifest["libraries"][name]
        if expected is None:
            # Without a manifest (for example an environment patched before
            # this helper existed) still compare against the official 2.3.3
            # digests so "verified" means something for copied environments.
            expected = VERIFIED_LIBRARY_SHA256.get(name)
        if expected and digest != expected:
            problems.append(
                f"{path} has unexpected SHA256 {digest} "
                f"(expected official OIDN {OIDN_VERSION} digest {expected})"
            )

    loader_text = loader.read_text(encoding="utf-8") if loader.is_file() else ""
    if not loader.is_file():
        problems.append(f"missing loader: {loader}")
    elif f".so.{OIDN_VERSION}" not in loader_text:
        problems.append(f"{loader} does not preload OIDN {OIDN_VERSION}")

    if problems:
        raise OidnPatchError(
            "OIDN 2.3.3 verification failed:\n"
            + "\n".join(f"  - {item}" for item in problems)
        )

    _, public = _load_oidn(library_dir)
    device_count = public.oidnGetNumPhysicalDevices()
    if device_count < 1:
        raise OidnPatchError("OIDN reported no physical devices after patching.")

    name = public.oidnGetPhysicalDeviceString(0, b"name")
    device_name = name.decode("utf-8", "replace") if name else "unknown"

    if not require_device:
        print(
            f"OIDN {OIDN_VERSION} verified (libraries only): {library_dir} "
            f"physical_devices={device_count} device='{device_name}'"
        )
        return 0

    cuda_supported = bool(public.oidnIsCUDADeviceSupported(0))
    error_message = ctypes.c_char_p()
    if not cuda_supported:
        message = (
            f"OIDN {OIDN_VERSION} loaded from {library_dir} but reports no CUDA "
            "device; denoising would fall back to CPU."
        )
        if allow_no_cuda:
            print("Warning: " + message)
            print(f"OIDN {OIDN_VERSION} verified (libraries only): {library_dir}")
            return 0
        raise OidnPatchError(FALLBACK_CUDA_MISSING + "\n  " + message)

    device = public.oidnNewDevice(OIDN_DEVICE_TYPE_CUDA)
    if not device:
        code = public.oidnGetDeviceError(None, ctypes.byref(error_message))
        detail = error_message.value.decode("utf-8", "replace") if error_message.value else ""
        if allow_no_cuda:
            print(f"Warning: OIDN CUDA device creation failed ({code}: {detail})")
            print(f"OIDN {OIDN_VERSION} verified (libraries only): {library_dir}")
            return 0
        raise OidnPatchError(
            FALLBACK_CUDA_MISSING + f"\n  device creation error {code}: {detail}"
        )

    public.oidnCommitDevice(device)
    commit_error = public.oidnGetDeviceError(device, ctypes.byref(error_message))
    public.oidnReleaseDevice(device)
    if commit_error != OIDN_ERROR_NONE:
        detail = error_message.value.decode("utf-8", "replace") if error_message.value else ""
        raise OidnPatchError(
            f"OIDN {OIDN_VERSION} CUDA device commit failed with error "
            f"{commit_error}: {detail}"
        )

    print(
        f"OIDN {OIDN_VERSION} verified: libraries={library_dir} "
        f"physical_devices={device_count} cuda_device='{device_name}' "
        "commit_error=0"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install and verify the Open Image Denoise 2.3.3 SAPIEN fix."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-archive", help="validate a downloaded OIDN archive"
    )
    check.add_argument("--archive", required=True, type=Path)
    check.add_argument("--archive-sha256", default=None)

    patch_parser = subparsers.add_parser(
        "patch", help="copy the 2.3.3 libraries and update the SAPIEN loader"
    )
    patch_parser.add_argument("--prefix", required=True, type=Path)
    patch_parser.add_argument("--archive", required=True, type=Path)
    patch_parser.add_argument("--archive-sha256", default=None)

    verify_parser = subparsers.add_parser(
        "verify", help="validate libraries, loader and CUDA device creation"
    )
    verify_parser.add_argument("--prefix", required=True, type=Path)
    verify_parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="treat a missing CUDA device as a warning instead of an error",
    )
    verify_parser.add_argument(
        "--libraries-only",
        action="store_true",
        help="skip device creation entirely",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check-archive":
            hashes = check_archive(args.archive, args.archive_sha256)
            print(f"OIDN archive OK: {args.archive}")
            for name in LIBRARY_FILES:
                print(f"  {name}: {hashes[name]}")
            return 0
        if args.command == "patch":
            return patch(args.prefix, args.archive, args.archive_sha256)
        if args.command == "verify":
            return verify(
                args.prefix,
                allow_no_cuda=args.allow_no_cuda,
                require_device=not args.libraries_only,
            )
    except OidnPatchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 2  # pragma: no cover - argparse rejects unknown commands


if __name__ == "__main__":
    sys.exit(main())
