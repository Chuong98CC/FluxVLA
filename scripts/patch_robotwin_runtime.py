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
"""Apply RoboTwin's documented SAPIEN and mplib source fixes to an env prefix.

RoboTwin's upstream ``script/_install.sh`` edits two installed packages after
``pip install`` succeeds, and both edits are required by the verified
``Robotwin`` simulator environment:

* ``sapien/wrapper/urdf_loader.py`` opens URDF files without an explicit
  encoding (upstream: ``sed -i -E 's/("r")(\\))( as)/\\1, encoding="utf-8") as/g'``);
* ``mplib/planner.py`` drops ``or collide`` from the screw-plan failure check.

This helper performs the same edits without ``sed -i`` so a failure cannot
leave a truncated module: the original file is copied once to
``<module>.patchwam-backup`` and the patched content replaces the file with an
atomic rename. Files that are already patched are left untouched.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

SAPIEN_OPEN_PATTERN = re.compile(r'("r")(\))( as)')
SAPIEN_PATCHED_PATTERN = re.compile(r'open\([^\n]*"r", encoding="utf-8"\) as')

MPLIB_BROKEN = "np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
MPLIB_FIXED = "np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"

BACKUP_SUFFIX = ".patchwam-backup"


class RuntimePatchError(RuntimeError):
    """Raised when installed sources do not match a supported layout."""


def _site_packages(prefix: Path) -> Path:
    unique = {}
    expected = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for candidate in sorted(
        prefix.glob("lib/python3*/site-packages"),
        key=lambda path: (0 if expected in str(path) else 1, str(path)),
    ):
        unique.setdefault(candidate.resolve(), candidate)
    candidates = list(unique.values())
    if not candidates:
        raise RuntimePatchError(
            f"no site-packages directory found under {prefix}/lib/python3*"
        )
    if len(candidates) > 1:
        preferred = [path for path in candidates if expected in str(path)]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimePatchError(
            "multiple site-packages directories found: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def find_targets(prefix: Path) -> Tuple[Path, Path]:
    """Return the SAPIEN urdf_loader and mplib planner paths."""
    site_packages = _site_packages(prefix)
    sapien = site_packages / "sapien" / "wrapper" / "urdf_loader.py"
    planner = site_packages / "mplib" / "planner.py"
    if not planner.is_file():
        raise RuntimePatchError(
            f"mplib planner not found at {planner}; install mplib before "
            "applying the RoboTwin runtime fixes."
        )
    if not sapien.is_file():
        raise RuntimePatchError(
            f"SAPIEN urdf_loader not found at {sapien}; install sapien before "
            "applying the RoboTwin runtime fixes."
        )
    return sapien, planner


def _atomic_write(path: Path, text: str, mode: Optional[int]) -> None:
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


def _backup_once(path: Path) -> Optional[Path]:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        return None
    shutil.copy2(path, backup)
    return backup


def patch_sapien_urdf_loader(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if SAPIEN_OPEN_PATTERN.search(text):
        patched, count = SAPIEN_OPEN_PATTERN.subn(r'\1, encoding="utf-8")\3', text)
        _backup_once(path)
        _atomic_write(path, patched, path.stat().st_mode & 0o777)
        return f"patched {count} open() call(s)"
    if SAPIEN_PATCHED_PATTERN.search(text):
        return "already patched"
    raise RuntimePatchError(
        f"unrecognized SAPIEN urdf_loader layout in {path}: neither the "
        "upstream open(..., \"r\") pattern nor an encoding=\"utf-8\" patch was "
        "found. Apply RoboTwin's manual SAPIEN adjustment and rerun."
    )


def patch_mplib_planner(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MPLIB_BROKEN in text:
        patched = text.replace(MPLIB_BROKEN, MPLIB_FIXED)
        _backup_once(path)
        _atomic_write(path, patched, path.stat().st_mode & 0o777)
        return "patched screw-plan failure check"
    if MPLIB_FIXED in text:
        return "already patched"
    raise RuntimePatchError(
        f"unrecognized mplib planner layout in {path}: the RoboTwin screw-plan "
        "line was not found. Apply RoboTwin's manual mplib adjustment and rerun."
    )


def check(prefix: Path) -> List[str]:
    sapien, planner = find_targets(prefix)
    problems: List[str] = []
    sapien_text = sapien.read_text(encoding="utf-8")
    if SAPIEN_OPEN_PATTERN.search(sapien_text):
        problems.append(f"SAPIEN urdf_loader is unpatched: {sapien}")
    elif not SAPIEN_PATCHED_PATTERN.search(sapien_text):
        problems.append(f"SAPIEN urdf_loader layout is unrecognized: {sapien}")
    planner_text = planner.read_text(encoding="utf-8")
    if MPLIB_BROKEN in planner_text:
        problems.append(f"mplib planner is unpatched: {planner}")
    elif MPLIB_FIXED not in planner_text:
        problems.append(f"mplib planner layout is unrecognized: {planner}")
    return problems


def apply(prefix: Path) -> int:
    sapien, planner = find_targets(prefix)
    print(f"RoboTwin runtime fixes in {prefix}")
    print(f"  {sapien}: {patch_sapien_urdf_loader(sapien)}")
    print(f"  {planner}: {patch_mplib_planner(planner)}")
    backup = sapien.with_name(sapien.name + BACKUP_SUFFIX)
    if backup.exists():
        print(f"  backups: {backup}, {planner.with_name(planner.name + BACKUP_SUFFIX)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply RoboTwin's SAPIEN and mplib source fixes to an env prefix."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("patch", "apply the upstream source fixes"),
        ("check", "report whether the fixes are applied"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--prefix", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "patch":
            return apply(args.prefix)
        problems = check(args.prefix)
    except RuntimePatchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if problems:
        print("RoboTwin runtime fixes are incomplete:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"RoboTwin runtime fixes verified in {args.prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
