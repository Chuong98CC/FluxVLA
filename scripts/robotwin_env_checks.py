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
"""Validate a freshly installed RoboTwin simulator environment.

Run with the environment's own interpreter, e.g.

    /path/to/envs/Robotwin/bin/python scripts/robotwin_env_checks.py \
        --prefix /path/to/envs/Robotwin --profile cu128

The script never imports the RoboTwin project itself; it only proves that the
interpreter, the pinned simulator packages, the cuRobo CUDA extensions, the
PyTorch CUDA runtime and the ``site-packages`` layout are usable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Expected torch major.minor and CUDA runtime tag per installer profile.
PROFILES: Dict[str, Dict[str, object]] = {
    "cu128": {"torch": "2.7", "cuda": "12.8", "python": (3, 11)},
    "cu121": {"torch": "2.4", "cuda": "12.1", "python": (3, 11)},
}

CORE_IMPORTS: Sequence[Tuple[str, str]] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("sapien", "sapien"),
    ("mplib", "mplib"),
    ("curobo", "curobo"),
    ("warp", "warp-lang"),
    ("toppra", "toppra"),
    ("cv2", "opencv-python"),
    ("yaml", "PyYAML"),
    ("trimesh", "trimesh"),
    ("gymnasium", "gymnasium"),
    ("imageio", "imageio"),
    ("PIL", "Pillow"),
)

OPTIONAL_IMPORTS: Sequence[Tuple[str, str]] = (
    ("quaternion", "numpy-quaternion"),
    ("h5py", "h5py"),
    ("zarr", "zarr"),
    ("openai", "openai"),
    ("huggingface_hub", "huggingface_hub"),
    ("av", "av"),
    ("matplotlib", "matplotlib"),
    ("yourdfpy", "yourdfpy"),
)

CUROBO_EXTENSIONS: Sequence[str] = (
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
)

ABS_PATH_IN_FINDER = re.compile(r'["\'](/[^"\']+)["\']')


class CheckFailure(RuntimeError):
    """Raised when the environment is not usable."""


def site_packages(prefix: Path) -> Path:
    unique: Dict[Path, Path] = {}
    expected = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for candidate in sorted(
        prefix.glob("lib/python3*/site-packages"),
        key=lambda path: (0 if expected in str(path) else 1, str(path)),
    ):
        unique.setdefault(candidate.resolve(), candidate)
    candidates = list(unique.values())
    if not candidates:
        raise CheckFailure(f"no site-packages under {prefix}/lib/python3*")
    if len(candidates) > 1:
        preferred = [path for path in candidates if expected in str(path)]
        if len(preferred) == 1:
            return preferred[0]
        raise CheckFailure(
            "multiple site-packages directories found: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def check_interpreter(prefix: Path, problems: List[str]) -> None:
    expected = prefix.resolve()
    actual = Path(sys.prefix).resolve()
    if actual != expected:
        problems.append(
            f"interpreter prefix is {actual} but --prefix is {expected}; run the "
            "check with the environment's own bin/python"
        )


def check_python_version(profile: str, problems: List[str]) -> None:
    expected = PROFILES[profile]["python"]
    actual = sys.version_info[:2]
    if tuple(actual) != tuple(expected):
        problems.append(
            f"profile {profile} expects Python "
            f"{expected[0]}.{expected[1]}, interpreter is {actual[0]}.{actual[1]}"
        )


def check_imports(
    required: Sequence[Tuple[str, str]],
    optional: Sequence[Tuple[str, str]],
    problems: List[str],
    warnings: List[str],
) -> None:
    for module, package in required:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            problems.append(
                f"required import failed: {module} ({package}): "
                f"{type(exc).__name__}: {exc}"
            )
    for module, package in optional:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - optional, warn only
            warnings.append(
                f"optional import unavailable: {module} ({package}): "
                f"{type(exc).__name__}: {exc}"
            )


def check_curobo_extensions(problems: List[str]) -> None:
    for module in CUROBO_EXTENSIONS:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report any CUDA build failure
            problems.append(
                f"cuRobo CUDA extension import failed: {module}: "
                f"{type(exc).__name__}: {exc}"
            )


def check_torch(profile: str, allow_no_cuda: bool, problems: List[str],
                warnings: List[str]) -> List[str]:
    import torch

    lines = [f"torch {torch.__version__} (cuda {torch.version.cuda})"]
    expected = PROFILES[profile]
    torch_base = torch.__version__.split("+", 1)[0]
    if ".".join(torch_base.split(".")[:2]) != expected["torch"]:
        problems.append(
            f"profile {profile} expects torch {expected['torch']}.x, "
            f"installed {torch.__version__}"
        )
    if not (torch.version.cuda or "").startswith(str(expected["cuda"])):
        problems.append(
            f"profile {profile} expects a CUDA {expected['cuda']} torch wheel, "
            f"torch.version.cuda is {torch.version.cuda}"
        )

    if not torch.cuda.is_available():
        message = (
            "torch.cuda.is_available() is False; the simulator needs a working "
            "NVIDIA driver and visible GPU"
        )
        if allow_no_cuda:
            warnings.append(message)
            return lines
        problems.append(message)
        return lines

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    tensor = torch.rand(64, 64, device="cuda")
    product = tensor @ tensor.T
    torch.cuda.synchronize()
    lines.append(
        f"cuda device {device}: {name} sm_{capability[0]}{capability[1]}, "
        f"matmul finite={bool(torch.isfinite(product).all())}"
    )
    return lines


def check_robotwin_root(root: Path, problems: List[str]) -> None:
    if not root:
        return
    if not root.is_dir():
        problems.append(f"RoboTwin checkout not found: {root}")
        return
    missing = [
        name
        for name in ("envs/_GLOBAL_CONFIGS.py", "task_config")
        if not (root / name).exists()
    ]
    if missing:
        problems.append(
            f"{root} does not look like a RoboTwin checkout; missing {missing}"
        )


def foreign_path_hooks(
    site_packages_dir: Path, allowed_roots: Sequence[Path]
) -> List[str]:
    """Report site-packages hooks that point outside the environment."""
    allowed = [path.resolve() for path in allowed_roots if path]
    offenders: List[str] = []

    def is_allowed(target: Path) -> bool:
        resolved = target.resolve()
        return any(
            resolved == root or root in resolved.parents for root in allowed
        )

    def record(target: str, origin: Path) -> None:
        if not target.startswith("/"):
            return
        if is_allowed(Path(target)):
            return
        offenders.append(f"{origin} -> {target}")

    for pattern in ("*.pth", "*.egg-link"):
        for path in sorted(site_packages_dir.glob(pattern)):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("import "):
                    continue
                record(line, path)
    for path in sorted(site_packages_dir.glob("__editable__*_finder.py")):
        for match in ABS_PATH_IN_FINDER.findall(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            record(match, path)
    return offenders


def collect_versions() -> List[str]:
    lines = []
    for package in (
        "sapien",
        "mplib",
        "numpy",
        "warp-lang",
        "toppra",
        "nvidia-curobo",
        "opencv-python",
        "setuptools",
    ):
        try:
            lines.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            continue
    return lines


def parse_requirements(manifest_path: Path) -> Dict[str, str]:
    """Read ``name==version`` pins from a requirements manifest."""
    pins: Dict[str, str] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.strip()
        version = version.strip()
        if name and version:
            pins[name] = version
    return pins


def _version_tuple(text: str) -> Tuple[int, ...]:
    numbers = re.findall(r"\d+", text)
    return tuple(int(number) for number in numbers[:3]) or (0,)


def _fallback_satisfies(spec_text: str, version: str) -> bool:
    actual = _version_tuple(version)
    for raw_clause in spec_text.split(","):
        clause = raw_clause.strip()
        if not clause:
            continue
        for operator in (">=", "<=", "!=", "==", ">", "<"):
            if not clause.startswith(operator):
                continue
            value = clause[len(operator):].strip()
            if operator == "==":
                if value.endswith(".*"):
                    wanted = _version_tuple(value[:-2])
                    if actual[: len(wanted)] != wanted:
                        return False
                elif actual[: len(_version_tuple(value))] != _version_tuple(value):
                    return False
            elif operator == "!=":
                if actual[: len(_version_tuple(value))] == _version_tuple(value):
                    return False
            elif operator == ">=":
                if actual < _version_tuple(value):
                    return False
            elif operator == "<=":
                if actual > _version_tuple(value):
                    return False
            elif operator == ">":
                if actual <= _version_tuple(value):
                    return False
            elif operator == "<":
                if actual >= _version_tuple(value):
                    return False
            break
    return True


def python_satisfies(spec_text: str, version: str) -> bool:
    try:
        from packaging.specifiers import SpecifierSet
    except Exception:  # noqa: BLE001 - packaging is optional in slim envs
        return _fallback_satisfies(spec_text, version)
    try:
        return SpecifierSet(spec_text).contains(version, prereleases=True)
    except Exception:  # noqa: BLE001 - fall back for exotic specifiers
        return _fallback_satisfies(spec_text, version)


def check_requirements(pins: Dict[str, str], problems: List[str],
                       warnings: List[str]) -> None:
    """Validate every pin's installed version and Requires-Python contract."""
    interpreter = ".".join(str(part) for part in sys.version_info[:3])
    for name, expected_version in sorted(pins.items()):
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            problems.append(f"pinned package is not installed: {name}=={expected_version}")
            continue
        if distribution.version != expected_version:
            problems.append(
                f"pinned version mismatch: {name}=={expected_version}, "
                f"installed {distribution.version}"
            )
        requires_python = distribution.metadata.get("Requires-Python")
        if not requires_python:
            warnings.append(f"{name}=={expected_version} declares no Requires-Python")
            continue
        if not python_satisfies(requires_python, interpreter):
            problems.append(
                f"{name}=={expected_version} requires Python {requires_python}, "
                f"interpreter is {interpreter}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a RoboTwin simulator environment."
    )
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--robotwin-root", type=Path, default=None)
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="treat a missing/unusable GPU as a warning instead of an error",
    )
    parser.add_argument(
        "--skip-foreign-hook-check",
        action="store_true",
        help="do not scan site-packages for hooks pointing outside the env",
    )
    parser.add_argument(
        "--requirements-manifest",
        type=Path,
        default=None,
        help="verify every name==version pin, including Requires-Python",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    problems: List[str] = []
    warnings: List[str] = []

    check_interpreter(args.prefix, problems)
    if args.profile:
        check_python_version(args.profile, problems)
    check_robotwin_root(args.robotwin_root, problems)

    site_packages_dir = site_packages(args.prefix)
    check_imports(CORE_IMPORTS, OPTIONAL_IMPORTS, problems, warnings)
    check_curobo_extensions(problems)
    if args.requirements_manifest is not None:
        if not args.requirements_manifest.is_file():
            problems.append(
                f"requirements manifest not found: {args.requirements_manifest}"
            )
        else:
            check_requirements(
                parse_requirements(args.requirements_manifest), problems, warnings
            )
    if args.profile:
        try:
            for line in check_torch(
                args.profile, args.allow_no_cuda, problems, warnings
            ):
                print(line)
        except ImportError as exc:  # pragma: no cover - covered by import checks
            problems.append(f"torch import failed: {exc}")

    if not args.skip_foreign_hook_check:
        allowed_roots = [args.prefix]
        if args.robotwin_root:
            allowed_roots.append(args.robotwin_root)
        offenders = foreign_path_hooks(site_packages_dir, allowed_roots)
        if offenders:
            problems.append(
                "site-packages contains path hooks outside this environment: "
                + "; ".join(offenders)
            )

    print("Installed versions: " + ", ".join(collect_versions()))
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if problems:
        print("Environment checks failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"RoboTwin environment checks passed: {args.prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
