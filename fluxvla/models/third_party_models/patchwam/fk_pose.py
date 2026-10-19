"""ALOHA forward kinematics and the PatchWAM RoboTwin pose objective.

The module is import-safe without the optional ``pytorch_kinematics``
dependency: importing ``pytorch_kinematics`` happens only when a chain is first
used. FK statistics are plain JSON metadata so the dataset can attach episode
activity labels without loading robot assets.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

ALOHA_ACTION_DIM = 14
ALOHA_ARM_NAMES = ('left', 'right')
ALOHA_ARM_SLICES = (slice(0, 6), slice(7, 13))
ALOHA_GRIPPER_INDICES = (6, 13)
ALOHA_JOINT_NAMES = {
    'left': tuple(f'fl_joint{i}' for i in range(1, 7)),
    'right': tuple(f'fr_joint{i}' for i in range(1, 7)),
}
ALOHA_END_LINKS = {'left': 'fl_link6', 'right': 'fr_link6'}
ALOHA_ROOT_LINK = 'footprint'
ALOHA_TCP_OFFSET_M = 0.12
ALOHA_TCP_LOCAL_AXIS = (1.0, 0.0, 0.0)
ALOHA_ORIENTATION_CORRECTION = ((1.0, 0.0, 0.0),
                                (0.0, -1.0, 0.0),
                                (0.0, 0.0, -1.0))
FK_STATS_SCHEMA_VERSION = 1
FK_STATS_KIND = 'patchwam_aloha_fk_pose_stats'
DEFAULT_TRANSLATION_SPREAD_M = 0.01
DEFAULT_ORIENTATION_SPREAD = 0.05

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_URDF = (
    'src/RoboTwin/assets/embodiments/aloha-agilex/urdf/'
    'arx5_description_isaac.urdf')

__all__ = [
    'ALOHA_ACTION_DIM',
    'ALOHA_ARM_NAMES',
    'ALOHA_ARM_SLICES',
    'ALOHA_END_LINKS',
    'ALOHA_GRIPPER_INDICES',
    'ALOHA_JOINT_NAMES',
    'ALOHA_ORIENTATION_CORRECTION',
    'ALOHA_TCP_OFFSET_M',
    'DEFAULT_ORIENTATION_SPREAD',
    'DEFAULT_TRANSLATION_SPREAD_M',
    'AlohaFk',
    'EpisodePoseActivity',
    'FkPoseObjective',
    'FkPoseStatistics',
    'PoseBoundsAccumulator',
    'canonical_data_root',
    'compute_fk_pose_loss',
    'default_aloha_urdf_path',
    'episode_pose_activity',
    'file_sha256',
]


def canonical_data_root(root: Any) -> str:
    """Return the absolute real path used as the artifact root key."""
    if not isinstance(root, (str, os.PathLike)):
        raise TypeError(f'Data root must be a path, got {type(root)!r}')
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(root))))


def file_sha256(path: Any) -> str:
    """Hash a file without loading it as text."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def default_aloha_urdf_path(*, robotwin_root: Optional[str] = None) -> Path:
    """Resolve the installed RoboTwin ALOHA URDF."""
    configured = robotwin_root
    if configured is None:
        configured = os.environ.get('FLUXVLA_ROBOTWIN_ROOT')
    if configured:
        root = Path(configured).expanduser()
        return (root / 'assets/embodiments/aloha-agilex/urdf/'
                'arx5_description_isaac.urdf')
    return _REPO_ROOT / _DEFAULT_URDF


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a JSON object')
    return value


def _finite(value: Any, name: str, *, minimum: Optional[float] = None,
            positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f'{name} must be finite, got {value!r}')
    if positive and result <= 0:
        raise ValueError(f'{name} must be positive, got {result}')
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {result}')
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f'{name} must be a JSON boolean, got {value!r}')
    return value


def _integer(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer, got {value!r}')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {value}')
    return int(value)


def _vector(value: Any, length: int, name: str) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} must contain {length} numbers')
    return tuple(_finite(item, f'{name}[{index}]')
                 for index, item in enumerate(value))


def _matrix(value: Any, name: str) -> Tuple[Tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'{name} must be a 3x3 matrix')
    rows = tuple(_vector(row, 3, f'{name}[{index}]')
                 for index, row in enumerate(value))
    return rows


def _load_json_no_duplicates(path: Path) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f'FK statistics {path} contains duplicate key {key!r}')
            result[key] = value
        return result

    with open(path, 'r', encoding='utf-8') as handle:
        try:
            return json.load(handle, object_pairs_hook=reject_duplicates)
        except json.JSONDecodeError as error:
            raise ValueError(
                f'FK statistics {path} is not valid JSON: {error}') from error


@dataclass(frozen=True)
class EpisodePoseActivity:
    """Measured activity and TCP positions of one episode."""

    active_arm: Tuple[bool, bool]
    translation_spread_m: Tuple[float, float]
    orientation_spread: Tuple[float, float]
    positions: Tuple[Tensor, Tensor]

    def artifact_record(self) -> Dict[str, Any]:
        return {
            'active_arm': [bool(self.active_arm[0]),
                           bool(self.active_arm[1])],
            'translation_spread_m': [
                float(self.translation_spread_m[0]),
                float(self.translation_spread_m[1])],
            'orientation_spread': [
                float(self.orientation_spread[0]),
                float(self.orientation_spread[1])],
        }


@dataclass(frozen=True)
class _EpisodeRecord:
    active_arm: Tuple[bool, bool]
    translation_spread_m: Tuple[float, float]
    orientation_spread: Tuple[float, float]


@dataclass(frozen=True)
class ArmPoseBounds:
    minimum: Tuple[float, float, float]
    maximum: Tuple[float, float, float]
    center: Tuple[float, float, float]
    scale: float


class FkPoseStatistics:
    """Validated versioned training-pose artifact."""

    def __init__(
        self,
        *,
        urdf_sha256: str,
        urdf_path: str,
        thresholds: Mapping[str, float],
        bounds: Mapping[str, ArmPoseBounds],
        episodes: Mapping[str, Mapping[int, _EpisodeRecord]],
        training: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> None:
        self.urdf_sha256 = urdf_sha256
        self.urdf_path = urdf_path
        self.thresholds = dict(thresholds)
        self.bounds = dict(bounds)
        self.episodes = {root: dict(records)
                         for root, records in episodes.items()}
        self.training = dict(training)
        self.raw = dict(raw)

    @classmethod
    def from_path(cls, path: Any) -> 'FkPoseStatistics':
        artifact = Path(path).expanduser()
        if not artifact.is_file():
            raise FileNotFoundError(
                f'FK pose statistics not found: {artifact}')
        return cls.from_mapping(_load_json_no_duplicates(artifact))

    @classmethod
    def from_mapping(cls, payload: Any) -> 'FkPoseStatistics':
        payload = _mapping(payload, 'FK statistics')
        version = _integer(payload.get('schema_version'),
                           'schema_version', minimum=1)
        if version != FK_STATS_SCHEMA_VERSION:
            raise ValueError(
                'Unsupported FK statistics schema_version '
                f'{version}; expected {FK_STATS_SCHEMA_VERSION}')
        if payload.get('kind') != FK_STATS_KIND:
            raise ValueError(
                f"FK statistics kind must be {FK_STATS_KIND!r}, got "
                f"{payload.get('kind')!r}")

        robot = _mapping(payload.get('robot'), 'robot')
        digest = robot.get('urdf_sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError(
                'robot.urdf_sha256 must be a 64-character lowercase hex '
                'digest')
        urdf_path = robot.get('urdf_path')
        if not isinstance(urdf_path, str) or not urdf_path:
            raise ValueError('robot.urdf_path must be a nonempty string')
        if robot.get('root_link') != ALOHA_ROOT_LINK:
            raise ValueError(
                f'robot.root_link must be {ALOHA_ROOT_LINK!r}')
        end_links = _mapping(robot.get('end_links'), 'robot.end_links')
        joint_names = _mapping(robot.get('joint_names'),
                               'robot.joint_names')
        for arm in ALOHA_ARM_NAMES:
            if end_links.get(arm) != ALOHA_END_LINKS[arm]:
                raise ValueError(
                    f'robot.end_links.{arm} must be {ALOHA_END_LINKS[arm]!r}')
            names = joint_names.get(arm)
            if (not isinstance(names, (list, tuple))
                    or tuple(names) != ALOHA_JOINT_NAMES[arm]):
                raise ValueError(
                    f'robot.joint_names.{arm} must be '
                    f'{list(ALOHA_JOINT_NAMES[arm])}')

        tcp = _mapping(payload.get('tcp'), 'tcp')
        offset = _finite(tcp.get('offset_m'), 'tcp.offset_m', positive=True)
        if abs(offset - ALOHA_TCP_OFFSET_M) > 1e-9:
            raise ValueError(
                f'tcp.offset_m must be {ALOHA_TCP_OFFSET_M}, got {offset}')
        if _vector(tcp.get('local_axis'), 3,
                   'tcp.local_axis') != ALOHA_TCP_LOCAL_AXIS:
            raise ValueError('tcp.local_axis must be [1, 0, 0]')
        correction = _matrix(tcp.get('orientation_correction'),
                             'tcp.orientation_correction')
        if correction != ALOHA_ORIENTATION_CORRECTION:
            raise ValueError(
                'tcp.orientation_correction must be diag(1, -1, -1)')

        layout = _mapping(payload.get('action_layout'), 'action_layout')
        if _integer(layout.get('dim'), 'action_layout.dim') != (
                ALOHA_ACTION_DIM):
            raise ValueError(
                f'action_layout.dim must be {ALOHA_ACTION_DIM}')
        arms = _mapping(layout.get('arms'), 'action_layout.arms')
        for index, arm in enumerate(ALOHA_ARM_NAMES):
            expected = [ALOHA_ARM_SLICES[index].start,
                        ALOHA_ARM_SLICES[index].stop]
            value = arms.get(arm)
            if not isinstance(value, (list, tuple)) or list(value) != expected:
                raise ValueError(
                    f'action_layout.arms.{arm} must be {expected}')
        grippers = layout.get('grippers')
        if (not isinstance(grippers, (list, tuple))
                or tuple(grippers) != ALOHA_GRIPPER_INDICES):
            raise ValueError(
                'action_layout.grippers must be '
                f'{list(ALOHA_GRIPPER_INDICES)}')

        thresholds_raw = _mapping(payload.get('thresholds'), 'thresholds')
        thresholds = {
            'translation_spread_m': _finite(
                thresholds_raw.get('translation_spread_m'),
                'thresholds.translation_spread_m', positive=True),
            'orientation_spread': _finite(
                thresholds_raw.get('orientation_spread'),
                'thresholds.orientation_spread', positive=True),
        }

        training = _mapping(payload.get('training'), 'training')
        roots = training.get('roots')
        if not isinstance(roots, (list, tuple)) or not roots:
            raise ValueError('training.roots must be a nonempty list')
        canonical_roots = []
        for index, root in enumerate(roots):
            canonical = canonical_data_root(root)
            if canonical in canonical_roots:
                raise ValueError(
                    'training.roots contains the same canonical root twice: '
                    f'{canonical}')
            canonical_roots.append(canonical)

        bounds_raw = _mapping(payload.get('pose_bounds'), 'pose_bounds')
        bounds = {}
        for arm in ALOHA_ARM_NAMES:
            entry = _mapping(bounds_raw.get(arm),
                             f'pose_bounds.{arm}')
            minimum = _vector(entry.get('min'), 3,
                              f'pose_bounds.{arm}.min')
            maximum = _vector(entry.get('max'), 3,
                              f'pose_bounds.{arm}.max')
            center = _vector(entry.get('center'), 3,
                             f'pose_bounds.{arm}.center')
            scale = _finite(entry.get('scale'),
                            f'pose_bounds.{arm}.scale', positive=True)
            for axis in range(3):
                if minimum[axis] > maximum[axis]:
                    raise ValueError(
                        f'pose_bounds.{arm}.min exceeds max on axis {axis}')
                expected_center = (minimum[axis] + maximum[axis]) / 2
                if abs(center[axis] - expected_center) > 1e-6:
                    raise ValueError(
                        f'pose_bounds.{arm}.center must be the min/max '
                        f'midpoint on axis {axis}')
            expected_scale = max(
                (maximum[axis] - minimum[axis]) / 2 for axis in range(3))
            if expected_scale <= 0 or abs(scale - expected_scale) > 1e-6:
                raise ValueError(
                    f'pose_bounds.{arm}.scale must be the common max '
                    'half-range')
            bounds[arm] = ArmPoseBounds(minimum, maximum, center, scale)

        episodes_raw = _mapping(payload.get('episodes'), 'episodes')
        episodes: Dict[str, Dict[int, _EpisodeRecord]] = {}
        for root, records in episodes_raw.items():
            canonical = canonical_data_root(root)
            if canonical in episodes:
                raise ValueError(
                    'episodes contains ambiguous root mappings for '
                    f'{canonical}; use one canonical root key')
            records = _mapping(records, f'episodes[{root}]')
            converted = {}
            for episode, record in records.items():
                try:
                    episode_index = int(episode)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f'Episode key {episode!r} under {root} is not an '
                        'integer') from error
                if str(episode_index) != str(episode).strip():
                    raise ValueError(
                        f'Episode key {episode!r} under {root} is not a '
                        'canonical integer')
                record = _mapping(
                    record, f'episodes[{root}][{episode}]')
                active = record.get('active_arm')
                if not isinstance(active, (list, tuple)) or len(active) != 2:
                    raise ValueError(
                        f'episodes[{root}][{episode}].active_arm must '
                        'contain two booleans')
                converted[episode_index] = _EpisodeRecord(
                    active_arm=(_strict_bool(
                        active[0],
                        f'episodes[{root}][{episode}].active_arm[0]'),
                        _strict_bool(
                            active[1],
                            f'episodes[{root}][{episode}].active_arm[1]')),
                    translation_spread_m=_vector(
                        record.get('translation_spread_m'), 2,
                        f'episodes[{root}][{episode}].translation_spread_m'),
                    orientation_spread=_vector(
                        record.get('orientation_spread'), 2,
                        f'episodes[{root}][{episode}].orientation_spread'),
                )
            if converted:
                episodes[canonical] = converted
        if not episodes:
            raise ValueError('episodes must contain at least one episode')
        missing_roots = [root for root in canonical_roots
                         if root not in episodes]
        extra_roots = [root for root in episodes
                       if root not in canonical_roots]
        if missing_roots or extra_roots:
            raise ValueError(
                'training.roots and episodes disagree: missing '
                f'{missing_roots[:3]}, extra {extra_roots[:3]}')
        num_episodes = training.get('num_episodes')
        if num_episodes is not None:
            declared = _integer(num_episodes, 'training.num_episodes',
                                minimum=1)
            actual = sum(len(rows) for rows in episodes.values())
            if declared != actual:
                raise ValueError(
                    f'training.num_episodes={declared} but episodes contains '
                    f'{actual}')
        return cls(
            urdf_sha256=digest,
            urdf_path=urdf_path,
            thresholds=thresholds,
            bounds=bounds,
            episodes=episodes,
            training=training,
            raw=dict(payload),
        )

    def validate_robot(self, urdf_path: Any) -> Path:
        """Reject an artifact built for a different robot description."""
        path = Path(urdf_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'ALOHA URDF not found: {path}')
        actual = file_sha256(path)
        if actual != self.urdf_sha256:
            raise ValueError(
                'FK statistics robot identity does not match the configured '
                f'URDF: artifact {self.urdf_sha256}, file {actual} '
                f'({path})')
        return path

    def active_arm(self, root: Any, episode: Any) -> Tuple[bool, bool]:
        """Strict dataset-root/episode lookup in ``[left, right]`` order."""
        canonical = canonical_data_root(root)
        if canonical not in self.episodes:
            raise KeyError(
                f'No FK episode labels for data root {canonical}. The '
                'artifact training roots are '
                f'{sorted(self.episodes)[:3]}')
        if isinstance(episode, bool):
            raise ValueError('Episode index must be an integer')
        try:
            index = int(episode)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'Episode index must be an integer, got {episode!r}'
            ) from error
        records = self.episodes[canonical]
        if index not in records:
            raise KeyError(
                f'No FK episode label for root {canonical} episode '
                f'{index}; the artifact has {len(records)} episodes for '
                'that root')
        return records[index].active_arm

    def bounds_tensors(
        self, *, device: torch.device, dtype: torch.dtype = torch.float32,
    ) -> Tuple[Tensor, Tensor]:
        center = torch.tensor(
            [self.bounds[arm].center for arm in ALOHA_ARM_NAMES],
            device=device, dtype=dtype)
        scale = torch.tensor(
            [self.bounds[arm].scale for arm in ALOHA_ARM_NAMES],
            device=device, dtype=dtype)
        return center, scale


class AlohaFk:
    """Cached serial chains for both ALOHA arms with RoboTwin's TCP frame."""

    def __init__(
        self,
        urdf_path: Any = None,
        *,
        robotwin_root: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.urdf_path = Path(
            default_aloha_urdf_path(robotwin_root=robotwin_root)
            if urdf_path is None else urdf_path).expanduser()
        if self.urdf_path.suffix.lower() != '.urdf':
            raise ValueError(
                f'ALOHA URDF path must end in .urdf: {self.urdf_path}')
        self.dtype = dtype
        self.device: Optional[torch.device] = None
        self._chains = None
        self._end_indices = None
        self._offset = None
        self._correction = None

    def _build(self) -> None:
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f'ALOHA URDF not found: {self.urdf_path}')
        import pytorch_kinematics as pk

        # The asset has an XML encoding declaration, so pass bytes rather than
        # a decoded string.
        data = self.urdf_path.read_bytes()
        chains = {}
        end_indices = {}
        for arm in ALOHA_ARM_NAMES:
            chain = pk.build_serial_chain_from_urdf(
                data, ALOHA_END_LINKS[arm])
            names = tuple(chain.get_joint_parameter_names())
            if names != ALOHA_JOINT_NAMES[arm]:
                raise ValueError(
                    f'{ALOHA_END_LINKS[arm]} chain joints are {names}; '
                    f'expected {ALOHA_JOINT_NAMES[arm]}')
            chains[arm] = chain
            end_indices[arm] = int(
                chain.get_frame_indices(ALOHA_END_LINKS[arm]).item())
        self._chains = chains
        self._end_indices = end_indices
        self._offset = torch.tensor(
            ALOHA_TCP_LOCAL_AXIS, dtype=self.dtype) * ALOHA_TCP_OFFSET_M
        self._correction = torch.tensor(
            ALOHA_ORIENTATION_CORRECTION, dtype=self.dtype)

    def _ensure_device(self, device: torch.device) -> None:
        if self._chains is None:
            self._build()
        if self.device == device:
            return
        for chain in self._chains.values():
            chain.to(device=device, dtype=self.dtype)
        self.device = device
        self._offset = self._offset.to(device=device, dtype=self.dtype)
        self._correction = self._correction.to(
            device=device, dtype=self.dtype)

    def tcp_pose(self, joints: Tensor, arm: str) -> Tuple[Tensor, Tensor]:
        """Return TCP position and corrected rotation.

        Shapes are ``[...,3]`` and ``[...,3,3]``.
        """
        if arm not in ALOHA_ARM_NAMES:
            raise ValueError(f'arm must be left or right, got {arm!r}')
        if not torch.is_tensor(joints) or joints.shape[-1] != 6:
            raise ValueError('ALOHA joint input must be [..., 6]')
        if not torch.isfinite(joints).all():
            raise ValueError('ALOHA joint input contains non-finite values')
        self._ensure_device(joints.device)
        flat = joints.to(dtype=self.dtype).reshape(-1, 6)
        transforms = self._chains[arm].forward_kinematics_tensor(flat)
        transform = transforms[self._end_indices[arm]]
        rotation = transform[:, :3, :3]
        corrected = rotation @ self._correction
        position = transform[:, :3, 3] + corrected @ self._offset
        return (position.reshape(*joints.shape[:-1], 3),
                corrected.reshape(*joints.shape[:-1], 3, 3))


def episode_pose_activity(
    actions: Any,
    fk: AlohaFk,
    *,
    translation_threshold: float = DEFAULT_TRANSLATION_SPREAD_M,
    orientation_threshold: float = DEFAULT_ORIENTATION_SPREAD,
) -> EpisodePoseActivity:
    """Classify both arms from raw, full-episode physical joint actions."""
    translation_threshold = _finite(
        translation_threshold, 'translation_threshold', positive=True)
    orientation_threshold = _finite(
        orientation_threshold, 'orientation_threshold', positive=True)
    actions = torch.as_tensor(actions, dtype=fk.dtype)
    if (actions.ndim != 2 or actions.shape[1] != ALOHA_ACTION_DIM
            or actions.shape[0] < 1):
        raise ValueError(
            'Episode actions must be [T, 14] with at least one frame')
    if not torch.isfinite(actions).all():
        raise ValueError('Episode actions contain non-finite values')
    active = []
    translation_spreads = []
    orientation_spreads = []
    positions = []
    for arm_index, arm in enumerate(ALOHA_ARM_NAMES):
        joints = actions[:, ALOHA_ARM_SLICES[arm_index]]
        position, rotation = fk.tcp_pose(joints, arm)
        mean_position = position.mean(dim=0)
        translation = (position - mean_position).square().sum(-1).mean()
        translation = translation.sqrt()
        mean_rotation = rotation.mean(dim=0)
        orientation = (rotation - mean_rotation).square()
        orientation = (orientation.sum(dim=(-1, -2)).mean()
                       .div(2.0).sqrt())
        active.append(bool(
            translation.item() > translation_threshold
            or orientation.item() > orientation_threshold))
        translation_spreads.append(float(translation.item()))
        orientation_spreads.append(float(orientation.item()))
        positions.append(position.detach())
    return EpisodePoseActivity(
        active_arm=(active[0], active[1]),
        translation_spread_m=(translation_spreads[0],
                              translation_spreads[1]),
        orientation_spread=(orientation_spreads[0],
                            orientation_spreads[1]),
        positions=(positions[0], positions[1]),
    )


class PoseBoundsAccumulator:
    """Accumulate per-axis TCP min/max from active-arm frames only."""

    def __init__(self) -> None:
        self._minimum = [None, None]
        self._maximum = [None, None]

    def add(self, arm_index: int, positions: Tensor) -> None:
        if arm_index not in (0, 1):
            raise ValueError('arm_index must be 0 (left) or 1 (right)')
        positions = positions.detach()
        if (positions.ndim != 2 or positions.shape[1] != 3
                or positions.shape[0] < 1):
            raise ValueError('TCP positions must be [N, 3]')
        if not torch.isfinite(positions).all():
            raise ValueError('TCP positions contain non-finite values')
        low = positions.min(dim=0).values
        high = positions.max(dim=0).values
        if self._minimum[arm_index] is None:
            self._minimum[arm_index] = low
            self._maximum[arm_index] = high
        else:
            self._minimum[arm_index] = torch.minimum(
                self._minimum[arm_index], low)
            self._maximum[arm_index] = torch.maximum(
                self._maximum[arm_index], high)

    def finalize(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for arm_index, arm in enumerate(ALOHA_ARM_NAMES):
            if self._minimum[arm_index] is None:
                raise ValueError(
                    f'No active episodes for the {arm} arm; pose bounds '
                    'cannot be computed')
            minimum = self._minimum[arm_index].double()
            maximum = self._maximum[arm_index].double()
            center = (minimum + maximum) / 2
            half_range = (maximum - minimum) / 2
            scale = half_range.max()
            if not torch.isfinite(scale) or scale.item() <= 0:
                raise ValueError(
                    f'The {arm} arm has a non-positive/constant TCP bound')
            result[arm] = {
                'min': minimum.tolist(),
                'max': maximum.tolist(),
                'center': center.tolist(),
                'scale': float(scale.item()),
            }
        return result


def _autocast_disabled(tensor: Tensor):
    device_type = tensor.device.type
    if device_type in ('cpu', 'cuda'):
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _connected_zero(reference: Tensor) -> Tensor:
    """A finite differentiable zero that tolerates poisoned unused values."""
    anchor = reference.reshape(-1)[0]
    anchor = torch.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0)
    return anchor * 0.0


def _validate_action_epsilon(epsilon: Any, reference: Tensor,
                             batch: int) -> Tensor:
    if epsilon is None:
        return reference.new_tensor(1e-8).expand(batch)
    epsilon = torch.as_tensor(epsilon, device=reference.device,
                              dtype=torch.float32)
    if epsilon.ndim == 0:
        epsilon = epsilon.expand(batch)
    elif epsilon.ndim == 1 and epsilon.shape[0] == batch:
        pass
    elif epsilon.ndim == 2 and epsilon.shape == (batch, 1):
        epsilon = epsilon[:, 0]
    else:
        raise ValueError(
            'fk_action_epsilon must be a scalar or [B] values')
    if not torch.isfinite(epsilon).all() or (epsilon < 0).any():
        raise ValueError('fk_action_epsilon must be finite and nonnegative')
    return epsilon


def compute_fk_pose_loss(
    fk: AlohaFk,
    predicted_copies: Tensor,
    target_actions: Tensor,
    active_arm: Tensor,
    valid_dimensions: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
    action_epsilon: Any,
    center: Tensor,
    scale: Tensor,
    *,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """Per-copy TCP pose loss over active, valid arm/timestep tuples.

    ``predicted_copies`` is ``[B, H, R, 14]`` in normalized action space with
    its copy axis intact. Each copy is denormalized and passed through FK
    independently; the target pose is shared across copies.
    """
    position_weight = _finite(position_weight, 'position_weight', minimum=0)
    rotation_weight = _finite(rotation_weight, 'rotation_weight', minimum=0)
    if (not torch.is_tensor(predicted_copies)
            or predicted_copies.ndim != 4
            or predicted_copies.shape[-1] != ALOHA_ACTION_DIM):
        raise ValueError('predicted_copies must be [B,H,R,14]')
    batch, horizon, copies, _ = predicted_copies.shape
    if copies < 1:
        raise ValueError('predicted_copies needs at least one copy')
    if (not torch.is_tensor(target_actions)
            or target_actions.shape != (batch, horizon, ALOHA_ACTION_DIM)):
        raise ValueError('target_actions must be [B,H,14]')
    if (not torch.is_tensor(active_arm)
            or active_arm.shape != (batch, 2)):
        raise ValueError('active_arm must be [B,2]')
    if (not torch.is_tensor(valid_dimensions)
            or valid_dimensions.shape
            != (batch, horizon, ALOHA_ACTION_DIM)):
        raise ValueError('valid_dimensions must be [B,H,14]')
    if (not torch.is_tensor(action_mean)
            or action_mean.shape != (batch, ALOHA_ACTION_DIM)):
        raise ValueError('fk_action_mean must be [B,14]')
    if (not torch.is_tensor(action_std)
            or action_std.shape != (batch, ALOHA_ACTION_DIM)):
        raise ValueError('fk_action_std must be [B,14]')
    if center.shape != (2, 3) or scale.shape != (2,):
        raise ValueError('Pose bounds must be center [2,3] and scale [2]')
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
        raise ValueError('Pose bounds must be finite')
    if (scale <= 0).any():
        raise ValueError('Pose scales must be positive')

    device = predicted_copies.device
    active_arm = active_arm.to(device=device, dtype=torch.bool)
    valid_dimensions = valid_dimensions.to(device=device, dtype=torch.bool)
    action_mean = action_mean.to(device=device, dtype=torch.float32)
    action_std = action_std.to(device=device, dtype=torch.float32)
    center = center.to(device=device, dtype=torch.float32)
    scale = scale.to(device=device, dtype=torch.float32)
    epsilon = _validate_action_epsilon(
        action_epsilon, predicted_copies, batch)

    with _autocast_disabled(predicted_copies):
        predicted = predicted_copies.float()
        target = target_actions.to(device=device, dtype=torch.float32)
        denominator = (action_std + epsilon[:, None])
        predicted_physical = (
            predicted * denominator[:, None, None, :]
            + action_mean[:, None, None, :])
        target_physical = (
            target * denominator[:, None, :]
            + action_mean[:, None, :])

        position_error_sum = None
        rotation_error_sum = None
        distance_sum = None
        mean_distance_sum = None
        pair_count = 0
        tuple_count = 0
        for arm_index, arm in enumerate(ALOHA_ARM_NAMES):
            joints = ALOHA_ARM_SLICES[arm_index]
            arm_selected = (
                active_arm[:, arm_index, None]
                & valid_dimensions[..., joints].all(dim=-1))
            if not arm_selected.any():
                continue
            predicted_joints = predicted_physical[..., joints][arm_selected]
            target_joints = target_physical[..., joints][arm_selected]
            if not torch.isfinite(predicted_joints).all():
                raise ValueError(
                    f'Predicted {arm} joint values are non-finite for an '
                    'active valid arm/timestep')
            if not torch.isfinite(target_joints).all():
                raise ValueError(
                    f'Target {arm} joint values are non-finite for an '
                    'active valid arm/timestep')

            rows, copy_count, _ = predicted_joints.shape
            with torch.no_grad():
                target_position, target_rotation = fk.tcp_pose(
                    target_joints, arm)
            predicted_position, predicted_rotation = fk.tcp_pose(
                predicted_joints.reshape(-1, 6), arm)
            predicted_position = predicted_position.reshape(
                rows, copy_count, 3)
            predicted_rotation = predicted_rotation.reshape(
                rows, copy_count, 3, 3)
            target_position = target_position[:, None, :]
            target_rotation = target_rotation[:, None, :, :]

            arm_scale = scale[arm_index]
            position_error = (
                (predicted_position - target_position) / arm_scale
            ).square().mean(dim=-1)
            rotation_error = (
                predicted_rotation - target_rotation
            ).square().mean(dim=(-1, -2))
            distance = (predicted_position - target_position).norm(dim=-1)

            if position_error_sum is None:
                position_error_sum = position_error.sum()
                rotation_error_sum = rotation_error.sum()
                distance_sum = distance.sum()
                mean_distance_sum = predicted_position.new_zeros(())
            else:
                position_error_sum = position_error_sum + position_error.sum()
                rotation_error_sum = (
                    rotation_error_sum + rotation_error.sum())
                distance_sum = distance_sum + distance.sum()

            with torch.no_grad():
                mean_joints = predicted_joints.mean(dim=1)
                mean_position, _ = fk.tcp_pose(mean_joints, arm)
                mean_distance = (
                    mean_position - target_position[:, 0, :]).norm(dim=-1)
                mean_distance_sum = (
                    mean_distance_sum + mean_distance.sum())
            pair_count += rows
            tuple_count += rows * copy_count

        if tuple_count == 0:
            zero = _connected_zero(predicted_copies)
            detached_zero = zero.detach()
            return {
                'loss_fk': zero,
                'loss_fk_position': zero,
                'loss_fk_rotation': zero,
                'loss_fk_tcp_error_m': detached_zero,
                'loss_fk_tcp_error_mean_action_m': detached_zero,
                'loss_fk_active_arm_timestep_pairs':
                    predicted_copies.new_tensor(0),
                'loss_fk_active_arm_timestep_copy_tuples':
                    predicted_copies.new_tensor(0),
            }

        position_error = position_error_sum / tuple_count
        rotation_error = rotation_error_sum / tuple_count
        composite = (position_weight * position_error
                     + rotation_weight * rotation_error)
        return {
            'loss_fk': composite,
            'loss_fk_position': position_error,
            'loss_fk_rotation': rotation_error,
            'loss_fk_tcp_error_m': distance_sum.detach() / tuple_count,
            'loss_fk_tcp_error_mean_action_m':
                mean_distance_sum.detach() / pair_count,
            'loss_fk_active_arm_timestep_pairs': (
                predicted_copies.new_tensor(float(pair_count))),
            'loss_fk_active_arm_timestep_copy_tuples': (
                predicted_copies.new_tensor(float(tuple_count))),
        }


class FkPoseObjective:
    """Validated statistics plus cached ALOHA chains for training."""

    def __init__(
        self,
        statistics: FkPoseStatistics,
        *,
        urdf_path: Any = None,
        robotwin_root: Optional[str] = None,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> None:
        self.statistics = statistics
        self.position_weight = _finite(
            position_weight, 'position_weight', minimum=0)
        self.rotation_weight = _finite(
            rotation_weight, 'rotation_weight', minimum=0)
        self.urdf_path = statistics.validate_robot(
            default_aloha_urdf_path(robotwin_root=robotwin_root)
            if urdf_path is None else urdf_path)
        self.fk = AlohaFk(self.urdf_path, dtype=torch.float32)

    @classmethod
    def from_path(
        cls,
        stats_path: Any,
        *,
        urdf_path: Any = None,
        robotwin_root: Optional[str] = None,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
    ) -> 'FkPoseObjective':
        return cls(
            FkPoseStatistics.from_path(stats_path),
            urdf_path=urdf_path,
            robotwin_root=robotwin_root,
            position_weight=position_weight,
            rotation_weight=rotation_weight,
        )

    def __call__(
        self,
        predicted_copies: Tensor,
        target_actions: Tensor,
        active_arm: Tensor,
        valid_dimensions: Tensor,
        action_mean: Tensor,
        action_std: Tensor,
        action_epsilon: Any,
        center: Tensor,
        scale: Tensor,
    ) -> Dict[str, Tensor]:
        return compute_fk_pose_loss(
            self.fk,
            predicted_copies,
            target_actions,
            active_arm,
            valid_dimensions,
            action_mean,
            action_std,
            action_epsilon,
            center,
            scale,
            position_weight=self.position_weight,
            rotation_weight=self.rotation_weight,
        )
