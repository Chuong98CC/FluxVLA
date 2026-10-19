"""Generate PatchWAM RoboTwin ALOHA FK pose and episode-activity statistics.

The artifact is training-only metadata: per-episode arm activity and spreads,
plus TCP position bounds computed from actively moving arms. Only the action
column (and episode index) of the selected Parquet episodes is read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from mmengine import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module(name: str, relative_path: str):
    """Load a dependency-light module without importing fluxvla/__init__.

    Importing the package root initializes the full model registry, including
    optional CUDA-only backends. The generator is a CPU-only data tool and must
    run in an environment without a visible GPU.
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_fk_pose = _load_module(
    '_patchwam_fk_pose',
    'fluxvla/models/third_party_models/patchwam/fk_pose.py')
_task_selection = _load_module(
    '_patchwam_task_selection',
    'fluxvla/datasets/utils/task_selection.py')
ALOHA_ACTION_DIM = _fk_pose.ALOHA_ACTION_DIM
ALOHA_ARM_NAMES = _fk_pose.ALOHA_ARM_NAMES
ALOHA_ARM_SLICES = _fk_pose.ALOHA_ARM_SLICES
ALOHA_END_LINKS = _fk_pose.ALOHA_END_LINKS
ALOHA_GRIPPER_INDICES = _fk_pose.ALOHA_GRIPPER_INDICES
ALOHA_JOINT_NAMES = _fk_pose.ALOHA_JOINT_NAMES
ALOHA_ORIENTATION_CORRECTION = _fk_pose.ALOHA_ORIENTATION_CORRECTION
ALOHA_ROOT_LINK = _fk_pose.ALOHA_ROOT_LINK
ALOHA_TCP_LOCAL_AXIS = _fk_pose.ALOHA_TCP_LOCAL_AXIS
ALOHA_TCP_OFFSET_M = _fk_pose.ALOHA_TCP_OFFSET_M
DEFAULT_ORIENTATION_SPREAD = _fk_pose.DEFAULT_ORIENTATION_SPREAD
DEFAULT_TRANSLATION_SPREAD_M = _fk_pose.DEFAULT_TRANSLATION_SPREAD_M
FK_STATS_KIND = _fk_pose.FK_STATS_KIND
FK_STATS_SCHEMA_VERSION = _fk_pose.FK_STATS_SCHEMA_VERSION
AlohaFk = _fk_pose.AlohaFk
PoseBoundsAccumulator = _fk_pose.PoseBoundsAccumulator
canonical_data_root = _fk_pose.canonical_data_root
episode_pose_activity = _fk_pose.episode_pose_activity
file_sha256 = _fk_pose.file_sha256
resolve_task_selection = _task_selection.resolve_task_selection

DEFAULT_OUTPUT = (
    'work_dirs/patchwam_fk_loss/robotwin_fk_pose_stats.json')
DEFAULT_CONFIG = 'configs/patchwam/robotwin/patchwam_robotwin.py'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', default=None,
        help='PatchWAM training config used in default mode. Its '
             'train_dataloader dataset roots and episode/task selection '
             f'define the statistics set. Defaults to {DEFAULT_CONFIG}.')
    parser.add_argument(
        '--data-root',
        default=None,
        help='Manual-mode parent directory containing one task root per '
             'subdirectory. Defaults to PATCHWAM_DATA_ROOT or '
             '/data/RoboTwin/RoboTwin-Clean.')
    parser.add_argument(
        '--tasks', nargs='+', default=None,
        help='Manual-mode task subdirectories under --data-root. Using this '
             'bypasses config-selected roots and episodes.')
    parser.add_argument(
        '--root', action='append', default=None,
        help='Manual-mode dataset root. May be repeated; bypasses the '
             'config-provided roots and episode selection.')
    parser.add_argument(
        '--episode-selection', default=None,
        help='JSON mapping a canonical root path or task name to episode '
             'indices. In config mode it must be a subset of the config '
             'selection; in manual mode it defines the episode subset.')
    parser.add_argument(
        '--output', default=DEFAULT_OUTPUT,
        help='Destination JSON artifact.')
    parser.add_argument(
        '--urdf-path', default=os.environ.get('PATCHWAM_FK_URDF') or None,
        help='ALOHA URDF. Defaults to the installed RoboTwin asset.')
    parser.add_argument(
        '--robotwin-root', default=os.environ.get('FLUXVLA_ROBOTWIN_ROOT')
        or None,
        help='RoboTwin checkout root used to resolve the default URDF.')
    parser.add_argument(
        '--translation-threshold', type=float,
        default=DEFAULT_TRANSLATION_SPREAD_M)
    parser.add_argument(
        '--rotation-threshold', type=float,
        default=DEFAULT_ORIENTATION_SPREAD)
    return parser.parse_args()


def _discover_roots(
    data_root: Path, tasks: Optional[Iterable[str]],
) -> List[Path]:
    if tasks is None:
        roots = [
            path for path in sorted(data_root.iterdir())
            if path.is_dir() and (path / 'meta/info.json').is_file()
            and (path / 'data').is_dir()
        ]
    else:
        roots = [data_root / task for task in tasks]
    if not roots:
        raise ValueError(f'No task roots found under {data_root}')
    for root in roots:
        if not (root / 'meta/info.json').is_file():
            raise FileNotFoundError(f'Missing task metadata: {root}')
        if not (root / 'data').is_dir():
            raise FileNotFoundError(f'Missing task data: {root}')
    canonical = [canonical_data_root(root) for root in roots]
    if len(set(canonical)) != len(canonical):
        raise ValueError('Selected roots contain duplicate canonical paths')
    return [Path(path) for path in canonical]


def _load_episode_selection(path: Optional[str], roots: List[Path]):
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError('--episode-selection must contain a JSON object')
    by_root = {canonical_data_root(root): root for root in roots}
    by_task = {root.name: canonical_data_root(root) for root in roots}
    resolved = {}
    for key, value in payload.items():
        if not isinstance(value, list) or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in value):
            raise ValueError(
                f'Episode selection for {key!r} must be a list of integers')
        if key in by_task:
            canonical = by_task[key]
        else:
            canonical = canonical_data_root(key)
        if canonical not in by_root:
            raise ValueError(
                f'Episode selection key {key!r} does not match a selected '
                'root')
        if canonical in resolved:
            raise ValueError(
                f'Episode selection has duplicate entries for {canonical}')
        resolved[canonical] = sorted(set(value))
    return resolved


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing metadata file: {path}')
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_order(root: Path) -> List[int]:
    """Episode order of the concatenated Parquet table, without actions."""
    files = sorted((root / 'data').glob('**/*.parquet'))
    if not files:
        raise FileNotFoundError(f'No Parquet data under {root / "data"}')
    order = []
    for path in files:
        table = pq.read_table(path, columns=['episode_index'])
        for episode in table['episode_index'].to_pylist():
            episode = int(episode)
            if not order or order[-1] != episode:
                order.append(episode)
    return list(dict.fromkeys(order))


def _config_training_selection(config_path: Path):
    """Resolve roots/episodes from the actual training dataloader config."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f'Training config not found: {config_path}')
    cfg = Config.fromfile(str(config_path))
    wrapper = cfg.train_dataloader.dataset
    if wrapper.get('type') != 'DistributedRepeatingDataset':
        raise ValueError(
            'Stats generation supports the PatchWAM training recipe with a '
            'DistributedRepeatingDataset train_dataloader, got '
            f'{wrapper.get("type")!r}')
    if wrapper.get('task_selection') is not None:
        raise ValueError(
            'Wrapper-level task_selection is unsupported by the FK statistics '
            'generator; put task_selection on the PatchWAM dataset child')
    child = wrapper.get('datasets')
    if not isinstance(child, dict):
        raise ValueError(
            'Stats generation requires the training wrapper to contain one '
            'PatchWAMParquetDataset mapping; list/grouped datasets are '
            'unsupported because their episode sets cannot be resolved '
            'unambiguously')
    if child.get('type') != 'PatchWAMParquetDataset':
        raise ValueError(
            'Stats generation requires a PatchWAMParquetDataset training '
            f'child, got {child.get("type")!r}')
    raw_roots = child.get('data_root_path')
    if isinstance(raw_roots, (str, os.PathLike)):
        raw_roots = [raw_roots]
    if not isinstance(raw_roots, (list, tuple)) or not raw_roots:
        raise ValueError(
            'PatchWAM training config must declare data_root_path')
    roots = [Path(canonical_data_root(root)) for root in raw_roots]
    if len(set(roots)) != len(roots):
        raise ValueError('Training config contains duplicate data roots')
    for root in roots:
        if not (root / 'meta/tasks.jsonl').is_file():
            raise FileNotFoundError(f'Missing tasks metadata: {root}')
        if not (root / 'meta/episodes.jsonl').is_file():
            raise FileNotFoundError(f'Missing episodes metadata: {root}')
        if not (root / 'data').is_dir():
            raise FileNotFoundError(f'Missing task data: {root}')

    fraction = child.get('train_episode_fraction', 1.0)
    if isinstance(fraction, bool):
        raise ValueError('train_episode_fraction must be a number')
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError(
            'train_episode_fraction must be in (0, 1], got '
            f'{fraction}')
    task_selection = child.get('task_selection')
    resolved_selection = None
    keep = None
    if task_selection is not None:
        tasks = [_read_jsonl(root / 'meta/tasks.jsonl') for root in roots]
        episodes = [
            _read_jsonl(root / 'meta/episodes.jsonl') for root in roots]
        resolved = resolve_task_selection(tasks, episodes, task_selection)
        resolved_selection = resolved['selection']
        keep = resolved['keep']

    selected = {}
    for root_index, root in enumerate(roots):
        order = _episode_order(root)
        if keep is not None:
            allowed = {episode for index, episode in keep
                       if index == root_index}
            missing = sorted(allowed - set(order))
            if missing:
                raise ValueError(
                    f'Config selection refers to episodes missing from '
                    f'{root}: {missing[:8]}')
            order = [episode for episode in order if episode in allowed]
        if not order:
            # A config root can be present in data_root_path while its task is
            # excluded by task_selection. It contributes no training episodes
            # and must not enter the pose bounds.
            continue
        if fraction != 1.0:
            count = max(1, min(int(len(order) * fraction), len(order)))
            order = order[:count]
        selected[canonical_data_root(root)] = order
    if not selected:
        raise ValueError('Config training selection leaves no episodes at all')
    roots = [root for root in roots
             if canonical_data_root(root) in selected]
    report = {
        'source': 'config',
        'config': str(config_path.resolve()),
        'train_episode_fraction': fraction,
        'repeat_to_full_length': bool(
            child.get('repeat_to_full_length', False)),
        'task_selection': resolved_selection,
        'roots': list(selected),
        'episodes': {
            root: len(episodes) for root, episodes in selected.items()},
    }
    return roots, selected, report


def _read_actions(
    root: Path, selected: Optional[List[int]],
) -> Dict[int, np.ndarray]:
    files = sorted((root / 'data').glob('**/*.parquet'))
    if not files:
        raise FileNotFoundError(f'No Parquet data under {root / "data"}')
    episodes: Dict[int, np.ndarray] = {}
    for path in files:
        table = pq.read_table(path, columns=['episode_index', 'action'])
        if table.num_rows == 0:
            continue
        unique = np.unique(np.asarray(
            table['episode_index'].to_pylist(), dtype=np.int64))
        if unique.size != 1:
            raise ValueError(
                f'Parquet file mixes episodes: {path} ({unique.tolist()})')
        episode = int(unique[0])
        if selected is not None and episode not in selected:
            continue
        values = table['action'].to_pylist()
        actions = np.asarray(values, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ALOHA_ACTION_DIM:
            raise ValueError(
                f'{path} actions are {actions.shape}; expected '
                f'[T,{ALOHA_ACTION_DIM}]')
        if actions.shape[0] == 0:
            raise ValueError(f'{path} has no action rows')
        if not np.isfinite(actions).all():
            raise ValueError(f'{path} contains non-finite actions')
        if episode in episodes:
            episodes[episode] = np.concatenate(
                (episodes[episode], actions), axis=0)
        else:
            episodes[episode] = actions
    if selected is not None:
        missing = sorted(set(selected) - set(episodes))
        if missing:
            raise ValueError(
                f'Selected episodes are missing from {root}: {missing}')
    if not episodes:
        raise ValueError(f'No episodes selected from {root}')
    return episodes


def _distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {'min': float('nan'), 'median': float('nan'),
                'max': float('nan')}
    array = np.asarray(values, dtype=np.float64)
    return {
        'min': float(array.min()),
        'median': float(np.median(array)),
        'max': float(array.max()),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def main() -> None:
    args = _parse_args()
    manual_mode = bool(args.root or args.tasks)
    if args.root and args.tasks:
        raise ValueError('Use either --root or --tasks, not both')
    if manual_mode and args.config:
        raise ValueError(
            '--config cannot be combined with manual --root/--tasks; manual '
            'roots bypass config episode selection')
    if not manual_mode and args.data_root:
        raise ValueError(
            '--data-root is a manual-mode option; pass --tasks with it or '
            'omit it to use the training config')
    if args.root and args.data_root:
        raise ValueError(
            '--data-root only applies to --tasks; --root entries are explicit')

    if manual_mode:
        if args.tasks:
            data_root = Path(
                args.data_root
                or os.environ.get('PATCHWAM_DATA_ROOT')
                or '/data/RoboTwin/RoboTwin-Clean')
            roots = _discover_roots(data_root, args.tasks)
        else:
            roots = [Path(canonical_data_root(root)) for root in args.root]
            for root in roots:
                if not (root / 'data').is_dir():
                    raise FileNotFoundError(f'Missing task data: {root}')
        manual_selection = _load_episode_selection(
            args.episode_selection, roots)
        selected_by_root = {
            canonical_data_root(root): manual_selection.get(
                canonical_data_root(root))
            for root in roots
        }
        selection_report = {
            'source': 'manual',
            'config': None,
            'data_root': (
                canonical_data_root(data_root)
                if args.tasks else None),
            'tasks': [root.name for root in roots] if args.tasks else None,
            'manual_roots': [canonical_data_root(root) for root in roots],
            'episode_selection': {
                key: list(value)
                for key, value in manual_selection.items()
            },
        }
    else:
        config_path = Path(args.config or DEFAULT_CONFIG).expanduser()
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        roots, selected_by_root, selection_report = (
            _config_training_selection(config_path))
        manual_selection = _load_episode_selection(
            args.episode_selection, roots)
        for canonical, episodes in manual_selection.items():
            allowed = selected_by_root[canonical]
            outside = sorted(set(episodes) - set(allowed))
            if outside:
                raise ValueError(
                    '--episode-selection includes episodes excluded by the '
                    f'training config for {canonical}: {outside[:8]}')
            selected_by_root[canonical] = episodes
        if manual_selection:
            selection_report['manual_episode_selection'] = {
                key: list(value)
                for key, value in manual_selection.items()
            }

    fk = AlohaFk(args.urdf_path, robotwin_root=args.robotwin_root,
                 dtype=torch.float64)
    accumulator = PoseBoundsAccumulator()
    episodes_artifact: Dict[str, Dict[str, Any]] = {}
    spreads = {
        arm: {'translation_spread_m': [], 'orientation_spread': []}
        for arm in ALOHA_ARM_NAMES
    }
    counts = {'left_only': 0, 'right_only': 0, 'both': 0, 'neither': 0}
    active_counts = {arm: 0 for arm in ALOHA_ARM_NAMES}
    total_frames = 0
    total_episodes = 0

    for root in roots:
        canonical = canonical_data_root(root)
        selected = selected_by_root[canonical]
        actions_by_episode = _read_actions(root, selected)
        records = {}
        for episode, actions in sorted(actions_by_episode.items()):
            activity = episode_pose_activity(
                torch.from_numpy(actions),
                fk,
                translation_threshold=args.translation_threshold,
                orientation_threshold=args.rotation_threshold,
            )
            records[str(episode)] = activity.artifact_record()
            total_episodes += 1
            total_frames += int(actions.shape[0])
            left, right = activity.active_arm
            if left and right:
                counts['both'] += 1
            elif left:
                counts['left_only'] += 1
            elif right:
                counts['right_only'] += 1
            else:
                counts['neither'] += 1
            for arm_index, arm in enumerate(ALOHA_ARM_NAMES):
                if activity.active_arm[arm_index]:
                    active_counts[arm] += 1
                    accumulator.add(arm_index, activity.positions[arm_index])
                    spreads[arm]['translation_spread_m'].append(
                        activity.translation_spread_m[arm_index])
                    spreads[arm]['orientation_spread'].append(
                        activity.orientation_spread[arm_index])
        episodes_artifact[canonical] = records
        left_count = sum(
            1 for record in records.values() if record['active_arm'][0])
        right_count = sum(
            1 for record in records.values() if record['active_arm'][1])
        print(
            f'{root.name}: {len(records)} episodes, '
            f'{left_count} left, {right_count} right',
            flush=True,
        )

    bounds = accumulator.finalize()
    urdf_path = (
        Path(args.urdf_path).expanduser()
        if args.urdf_path else fk.urdf_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(f'ALOHA URDF not found: {urdf_path}')
    artifact = {
        'schema_version': FK_STATS_SCHEMA_VERSION,
        'kind': FK_STATS_KIND,
        'robot': {
            'urdf_path': str(urdf_path.resolve()),
            'urdf_sha256': file_sha256(urdf_path),
            'root_link': ALOHA_ROOT_LINK,
            'end_links': dict(ALOHA_END_LINKS),
            'joint_names': {
                arm: list(ALOHA_JOINT_NAMES[arm])
                for arm in ALOHA_ARM_NAMES
            },
        },
        'tcp': {
            'offset_m': ALOHA_TCP_OFFSET_M,
            'local_axis': list(ALOHA_TCP_LOCAL_AXIS),
            'orientation_correction': [
                list(row) for row in ALOHA_ORIENTATION_CORRECTION
            ],
        },
        'action_layout': {
            'dim': ALOHA_ACTION_DIM,
            'arms': {
                arm: [ALOHA_ARM_SLICES[index].start,
                      ALOHA_ARM_SLICES[index].stop]
                for index, arm in enumerate(ALOHA_ARM_NAMES)
            },
            'grippers': list(ALOHA_GRIPPER_INDICES),
        },
        'thresholds': {
            'translation_spread_m': float(args.translation_threshold),
            'orientation_spread': float(args.rotation_threshold),
        },
        'training': {
            'roots': list(episodes_artifact),
            'num_episodes': total_episodes,
            'num_frames': total_frames,
            'selection': selection_report,
        },
        'activity_summary': {
            'counts': counts,
            'active_episodes': active_counts,
            'per_arm': {
                arm: {
                    'active_episodes': active_counts[arm],
                    'inactive_episodes': (
                        total_episodes - active_counts[arm]),
                    'translation_spread_m': _distribution(
                        spreads[arm]['translation_spread_m']),
                    'orientation_spread': _distribution(
                        spreads[arm]['orientation_spread']),
                }
                for arm in ALOHA_ARM_NAMES
            },
        },
        'pose_bounds': bounds,
        'episodes': episodes_artifact,
    }
    output = Path(args.output).expanduser()
    _write_json(output, artifact)
    print(json.dumps({
        'output': str(output),
        'num_roots': len(episodes_artifact),
        'num_episodes': total_episodes,
        'num_frames': total_frames,
        'activity': counts,
        'active_episodes': active_counts,
        'pose_bounds': {
            arm: {'center': bounds[arm]['center'],
                  'scale': bounds[arm]['scale']}
            for arm in ALOHA_ARM_NAMES
        },
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
