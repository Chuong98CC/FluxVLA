"""End-to-end test of the PatchWAM FK statistics generator."""

from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from fluxvla.models.third_party_models.patchwam.fk_pose import (
    AlohaFk,
    FkPoseStatistics,
    canonical_data_root,
    episode_pose_activity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts/generate_patchwam_fk_pose_stats.py'
URDF = REPO_ROOT / (
    'src/RoboTwin/assets/embodiments/aloha-agilex/urdf/'
    'arx5_description_isaac.urdf')
_FORCE_MISSING_FK_ASSETS = (
    os.environ.get('FLUXVLA_TEST_FORCE_MISSING_FK_ASSETS') == '1')


def _runtime_available():
    return (not _FORCE_MISSING_FK_ASSETS
            and importlib.util.find_spec('pytorch_kinematics') is not None)


def _asset_available():
    return not _FORCE_MISSING_FK_ASSETS and URDF.is_file()


def _skip_reason():
    missing = []
    if not _runtime_available():
        missing.append('pytorch_kinematics')
    if not _asset_available():
        missing.append(str(URDF))
    return None if not missing else 'Missing optional FK test dependency: ' + (
        ', '.join(missing))


pytestmark = pytest.mark.skipif(
    _skip_reason() is not None, reason=_skip_reason() or '')


def _episode_actions(left: bool, active: bool) -> list:
    actions = []
    for frame in range(8):
        values = [0.0] * 14
        if active:
            values[0 if left else 7] = -0.2 + 0.4 * frame / 7
        elif left:
            values[0] = 1.2
        else:
            values[7] = 1.2
        actions.append(values)
    return actions


def _both_actions(offset: float = 0.0) -> list:
    actions = []
    for frame in range(8):
        values = [0.0] * 14
        values[0] = -0.2 + 0.4 * frame / 7 + offset
        values[7] = 0.2 - 0.4 * frame / 7 + offset
        actions.append(values)
    return actions


def _write_config_root(root: Path, task: str) -> None:
    meta = root / 'meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text('{}', encoding='utf-8')
    (meta / 'tasks.jsonl').write_text(
        json.dumps({'task_index': 0, 'task': task}) + '\n',
        encoding='utf-8')
    with (meta / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(2):
            handle.write(json.dumps({
                'episode_index': episode,
                'tasks': [task],
                'length': 8,
            }) + '\n')
    data = root / 'data' / 'chunk-000'
    data.mkdir(parents=True)
    for episode in range(2):
        pd.DataFrame({
            'episode_index': [episode] * 8,
            'action': _both_actions(
                0.0 if task == 'task_a' else 0.8),
        }).to_parquet(data / f'episode_{episode:06d}.parquet')


def test_generator_classifies_episodes_and_excludes_inactive_frames(tmp_path):
    root = tmp_path / 'task'
    meta = root / 'meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text('{}', encoding='utf-8')
    data = root / 'data' / 'chunk-000'
    data.mkdir(parents=True)
    for episode, (left, active) in enumerate(
            ((True, True), (False, True))):
        pd.DataFrame({
            'episode_index': [episode] * 8,
            'action': _episode_actions(left, active),
        }).to_parquet(data / f'episode_{episode:06d}.parquet')

    output = tmp_path / 'stats.json'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--root', str(root),
         '--output', str(output), '--urdf-path', str(URDF)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding='utf-8'))
    canonical = canonical_data_root(root)
    assert artifact['training']['roots'] == [canonical]
    assert artifact['training']['num_episodes'] == 2
    assert artifact['activity_summary']['counts'] == {
        'left_only': 1, 'right_only': 1, 'both': 0, 'neither': 0}
    assert artifact['episodes'][canonical]['0']['active_arm'] == [True, False]
    assert artifact['episodes'][canonical]['1']['active_arm'] == [False, True]

    stats = FkPoseStatistics.from_path(output)
    assert stats.active_arm(root, 0) == (True, False)
    assert stats.active_arm(root, 1) == (False, True)
    assert stats.bounds['left'].scale > 0
    assert stats.bounds['right'].scale > 0

    fk = AlohaFk(URDF, dtype=torch.float64)
    left_episode = episode_pose_activity(
        torch.tensor(_episode_actions(True, True), dtype=torch.float64), fk)
    left_min = left_episode.positions[0].min(dim=0).values
    left_max = left_episode.positions[0].max(dim=0).values
    torch.testing.assert_close(
        torch.tensor(artifact['pose_bounds']['left']['min'],
                     dtype=torch.float64),
        left_min.double(), rtol=0, atol=1e-9)
    torch.testing.assert_close(
        torch.tensor(artifact['pose_bounds']['left']['max'],
                     dtype=torch.float64),
        left_max.double(), rtol=0, atol=1e-9)
    far_left = fk.tcp_pose(
        torch.tensor([[1.2, 0, 0, 0, 0, 0]], dtype=torch.float64),
        'left')[0][0]
    low = torch.tensor(artifact['pose_bounds']['left']['min'],
                       dtype=torch.float64)
    high = torch.tensor(artifact['pose_bounds']['left']['max'],
                        dtype=torch.float64)
    assert not ((far_left >= low) & (far_left <= high)).all()


def test_config_selection_excludes_other_roots_from_bounds(tmp_path):
    root_a = tmp_path / 'root_a'
    root_b = tmp_path / 'root_b'
    _write_config_root(root_a, 'task_a')
    _write_config_root(root_b, 'task_b')
    config = tmp_path / 'train_config.py'
    config.write_text(
        'train_dataloader = dict(dataset=dict(\n'
        "    type='DistributedRepeatingDataset',\n"
        '    datasets=dict(\n'
        "        type='PatchWAMParquetDataset',\n"
        f'        data_root_path=[{str(root_a)!r}, {str(root_b)!r}],\n'
        "        task_selection=dict(tasks=['task_a']),\n"
        '    ),\n'
        '))\n',
        encoding='utf-8')
    output = tmp_path / 'selected_stats.json'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--config', str(config),
         '--output', str(output), '--urdf-path', str(URDF)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding='utf-8'))
    canonical_a = canonical_data_root(root_a)
    assert artifact['training']['roots'] == [canonical_a]
    assert artifact['training']['selection']['source'] == 'config'
    assert artifact['training']['selection']['task_selection'] is not None
    assert set(artifact['episodes'][canonical_a]) == {'0', '1'}

    fk = AlohaFk(URDF, dtype=torch.float64)
    positions = {'left': [], 'right': []}
    for episode in (0, 1):
        activity = episode_pose_activity(
            torch.tensor(_both_actions(), dtype=torch.float64), fk)
        for arm_index, arm in enumerate(('left', 'right')):
            positions[arm].append(activity.positions[arm_index])
    for arm_index, arm in enumerate(('left', 'right')):
        expected = torch.cat(positions[arm])
        torch.testing.assert_close(
            torch.tensor(artifact['pose_bounds'][arm]['min'],
                         dtype=torch.float64),
            expected.min(dim=0).values, rtol=0, atol=1e-9)
        torch.testing.assert_close(
            torch.tensor(artifact['pose_bounds'][arm]['max'],
                         dtype=torch.float64),
            expected.max(dim=0).values, rtol=0, atol=1e-9)


def test_config_train_episode_fraction_limits_episodes(tmp_path):
    root = tmp_path / 'root'
    _write_config_root(root, 'task_a')
    config = tmp_path / 'fraction_config.py'
    config.write_text(
        'train_dataloader = dict(dataset=dict(\n'
        "    type='DistributedRepeatingDataset',\n"
        '    datasets=dict(\n'
        "        type='PatchWAMParquetDataset',\n"
        f'        data_root_path=[{str(root)!r}],\n'
        '        train_episode_fraction=0.5,\n'
        '    ),\n'
        '))\n',
        encoding='utf-8')
    output = tmp_path / 'fraction_stats.json'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--config', str(config),
         '--output', str(output), '--urdf-path', str(URDF)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding='utf-8'))
    canonical = canonical_data_root(root)
    assert artifact['training']['roots'] == [canonical]
    assert set(artifact['episodes'][canonical]) == {'0'}
    assert artifact['training']['selection']['train_episode_fraction'] == 0.5
