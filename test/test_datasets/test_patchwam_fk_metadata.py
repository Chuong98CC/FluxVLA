"""Dataset/transform/collator tests for PatchWAM FK metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fluxvla.collators.dict_collator import DictCollator
from fluxvla.engines import build_dataset_from_cfg
from fluxvla.transforms.transform_patchwam import PatchWAMFkActionMetadata


def _write_root(root: Path, episodes: int = 2, rows: int = 6) -> None:
    meta = root / 'meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text(json.dumps({
        'fps': 1.0,
        'chunks_size': 1000,
    }), encoding='utf-8')
    (meta / 'tasks.jsonl').write_text(
        json.dumps({'task_index': 0, 'task': 'move the block'}) + '\n',
        encoding='utf-8')
    with (meta / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(episodes):
            handle.write(json.dumps({
                'episode_index': episode,
                'tasks': [0],
                'length': rows,
            }) + '\n')
    action_stats = {
        'min': [0.0] * 14,
        'max': [2.0] * 14,
        'mean': [1.0] * 14,
        'std': [0.5] * 14,
    }
    state_stats = {
        'min': [0.0] * 14,
        'max': [20.0] * 14,
        'mean': [10.0] * 14,
        'std': [5.0] * 14,
    }
    with (meta / 'episodes_stats.jsonl').open(
            'w', encoding='utf-8') as handle:
        for episode in range(episodes):
            handle.write(json.dumps({
                'episode_index': episode,
                'stats': {
                    'observation.state': state_stats,
                    'action': action_stats,
                    'timestamp': {
                        'min': [0.0],
                        'max': [float(rows - 1)],
                        'mean': [float(rows - 1) / 2],
                        'std': [1.0],
                    },
                },
            }) + '\n')
    data_dir = root / 'data' / 'chunk-000'
    data_dir.mkdir(parents=True)
    for episode in range(episodes):
        frame = pd.DataFrame({
            'observation.state': [
                [float(frame)] * 14 for frame in range(rows)],
            'action': [
                [0.1 * frame] * 14 for frame in range(rows)],
            'timestamp': [float(frame) for frame in range(rows)],
            'frame_index': list(range(rows)),
            'episode_index': [episode] * rows,
            'task_index': [0] * rows,
            'subtask': ['grasp'] * rows,
        })
        frame.to_parquet(
            data_dir / f'episode_{episode:06d}.parquet')


def _write_stats(root: Path, episodes=(0,)) -> Path:
    payload = {
        'schema_version': 1,
        'kind': 'patchwam_aloha_fk_pose_stats',
        'robot': {
            'urdf_path': 'unused-by-dataset-tests.urdf',
            'urdf_sha256': 'a' * 64,
            'root_link': 'footprint',
            'end_links': {'left': 'fl_link6', 'right': 'fr_link6'},
            'joint_names': {
                'left': [f'fl_joint{i}' for i in range(1, 7)],
                'right': [f'fr_joint{i}' for i in range(1, 7)],
            },
        },
        'tcp': {
            'offset_m': 0.12,
            'local_axis': [1.0, 0.0, 0.0],
            'orientation_correction': [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        },
        'action_layout': {
            'dim': 14,
            'arms': {'left': [0, 6], 'right': [7, 13]},
            'grippers': [6, 13],
        },
        'thresholds': {
            'translation_spread_m': 0.01,
            'orientation_spread': 0.05,
        },
        'training': {
            'roots': [str(root)],
            'num_episodes': len(episodes),
        },
        'pose_bounds': {
            arm: {
                'min': [-3.0, -3.0, -3.0],
                'max': [3.0, 3.0, 3.0],
                'center': [0.0, 0.0, 0.0],
                'scale': 3.0,
            } for arm in ('left', 'right')
        },
        'episodes': {
            str(root): {
                str(episode): {
                    'active_arm': [True, episode % 2 == 0],
                    'translation_spread_m': [0.1, 0.1],
                    'orientation_spread': [0.1, 0.1],
                } for episode in episodes
            },
        },
    }
    path = root.parent / 'fk_stats.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _dataset_cfg(root: Path, stats: Path):
    return {
        'type': 'DistributedRepeatingDataset',
        'shuffle': False,
        'statistic_name': 'robotwin',
        'statistic_keys': ['observation.state', 'timestamp', 'action'],
        'name_mappings': {
            'observation.state': ['proprio'],
            'action': ['action'],
        },
        'datasets': {
            'type': 'PatchWAMParquetDataset',
            'data_root_path': [str(root)],
            'action_window_size': 4,
            'window_start_idx': 0,
            'frame_window_size': 5,
            'frame_sample_stride': 1,
            'future_offset': 4,
            'horizon': 4,
            'action_key': 'action',
            'statistic_name': 'robotwin',
            'training': True,
            'fk_stats_path': str(stats),
            'require_fk_stats': True,
            'transforms': [
                {'type': 'PatchWAMRawInputs'},
                {
                    'type': 'NormalizeStatesAndActions',
                    'state_key': 'proprio',
                    'action_key': 'action',
                    'action_dim': 14,
                    'state_dim': 14,
                    'norm_type': 'mean_std',
                    'statistics_key': 'stats',
                    'normalization_epsilon': 1e-8,
                },
                {
                    'type': 'PatchWAMFkActionMetadata',
                    'action_key': 'action',
                    'action_dim': 14,
                    'statistics_key': 'stats',
                    'normalization_epsilon': 1e-8,
                },
                {
                    'type': 'PatchWAMKeep',
                    'keys': [
                        'actions', 'action_masks', 'states', 'active_arm',
                        'fk_action_mean', 'fk_action_std',
                        'fk_action_epsilon',
                    ],
                    'require': True,
                },
            ],
        },
    }


def test_dataset_transform_and_collator_carry_fk_metadata(tmp_path):
    root = tmp_path / 'task'
    _write_root(root)
    stats = _write_stats(root)
    wrapper = build_dataset_from_cfg(_dataset_cfg(root, stats))
    sample = wrapper._get_item_from_global_idx(0)
    assert sample['active_arm'].dtype == np.bool_
    assert sample['active_arm'].tolist() == [True, True]
    torch.testing.assert_close(
        torch.as_tensor(sample['fk_action_mean']),
        torch.ones(14))
    torch.testing.assert_close(
        torch.as_tensor(sample['fk_action_std']),
        torch.full((14,), 0.5))
    assert float(torch.as_tensor(sample['fk_action_epsilon'])) == (
        float(np.float32(1e-8)))

    collator = DictCollator(keys=[
        'actions', 'action_masks', 'states', 'active_arm',
        'fk_action_mean', 'fk_action_std', 'fk_action_epsilon'])
    batch = collator([sample, wrapper._get_item_from_global_idx(1)])
    assert batch['active_arm'].shape == (2, 2)
    assert batch['active_arm'].dtype == torch.bool
    assert batch['fk_action_mean'].shape == (2, 14)
    assert batch['fk_action_std'].shape == (2, 14)
    assert batch['actions'].shape == (2, 4, 14)


def test_dataset_requires_and_rejects_missing_episode_labels(tmp_path):
    root = tmp_path / 'task'
    _write_root(root)
    stats = _write_stats(root, episodes=(0,))
    wrapper = build_dataset_from_cfg(_dataset_cfg(root, stats))
    with pytest.raises(ValueError, match='Missing FK episode activity'):
        wrapper._get_item_from_global_idx(6)

    cfg = _dataset_cfg(root, stats)
    cfg['datasets']['fk_stats_path'] = None
    with pytest.raises(ValueError, match='require_fk_stats'):
        build_dataset_from_cfg(cfg['datasets'])


def test_fk_metadata_transform_rejects_missing_fields():
    transform = PatchWAMFkActionMetadata(action_dim=14)
    with pytest.raises(ValueError, match='active_arm'):
        transform({'stats': {'action': {
            'mean': [0.0] * 14, 'std': [1.0] * 14}}})
    with pytest.raises(ValueError, match='mean'):
        transform({
            'active_arm': [True, False],
            'stats': {'action': {'std': [1.0] * 14}},
        })
    with pytest.raises(ValueError, match='width'):
        transform({
            'active_arm': [True, False],
            'stats': {'action': {
                'mean': [0.0] * 13, 'std': [1.0] * 13}},
        })
