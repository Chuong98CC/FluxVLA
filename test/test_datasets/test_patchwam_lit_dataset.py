# Copyright 2026 Limx Dynamics
"""CPU units for the terminal LIT goal dataset/collator integration."""

import numpy as np
import pytest
from datasets import Dataset
from unittest import mock

from fluxvla.collators.dict_collator import DictCollator
from fluxvla.datasets.patchwam_dataset import PatchWAMParquetDataset
from fluxvla.datasets.patchwam_lit_dataset import PatchWAMLITParquetDataset
from fluxvla.transforms.transform_patchwam_lit import PatchWAMLITGoal


def _make_dataset(rows_per_episode=20, episodes=1, goal_offset=16,
                  with_state=True, **settings):
    total = rows_per_episode * episodes
    episode_index = np.repeat(np.arange(episodes), rows_per_episode)
    frame_index = np.tile(np.arange(rows_per_episode), episodes)
    dataset = PatchWAMLITParquetDataset.__new__(PatchWAMLITParquetDataset)
    columns = {
        'episode_index': episode_index.tolist(),
        'frame_index': frame_index.tolist(),
        'task_index': [0] * total,
        'timestamp': frame_index.tolist(),
        'action': [[float(i)] for i in range(total)],
    }
    if with_state:
        columns['observation.state'] = [[float(100 + i), float(200 + i)]
                                        for i in range(total)]
    dataset.dataset = Dataset.from_dict(columns)
    dataset.dataset_cumulative_sizes = np.array([0, total])
    dataset.sample_indices = np.arange(total)
    dataset.tasks = [[dict(task='move')]]
    dataset.info = [dict(fps=2.0)]
    dataset.data_root_path = ['unused']
    dataset.statistic_name = 'test'
    dataset.transforms = []
    dataset.action_key = 'action'
    dataset.action_dtype = np.dtype('float32')
    dataset.use_delta = False
    dataset.expose_index = True
    dataset.supervise_terminal_padding = False
    dataset.require_full_window = False
    dataset.frame_window_size = 1
    dataset.frame_sample_stride = 1
    dataset.window_start_idx = 0
    dataset.action_window_size = 16
    dataset.future_offset = 16
    dataset.horizon = 16
    dataset.goal_offset = goal_offset
    dataset.goal_state_key = 'observation.state'
    dataset.history_enabled = False
    dataset.training = True
    for name, value in settings.items():
        setattr(dataset, name, value)
    return dataset


def test_goal_is_the_state_at_start_plus_offset_not_the_last_action():
    dataset = _make_dataset()
    sample = dataset.__getitem__(0, {'test': {}})
    assert sample['patchwam_goal_valid'] is True
    assert sample['patchwam_goal_row'] == 16
    np.testing.assert_array_equal(sample['patchwam_goal_state_raw'],
                                  np.array([116.0, 216.0], dtype=np.float32))
    # Not the t+15 state, not the last action of the 16-step window.
    assert not np.allclose(sample['patchwam_goal_state_raw'], [115.0, 215.0])
    assert sample['patchwam_goal_state_raw'][0] != sample['actions'][-1, 0]


def test_goal_is_invalid_past_the_episode_tail():
    dataset = _make_dataset(rows_per_episode=20)
    sample = dataset.__getitem__(4, {'test': {}})
    assert sample['patchwam_future_valid'] is False
    assert sample['patchwam_goal_valid'] is False
    assert sample['patchwam_goal_state_raw'] is None
    assert sample['patchwam_goal_row'] == 4


def test_goal_never_crosses_an_episode_boundary():
    dataset = _make_dataset(rows_per_episode=20, episodes=2)
    sample = dataset.__getitem__(5, {'test': {}})
    # Row 21 is episode 1; the goal must stay inside episode 0.
    assert sample['patchwam_goal_valid'] is False
    assert sample['patchwam_goal_state_raw'] is None


def test_goal_offset_is_configurable_and_bounded():
    dataset = _make_dataset(rows_per_episode=20, goal_offset=3)
    sample = dataset.__getitem__(0, {'test': {}})
    assert sample['patchwam_goal_row'] == 3
    np.testing.assert_array_equal(sample['patchwam_goal_state_raw'],
                                  np.array([103.0, 203.0], dtype=np.float32))
    with pytest.raises(ValueError, match='goal_offset'):
        with mock.patch.object(PatchWAMParquetDataset, '__init__',
                               lambda self, **kwargs: None):
            PatchWAMLITParquetDataset(goal_offset=0)


def test_goal_transform_uses_state_statistics_not_action_statistics():
    state_stats = {'mean': [10.0, 20.0], 'std': [2.0, 4.0],
                   'min': [0.0, 0.0], 'max': [20.0, 40.0]}
    action_stats = {'mean': [1.0, 1.0], 'std': [0.5, 0.5],
                    'min': [0.0, 0.0], 'max': [2.0, 2.0]}
    transform = PatchWAMLITGoal(state_key='proprio', state_dim=14,
                                norm_type='mean_std')
    data = {
        'patchwam_goal_state_raw': np.array([12.0, 24.0], dtype=np.float32),
        'patchwam_goal_valid': True,
        'stats': {'proprio': state_stats, 'action': action_stats},
    }
    out = transform(data)
    goal = out['goal_state']
    assert goal.shape == (14,)
    np.testing.assert_allclose(goal[:2], [1.0, 1.0], rtol=1e-5)
    assert goal[2:].sum() == 0.0
    assert bool(out['goal_valid'])
    # Using the action statistics instead would give a different answer.
    assert not np.allclose(goal[:2], [(12.0 - 1.0) / 0.5, (24.0 - 1.0) / 0.5])


def test_goal_transform_min_max_and_invalid_goals():
    stats = {'proprio': {'min': [0.0, 0.0], 'max': [10.0, 20.0],
                         'mean': [5.0, 10.0], 'std': [1.0, 1.0]}}
    transform = PatchWAMLITGoal(state_key='proprio', state_dim=2,
                                norm_type='min_max')
    out = transform({
        'patchwam_goal_state_raw': np.array([5.0, 20.0], dtype=np.float32),
        'patchwam_goal_valid': True, 'stats': stats})
    np.testing.assert_allclose(out['goal_state'], [0.0, 1.0], atol=1e-6)
    invalid = transform({
        'patchwam_goal_state_raw': None, 'patchwam_goal_valid': False,
        'stats': stats})
    np.testing.assert_array_equal(invalid['goal_state'], np.zeros(2))
    assert not bool(invalid['goal_valid'])


def test_goal_transform_fails_loudly_on_missing_or_mismatched_stats():
    transform = PatchWAMLITGoal(state_key='proprio', state_dim=2,
                                norm_type='mean_std')
    with pytest.raises(ValueError, match='Missing state statistics'):
        transform({
            'patchwam_goal_state_raw': np.array([1.0, 1.0], dtype=np.float32),
            'patchwam_goal_valid': True, 'stats': {}})
    with pytest.raises(ValueError, match='does not match'):
        transform({
            'patchwam_goal_state_raw': np.array([1.0, 1.0, 1.0],
                                                dtype=np.float32),
            'patchwam_goal_valid': True,
            'stats': {'proprio': {'mean': [0.0, 0.0], 'std': [1.0, 1.0]}}})
    with pytest.raises(ValueError, match='non-finite'):
        transform({
            'patchwam_goal_state_raw': np.array([float('nan'), 1.0],
                                                dtype=np.float32),
            'patchwam_goal_valid': True,
            'stats': {'proprio': {'mean': [0.0, 0.0], 'std': [1.0, 1.0]}}})
    # norm_type='none' needs no statistics at all.
    identity = PatchWAMLITGoal(state_key='proprio', state_dim=2,
                               norm_type='none')
    out = identity({
        'patchwam_goal_state_raw': np.array([3.0, 4.0], dtype=np.float32),
        'patchwam_goal_valid': True, 'stats': {}})
    np.testing.assert_allclose(out['goal_state'], [3.0, 4.0])
    assert bool(out['goal_valid'])


def test_goal_transform_epsilon_matches_the_state_configuration():
    # A zero-variance dimension makes the configured epsilon observable.
    stats = {'proprio': {'mean': [0.0, 0.0], 'std': [1e-8, 1.0],
                         'min': [0.0, 0.0], 'max': [1.0, 1.0]}}
    transform = PatchWAMLITGoal(state_key='proprio', state_dim=2,
                                norm_type='mean_std',
                                normalization_epsilon=1e-8)
    out = transform({
        'patchwam_goal_state_raw': np.array([1.0, 0.0], dtype=np.float32),
        'patchwam_goal_valid': True, 'stats': stats})
    np.testing.assert_allclose(out['goal_state'][0], 1.0 / (1e-8 + 1e-8),
                               rtol=1e-5)
    small = PatchWAMLITGoal(state_key='proprio', state_dim=2,
                            norm_type='mean_std',
                            normalization_epsilon=1e-6)
    other = small({
        'patchwam_goal_state_raw': np.array([1.0, 0.0], dtype=np.float32),
        'patchwam_goal_valid': True, 'stats': stats})
    assert not np.allclose(out['goal_state'][0], other['goal_state'][0])


def test_missing_goal_state_column_at_a_valid_row_is_a_schema_error():
    dataset = _make_dataset(with_state=False)
    with pytest.raises(KeyError, match='observation.state'):
        dataset.__getitem__(0, {'test': {}})


def test_collator_stacks_goal_state_and_validity():
    collator = DictCollator(
        keys=['actions', 'action_masks', 'states', 'goal_state', 'goal_valid'],
        meta_keys=['task_description'])
    samples = []
    for index in range(2):
        samples.append({
            'actions': np.zeros((16, 14), dtype=np.float32),
            'action_masks': np.ones((16, 14), dtype=np.float32),
            'states': np.zeros(14, dtype=np.float32),
            'goal_state': np.full(14, float(index), dtype=np.float32),
            'goal_valid': np.asarray(index == 0, dtype=bool),
            'task_description': 'move',
        })
    batch = collator(samples)
    assert batch['goal_state'].shape == (2, 14)
    assert batch['goal_valid'].shape == (2,)
    assert batch['goal_valid'].tolist() == [True, False]


def test_baseline_patchwam_dataset_is_unchanged():
    dataset = PatchWAMParquetDataset.__new__(PatchWAMParquetDataset)
    dataset.dataset = Dataset.from_dict({
        'episode_index': [0, 0],
        'frame_index': [0, 1],
        'task_index': [0, 0],
        'timestamp': [0.0, 1.0],
        'action': [[0.0], [1.0]],
    })
    dataset.dataset_cumulative_sizes = np.array([0, 2])
    dataset.sample_indices = np.arange(2)
    dataset.tasks = [[dict(task='move')]]
    dataset.info = [dict(fps=2.0)]
    dataset.data_root_path = ['unused']
    dataset.statistic_name = 'test'
    dataset.transforms = []
    dataset.action_key = 'action'
    dataset.action_dtype = np.dtype('float32')
    dataset.use_delta = False
    dataset.expose_index = True
    dataset.supervise_terminal_padding = False
    dataset.require_full_window = False
    dataset.frame_window_size = 1
    dataset.frame_sample_stride = 1
    dataset.window_start_idx = 0
    dataset.action_window_size = 16
    dataset.future_offset = 16
    dataset.horizon = 16
    dataset.history_enabled = False
    dataset.training = True
    sample = dataset.__getitem__(0, {'test': {}})
    assert 'patchwam_goal_state_raw' not in sample
    assert 'patchwam_goal_valid' not in sample
    assert sample['patchwam_future_row'] == 0
