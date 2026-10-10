# Copyright 2026 Limx Dynamics
"""CPU units for configurable PatchWAM action sources (synthetic fixtures)."""

import numpy as np
import pytest

from datasets import Dataset
from fluxvla.datasets.patchwam_dataset import PatchWAMParquetDataset
from fluxvla.datasets.parquet_dataset import ParquetDataset

JOINT = [[float(i), float(i) + 100.0] for i in range(40)]
END_EFFECTOR = [[float(i) + 1000.0, float(i) + 2000.0, float(i) + 3000.0]
                for i in range(40)]
JOINT_STATS = {'mean': [0.0, 1.0], 'std': [1.0, 2.0], 'min': [-1.0, 0.0],
               'max': [1.0, 2.0], 'q01': [-1.0, 0.0], 'q99': [1.0, 2.0]}
EE_STATS = {'mean': [10.0, 11.0, 12.0], 'std': [2.0, 2.0, 2.0],
            'min': [0.0, 0.0, 0.0], 'max': [20.0, 20.0, 20.0],
            'q01': [1.0, 1.0, 1.0], 'q99': [19.0, 19.0, 19.0]}


def _make_dataset(*, action_mode='joint_only', **settings):
    total = 40
    dataset = PatchWAMParquetDataset.__new__(PatchWAMParquetDataset)
    dataset.dataset = Dataset.from_dict({
        'episode_index': [0] * total,
        'frame_index': list(range(total)),
        'task_index': [0] * total,
        'timestamp': [i / 2.0 for i in range(total)],
        'action': JOINT,
        'ee_action': END_EFFECTOR,
    })
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
    dataset.history_enabled = False
    dataset.training = True
    dataset.statistics_key = 'stats'
    dataset.action_mode = action_mode
    dataset.joint_action_key = 'action'
    dataset.end_effector_action_key = None
    dataset.joint_action_dim = None
    dataset.end_effector_action_dim = None
    dataset.joint_action_stats_key = 'action'
    dataset.end_effector_action_stats_key = None
    for name, value in settings.items():
        setattr(dataset, name, value)
    return dataset


def _stats():
    return {'test': {'action': JOINT_STATS, 'ee_action': EE_STATS}}


def _dual_dataset(**settings):
    cfg = dict(action_mode='dual', end_effector_action_key='ee_action',
               joint_action_dim=2, end_effector_action_dim=3,
               end_effector_action_stats_key='ee_action')
    cfg.update(settings)
    return _make_dataset(**cfg)


def test_joint_only_action_source_is_exactly_legacy():
    dataset = _make_dataset()
    assert dataset._get_action_value(3) == JOINT[3]
    sample = dataset.__getitem__(0, {'test': {'action': JOINT_STATS}})
    np.testing.assert_allclose(sample['actions'][:, 0], np.arange(16))
    assert sample['stats']['action'] is JOINT_STATS


def test_end_effector_only_reads_the_pose_field_and_reroutes_stats():
    dataset = _make_dataset(action_mode='end_effector_only',
                            end_effector_action_key='ee_action',
                            end_effector_action_dim=3,
                            end_effector_action_stats_key='ee_action')
    np.testing.assert_allclose(dataset._get_action_value(2), END_EFFECTOR[2])
    sample = dataset.__getitem__(0, _stats())
    assert sample['actions'].shape == (16, 3)
    np.testing.assert_allclose(sample['actions'][:, 0],
                               np.arange(16) + 1000.0)
    # stats['action'] is the pose statistics, never the joint field's.
    assert sample['stats']['action'] is EE_STATS
    assert sample['stats']['ee_action'] is EE_STATS


def test_dual_packs_joint_then_pose_and_concatenates_statistics():
    dataset = _dual_dataset()
    value = dataset._get_action_value(4)
    assert value.shape == (5,)
    np.testing.assert_allclose(value, [4.0, 104.0, 1004.0, 2004.0, 3004.0])
    sample = dataset.__getitem__(0, _stats())
    assert sample['actions'].shape == (16, 5)
    np.testing.assert_allclose(sample['actions'][0],
                               [0.0, 100.0, 1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(sample['actions'][5],
                               [5.0, 105.0, 1005.0, 2005.0, 3005.0])
    assert sample['patchwam_window_rows'][0] == 0
    assert sample['patchwam_future_row'] == 16
    derived = sample['stats']['action']
    assert derived['mean'] == [0.0, 1.0, 10.0, 11.0, 12.0]
    assert derived['std'] == [1.0, 2.0, 2.0, 2.0, 2.0]
    assert derived['min'] == [-1.0, 0.0, 0.0, 0.0, 0.0]
    assert derived['max'] == [1.0, 2.0, 20.0, 20.0, 20.0]
    assert derived['q01'] == [-1.0, 0.0, 1.0, 1.0, 1.0]
    # The original per-source statistics stay untouched.
    assert sample['stats']['action'] is not JOINT_STATS
    assert sample['stats']['ee_action'] is EE_STATS


def test_dual_terminal_padding_repeats_the_start_row_and_masks():
    sample = _dual_dataset().__getitem__(30, _stats())
    assert sample['patchwam_future_valid'] is False
    # Out-of-episode steps repeat the last valid packed action and are masked.
    np.testing.assert_allclose(sample['actions'][-1],
                               [39.0, 139.0, 1039.0, 2039.0, 3039.0])
    assert np.count_nonzero(sample['action_masks']) == 10


def test_missing_action_source_field_fails_clearly():
    dataset = _dual_dataset(end_effector_action_key='missing_field')
    with pytest.raises(ValueError, match='missing_field'):
        dataset._get_action_value(0)


def test_configured_width_mismatch_fails_clearly():
    dataset = _dual_dataset()
    dataset.end_effector_action_dim = 2
    with pytest.raises(ValueError, match='end_effector_action_dim'):
        dataset._get_action_value(0)
    dataset = _make_dataset(action_mode='end_effector_only',
                            end_effector_action_key='ee_action',
                            end_effector_action_dim=4)
    with pytest.raises(ValueError, match='end_effector_action_dim'):
        dataset._get_action_value(0)


def test_missing_action_statistics_fail_clearly():
    dataset = _dual_dataset(end_effector_action_stats_key='missing_stats')
    with pytest.raises(KeyError, match='missing_stats'):
        dataset.__getitem__(0, _stats())
    dataset = _dual_dataset(joint_action_stats_key='missing_joint_stats')
    with pytest.raises(KeyError, match='missing_joint_stats'):
        dataset.__getitem__(0, _stats())


def test_partial_and_prefix_mismatched_statistics_are_rejected():
    from fluxvla.datasets.patchwam_dataset import _combine_action_statistics
    # A quantile present for only one source must never be fabricated.
    with pytest.raises(ValueError, match='q01'):
        _combine_action_statistics([
            {'mean': [0.0, 1.0], 'q01': [-1.0, 0.0]},
            {'mean': [10.0, 11.0]},
        ])
    # Prefix shapes must agree; concatenation is along the last axis only.
    combined = _combine_action_statistics([
        {'mean': [[0.0, 1.0], [2.0, 3.0]]},
        {'mean': [[10.0], [12.0]]},
    ])
    assert combined['mean'] == [[0.0, 1.0, 10.0], [2.0, 3.0, 12.0]]
    with pytest.raises(ValueError, match='prefix'):
        _combine_action_statistics([
            {'mean': [[0.0, 1.0], [2.0, 3.0]]},
            {'mean': [10.0, 11.0]},
        ])


def test_custom_joint_source_and_statistics_reach_normalization():
    dataset = _make_dataset(
        joint_action_key='ee_action', joint_action_dim=3,
        joint_action_stats_key='ee_action')
    sample = dataset.__getitem__(0, _stats())
    # The configured joint field (not the legacy 'action' column) is read.
    np.testing.assert_allclose(sample['actions'][:, 0],
                               np.arange(16) + 1000.0)
    # ... and the configured statistics column becomes stats['action'] for the
    # normalization transform (which reads `statistics[action_key]`).
    assert sample['stats']['action'] is EE_STATS
    assert sample['stats']['ee_action'] is EE_STATS
    assert sample['actions'].shape == (16, 3)
    from fluxvla.transforms.normalize import NormalizeStatesAndActions
    raw = sample['actions'].copy()
    sample['states'] = np.zeros((1, 1), dtype=np.float32)
    normalize = NormalizeStatesAndActions(
        state_key=None, action_key='action', action_dim=3, state_dim=1,
        norm_type='mean_std', normalize_states=False)
    normalized = normalize(sample)
    expected = (raw - np.asarray(EE_STATS['mean'])) / np.asarray(EE_STATS['std'])
    np.testing.assert_allclose(normalized['actions'], expected, rtol=1e-6)


def test_prefix_statistics_reach_the_dataset_without_flattening():
    dataset = _dual_dataset()
    stats = {'test': {
        'action': {'mean': [[1.0, 2.0], [3.0, 4.0]]},
        'ee_action': {'mean': [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]}}}
    sample = dataset.__getitem__(0, stats)
    np.testing.assert_array_equal(sample['stats']['action']['mean'],
                                  [[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]])


def test_dataset_constructor_validates_action_settings(monkeypatch):
    # Isolate the action-metadata validation from the native storage init.
    def _stub_init(self, *args, **kwargs):
        self.window_start_idx = 0
        self.action_window_size = 16
        self.action_key = kwargs.get('action_key', 'action')

    monkeypatch.setattr(ParquetDataset, '__init__', _stub_init)
    with pytest.raises(ValueError, match='end_effector_action_key'):
        PatchWAMParquetDataset(
            data_root_path=['unused'], transforms=[], action_key='action',
            statistic_name='test', action_mode='dual',
            joint_action_dim=14, end_effector_action_dim=14)
    with pytest.raises(ValueError, match='action_mode'):
        PatchWAMParquetDataset(
            data_root_path=['unused'], transforms=[], action_key='action',
            statistic_name='test', action_mode='both')
    with pytest.raises(ValueError, match='distinct'):
        PatchWAMParquetDataset(
            data_root_path=['unused'], transforms=[], action_key='action',
            statistic_name='test', action_mode='dual',
            joint_action_key='action', end_effector_action_key='action',
            joint_action_dim=14, end_effector_action_dim=14)
