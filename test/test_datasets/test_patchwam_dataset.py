# Copyright 2026 Limx Dynamics
"""CPU units for the PatchWAM temporal dataset metadata."""

import numpy as np
import pytest

from datasets import Dataset
from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.datasets.patchwam_dataset import PatchWAMParquetDataset


def _make_dataset(rows_per_episode=20, episodes=1, fps=2.0, **settings):
    total = rows_per_episode * episodes
    episode_index = np.repeat(np.arange(episodes), rows_per_episode)
    frame_index = np.tile(np.arange(rows_per_episode), episodes)
    dataset = PatchWAMParquetDataset.__new__(PatchWAMParquetDataset)
    dataset.dataset = Dataset.from_dict({
        'episode_index': episode_index.tolist(),
        'frame_index': frame_index.tolist(),
        'task_index': [0] * total,
        'timestamp': ((frame_index / fps) if fps > 0 else frame_index).tolist(),
        'action': [[float(i)] for i in range(total)],
    })
    dataset.dataset_cumulative_sizes = np.array([0, total])
    dataset.sample_indices = np.arange(total)
    dataset.tasks = [[dict(task='move')]]
    dataset.info = [dict(fps=fps)]
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
    for name, value in settings.items():
        setattr(dataset, name, value)
    return dataset


def test_endpoint_is_start_plus_horizon_not_the_next_row():
    dataset = _make_dataset()
    sample = dataset.__getitem__(0, {'test': {}})
    assert sample['patchwam_row'] == 0
    assert sample['patchwam_future_row'] == 16
    assert sample['patchwam_future_valid'] is True
    assert sample['patchwam_frame_index'] == 0
    assert sample['patchwam_reset'] is True
    # 16 action targets spent the rows [0, 16); the future endpoint is 16.
    np.testing.assert_array_equal(sample['actions'][:, 0], np.arange(16))
    assert np.count_nonzero(sample['action_masks']) == 16


def test_endpoint_is_invalid_past_the_episode_tail():
    dataset = _make_dataset(rows_per_episode=20)
    sample = dataset.__getitem__(4, {'test': {}})
    assert sample['patchwam_future_row'] == 4
    assert sample['patchwam_future_valid'] is False


def test_endpoint_never_crosses_an_episode_boundary():
    dataset = _make_dataset(rows_per_episode=20, episodes=2)
    sample = dataset.__getitem__(5, {'test': {}})
    # Row 21 belongs to episode 1; the window must stay inside episode 0.
    assert sample['patchwam_future_row'] == 5
    assert sample['patchwam_future_valid'] is False


def test_history_rows_follow_the_original_clock_and_episode():
    from fluxvla.models.third_party_models.patchwam.data.history import \
        PastFrameSelector
    dataset = _make_dataset(rows_per_episode=20, episodes=2)
    dataset.history_enabled = True
    dataset.history_selector = PastFrameSelector(slots=3, interval_s=1.0,
                                                 jitter_s=0.0)
    dataset.training = False
    sample = dataset.__getitem__(5, {'test': {}})
    np.testing.assert_array_equal(sample['patchwam_history_rows'], [3, 1, 0])
    np.testing.assert_array_equal(sample['patchwam_history_valid'],
                                  [True, True, False])
    # The first frame of a new episode has no in-episode history.
    sample = dataset.__getitem__(20, {'test': {}})
    assert not sample['patchwam_history_valid'].any()
    assert sample['patchwam_reset'] is True


def test_history_dropout_only_applies_in_training_mode():
    from fluxvla.models.third_party_models.patchwam.data.history import \
        PastFrameSelector
    dataset = _make_dataset(rows_per_episode=20)
    dataset.history_enabled = True
    dataset.history_selector = PastFrameSelector(slots=2, interval_s=1.0,
                                                 jitter_s=0.0,
                                                 whole_dropout=0.0,
                                                 slot_dropout=0.0)
    dataset.training = False
    eval_sample = dataset.__getitem__(5, {'test': {}})
    np.testing.assert_array_equal(eval_sample['patchwam_history_valid'], [True, True])
    dataset.training = True
    train_sample = dataset.__getitem__(5, {'test': {}})
    assert train_sample['patchwam_augment'] is True
    assert eval_sample['patchwam_augment'] is False


def test_history_requires_positive_fps():
    from fluxvla.models.third_party_models.patchwam.data.history import \
        PastFrameSelector
    dataset = _make_dataset(fps=0.0)
    dataset.history_enabled = True
    dataset.history_selector = PastFrameSelector(slots=2)
    with pytest.raises(ValueError, match='positive dataset fps'):
        dataset.__getitem__(0, {'test': {}})


def test_vl_history_has_its_own_contract():
    from fluxvla.models.third_party_models.patchwam.data.history import \
        PastFrameSelector
    dataset = _make_dataset(rows_per_episode=20)
    dataset.history_enabled = True
    dataset.history_selector = PastFrameSelector(slots=2, interval_s=1.0,
                                                 jitter_s=0.4,
                                                 vl_jitter_frames=1)
    dataset.training = False
    sample = dataset.__getitem__(5, {'test': {}})
    # Without training jitter both contracts agree on the original clock.
    np.testing.assert_array_equal(sample['patchwam_vl_history_rows'],
                                  sample['patchwam_history_rows'])
    np.testing.assert_array_equal(sample['patchwam_vl_history_valid'],
                                  sample['patchwam_history_valid'])


def test_vl_history_requires_history_enabled():
    dataset = _make_dataset()
    sample = dataset.__getitem__(2, {'test': {}})
    assert 'patchwam_vl_history_rows' not in sample


def test_base_pre_transform_hook_is_a_noop():
    data = {'actions': np.zeros((2, 1))}
    assert ParquetDataset._pre_transform_hook(None, data, 3, 0) is data


def test_history_rows_never_cross_two_data_roots():
    from datasets import Dataset
    from fluxvla.models.third_party_models.patchwam.data.history import \
        PastFrameSelector
    total = 40
    dataset = PatchWAMParquetDataset.__new__(PatchWAMParquetDataset)
    dataset.dataset = Dataset.from_dict({
        'episode_index': [0] * 20 + [1] * 20,
        'frame_index': list(range(20)) + list(range(20)),
        'task_index': [0] * total,
        'timestamp': [float(i) for i in range(20)] * 2,
        'action': [[float(i)] for i in range(total)],
    })
    dataset.dataset_cumulative_sizes = np.array([0, 20, 40])
    dataset.sample_indices = np.arange(total)
    dataset.tasks = [[dict(task='move')], [dict(task='move')]]
    dataset.info = [dict(fps=1.0), dict(fps=1.0)]
    dataset.data_root_path = ['root0', 'root1']
    dataset.statistic_name = 'test'
    dataset.transforms = []
    dataset.action_key = 'action'
    dataset.action_dtype = np.dtype('float32')
    dataset.use_delta = False
    dataset.expose_index = False
    dataset.supervise_terminal_padding = False
    dataset.require_full_window = False
    dataset.frame_window_size = 1
    dataset.frame_sample_stride = 1
    dataset.window_start_idx = 0
    dataset.action_window_size = 1
    dataset.future_offset = 1
    dataset.horizon = 1
    dataset.history_enabled = True
    dataset.history_selector = PastFrameSelector(slots=5, interval_s=1.0,
                                                 jitter_s=0.0)
    dataset.training = False
    sample = dataset.__getitem__(25, {'test': {}})
    rows = sample['patchwam_history_rows']
    # The last five frames of root1, never crossing into root0.
    np.testing.assert_array_equal(rows, [24, 23, 22, 21, 20])
    assert sample['patchwam_history_valid'].all()
