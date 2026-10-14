"""CPU units for the opt-in PatchWAM non-idle row filter.

The source ``EpisodeDataset`` drops non-idle rows per episode from a JSON
mapping (top-level ``episode-index -> ranges`` or ``{"episodes": ...}``), keeps
the full episode when its key is absent, and keeps no rows for an explicit
``[]``. The filtered sequence is what the action window and future endpoint
walk, while history still resolves on the original recording clock, including
rows the filter removed. These units build real on-disk LeRobot v2 roots and
drive the real dataset construction path.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.datasets.patchwam_dataset import PatchWAMParquetDataset
from fluxvla.transforms.transform_patchwam import (PatchWAMEndpoints,
                                                   PatchWAMSelectHistory,
                                                   PatchWAMSelectWindow)

ROWS = 30
FPS = 1.0


def _write_root(path: Path, *, episodes=2, rows=ROWS, action_base=0,
                tasks=None, episode_tasks=None) -> Path:
    """Write a minimal LeRobot v2 root with native metadata statistics."""
    import pandas as pd

    meta = path / 'meta'
    (path / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    (meta / 'info.json').write_text(
        json.dumps({
            'codebase_version': 'v2.1',
            'fps': FPS,
            'chunks_size': 1000,
            'total_episodes': episodes,
            'video_path': ('videos/chunk-{episode_chunk:03d}/{video_key}/'
                           'episode_{episode_index:06d}.mp4'),
        }),
        encoding='utf-8')
    task_rows = tasks or [dict(task_index=0, task='move the cup')]
    (meta / 'tasks.jsonl').write_text(
        '\n'.join(json.dumps(row) for row in task_rows) + '\n',
        encoding='utf-8')
    episode_names = episode_tasks or [row['task'] for row in task_rows][:1]
    with (meta / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(episodes):
            handle.write(
                json.dumps({
                    'episode_index': episode,
                    'tasks': [episode_names[episode % len(episode_names)]],
                    'length': rows,
                }) + '\n')
    with (meta / 'episodes_stats.jsonl').open('w', encoding='utf-8') as handle:
        for episode in range(episodes):
            handle.write(
                json.dumps({
                    'episode_index': episode,
                    'stats': {
                        'observation.state': {
                            'min': [0.0], 'max': [float(rows - 1)],
                            'mean': [0.0], 'std': [1.0]},
                        'action': {
                            'min': [0.0], 'max': [float(rows - 1)],
                            'mean': [0.0], 'std': [1.0]},
                        'timestamp': {
                            'min': [0.0], 'max': [float(rows - 1)],
                            'mean': [0.0], 'std': [1.0]},
                    },
                }) + '\n')
    rows_data = {'observation.state': [], 'action': [], 'timestamp': [],
                 'frame_index': [], 'episode_index': [], 'task_index': []}
    for episode in range(episodes):
        for frame in range(rows):
            rows_data['observation.state'].append([float(frame)])
            rows_data['action'].append([float(action_base + episode * 1000
                                              + frame)])
            rows_data['timestamp'].append(frame / FPS)
            rows_data['frame_index'].append(frame)
            rows_data['episode_index'].append(episode)
            rows_data['task_index'].append(0)
    pd.DataFrame(rows_data).to_parquet(
        path / 'data' / 'chunk-000' / 'episode_000000.parquet')
    return path


def _write_filter(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


def _dataset(roots, *, nonidle_filter_path=None, require_nonidle_filter=False,
             history=None, training=True, task_selection=None, **overrides):
    settings = dict(
        data_root_path=[str(root) for root in roots],
        transforms=[],
        action_window_size=16,
        window_start_idx=0,
        frame_window_size=1,
        frame_sample_stride=1,
        future_offset=16,
        horizon=16,
        action_key='action',
        statistic_name='test',
        training=training,
        history=history,
        nonidle_filter_path=nonidle_filter_path,
        require_nonidle_filter=require_nonidle_filter,
        task_selection=task_selection,
    )
    settings.update(overrides)
    return PatchWAMParquetDataset(**settings)


def _sample(dataset, index):
    return dataset.__getitem__(index, {'test': {}})


def _frame_indices(dataset):
    return list(dataset.dataset['frame_index'])


def _episode_indices(dataset):
    return list(dataset.dataset['episode_index'])


def _decoded_rows(rows):
    """One decoded frame per original row, filled with ``row + 1``."""
    base = torch.arange(1, rows + 1, dtype=torch.float32)
    return base.reshape(rows, 1, 1, 1).expand(rows, 3, 2, 2).clone()


def _gather_history(dataset, index):
    sample = _sample(dataset, index)
    gathered = PatchWAMSelectHistory()(dict(
        decoded_frames=_decoded_rows(int(sample['patchwam_raw_total_rows'])),
        patchwam_raw_history_rows=sample['patchwam_raw_history_rows'],
        patchwam_history_valid=sample['patchwam_history_valid'],
        patchwam_nonidle_filtered=sample['patchwam_nonidle_filtered'],
        patchwam_raw_total_rows=sample['patchwam_raw_total_rows']))
    return sample, gathered


def test_ranges_are_clipped_sorted_and_deduplicated(tmp_path):
    root = _write_root(tmp_path / 'root', episodes=1)
    # [10, 20) overlaps [12, 26); [26, 99) clips past the 30-row episode.
    payload = {'episodes': {'0': [[10, 20], [26, 99], [12, 26]], '1': []}}
    dataset = _dataset([root],
                       nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                                        payload))
    assert _episode_indices(dataset) == [0] * 20
    assert _frame_indices(dataset) == list(range(10, 30))
    assert len(dataset.sample_indices) == 20
    sample = _sample(dataset, 0)
    assert sample['patchwam_frame_index'] == 10
    # Dataset row stays 0; the original-recording row is 10.
    assert sample['patchwam_row'] == 0
    assert sample['patchwam_raw_row'] == 10
    assert sample['patchwam_nonidle_filtered'] is True
    assert sample['patchwam_raw_total_rows'] == ROWS


def test_window_and_endpoint_progress_in_the_filtered_sequence(tmp_path):
    root = _write_root(tmp_path / 'root', episodes=1)
    payload = {'0': [[0, 5], [10, 30]]}
    dataset = _dataset([root],
                       nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                                        payload))
    filtered = _frame_indices(dataset)
    assert filtered == [0, 1, 2, 3, 4] + list(range(10, 30))
    sample = _sample(dataset, 4)
    # The window walks the filtered sequence: it steps across the gap instead
    # of treating removed rows as visited positions.
    assert sample['patchwam_frame_index'] == 4
    np.testing.assert_array_equal(sample['patchwam_window_rows'],
                                  np.arange(4, 21))
    np.testing.assert_array_equal(sample['patchwam_raw_window_rows'],
                                  [4] + list(range(10, 26)))
    assert sample['patchwam_future_row'] == 20
    assert filtered[sample['patchwam_future_row']] == 25
    assert sample['patchwam_raw_future_row'] == 25
    assert sample['patchwam_future_valid'] is True
    np.testing.assert_array_equal(
        sample['actions'][:, 0], [float(filtered[position])
                                  for position in range(4, 20)])


def test_tail_masks_match_the_unfiltered_contract(tmp_path):
    root = _write_root(tmp_path / 'root', episodes=1)
    identity = _write_filter(tmp_path / 'identity.json',
                             {'0': [[0, ROWS]]})
    plain = _dataset([root])
    filtered = _dataset([root], nonidle_filter_path=identity)
    assert _frame_indices(plain) == _frame_indices(filtered)
    for index in (0, 5, ROWS - 2):
        baseline = _sample(plain, index)
        narrowed = _sample(filtered, index)
        np.testing.assert_array_equal(baseline['patchwam_window_valid'],
                                      narrowed['patchwam_window_valid'])
        np.testing.assert_array_equal(baseline['action_masks'],
                                      narrowed['action_masks'])
        assert baseline['patchwam_future_valid'] == narrowed[
            'patchwam_future_valid']
    # The native tail contract is unchanged: only the final rows pad.
    tail = _sample(filtered, ROWS - 2)
    assert not tail['patchwam_window_valid'][-2:].any()
    assert tail['patchwam_future_valid'] is False


def test_missing_episode_key_keeps_the_full_episode(tmp_path):
    root = _write_root(tmp_path / 'root')
    dataset = _dataset(
        [root],
        nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                         {'0': [[10, 20]]}))
    # Episode 1 has no key: every row is retained.
    assert _episode_indices(dataset) == [0] * 10 + [1] * ROWS
    assert _frame_indices(dataset) == list(range(10, 20)) + list(
        range(ROWS))


def test_explicit_empty_ranges_remove_every_row_of_that_episode(tmp_path):
    root = _write_root(tmp_path / 'root')
    dataset = _dataset([root],
                       nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                                        {'1': []}))
    assert set(_episode_indices(dataset)) == {0}
    assert len(dataset.sample_indices) == ROWS


def test_per_root_files_keep_repeated_episode_ids_apart(tmp_path):
    first = _write_root(tmp_path / 'first', action_base=0)
    second = _write_root(tmp_path / 'second', action_base=5000)
    filters = {
        str(first): _write_filter(tmp_path / 'first.json', {'0': [[5, 15]]}),
        str(second): _write_filter(tmp_path / 'second.json', {'1': []}),
    }
    dataset = _dataset([first, second], nonidle_filter_path=filters)
    assert list(dataset.dataset_cumulative_sizes) == [0, 40, 70]
    # Root 0 keeps episode 0 [5, 15) and all of episode 1.
    assert _episode_indices(dataset)[:40] == [0] * 10 + [1] * ROWS
    # Root 1 drops its own episode 1; root 0's identically named episode stays.
    assert _episode_indices(dataset)[40:] == [0] * ROWS
    assert [sample['action'][0] for sample in
            (dataset.dataset[0], dataset.dataset[40])] == [5.0, 5000.0]


def test_a_single_shared_filter_applies_to_every_root(tmp_path):
    first = _write_root(tmp_path / 'first')
    second = _write_root(tmp_path / 'second')
    shared = _write_filter(tmp_path / 'shared.json',
                           {'episodes': {'0': [[1, 6]], '1': [[0, 4]]}})
    dataset = _dataset([first, second], nonidle_filter_path=shared)
    assert list(dataset.dataset_cumulative_sizes) == [0, 9, 18]
    assert _episode_indices(dataset) == [0] * 5 + [1] * 4 + [0] * 5 + [1] * 4


def test_root_absent_from_the_mapping_fails_like_the_source(tmp_path):
    first = _write_root(tmp_path / 'first')
    second = _write_root(tmp_path / 'second')
    filters = {
        str(first): _write_filter(tmp_path / 'first.json', {'0': [[0, 5]]}),
    }
    with pytest.raises(KeyError, match='No nonidle filter configured'):
        _dataset([first, second], nonidle_filter_path=filters)


def test_invalid_filter_inputs_fail_at_dataset_construction(tmp_path):
    root = _write_root(tmp_path / 'root')
    with pytest.raises(FileNotFoundError,
                       match='Nonidle filter file not found'):
        _dataset([root],
                 nonidle_filter_path=str(tmp_path / 'missing.json'))
    malformed = tmp_path / 'malformed.json'
    malformed.write_text('{not json', encoding='utf-8')
    with pytest.raises(ValueError, match='not valid JSON'):
        _dataset([root], nonidle_filter_path=malformed)
    bad_ranges = _write_filter(tmp_path / 'bad.json',
                               {'episodes': {'0': [[1]]}})
    with pytest.raises(ValueError, match='start/end pair'):
        _dataset([root], nonidle_filter_path=bad_ranges)
    bad_shape = _write_filter(tmp_path / 'shape.json', {'episodes': [1, 2]})
    with pytest.raises(ValueError, match='"episodes" must be a mapping'):
        _dataset([root], nonidle_filter_path=bad_shape)
    with pytest.raises(ValueError, match='require_nonidle_filter'):
        _dataset([root], require_nonidle_filter=True)
    with pytest.raises(ValueError, match='nonidle_filter_path'):
        _dataset([root], nonidle_filter_path=1234)


def test_a_filter_that_removes_every_row_fails_loudly(tmp_path):
    root = _write_root(tmp_path / 'root')
    payload = {'0': [], '1': []}
    with pytest.raises(ValueError, match='no rows'):
        _dataset([root],
                 nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                                  payload))


def test_an_empty_root_among_several_names_the_root(tmp_path):
    first = _write_root(tmp_path / 'first', episodes=1)
    second = _write_root(tmp_path / 'second', episodes=1)
    filters = {
        str(first): _write_filter(tmp_path / 'first.json', {'0': []}),
        str(second): _write_filter(tmp_path / 'second.json', {'0': [[0, 5]]}),
    }
    with pytest.raises(ValueError, match='first'):
        _dataset([first, second], nonidle_filter_path=filters)


def test_history_keeps_the_original_clock_including_filtered_out_rows(
        tmp_path):
    root = _write_root(tmp_path / 'root', episodes=1)
    # Frames 0-9 are removed from the sampling sequence.
    dataset = _dataset(
        [root],
        nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                          {'0': [[10, 30]]}),
        history=dict(slots=3, interval_s=1.0, jitter_s=0.0, whole_dropout=0.0,
                     slot_dropout=0.0),
        training=False)
    sample = _sample(dataset, 0)
    assert sample['patchwam_frame_index'] == 10
    # History resolves on the raw recording clock: frames 9, 8 and 7 exist in
    # the recording even though the filter removed them from the window axis.
    np.testing.assert_array_equal(sample['patchwam_history_timestamps'],
                                  [9.0, 8.0, 7.0])
    assert sample['patchwam_history_valid'].all()
    rows = sample['patchwam_raw_history_rows']
    np.testing.assert_array_equal(rows, [9, 8, 7])
    assert rows.min() >= 0 and rows.max() < sample['patchwam_raw_total_rows']
    # No approximate dataset-axis ids are advertised for filtered history.
    assert 'patchwam_history_rows' not in sample


def test_task_selection_cannot_reintroduce_filtered_rows(tmp_path):
    first = _write_root(
        tmp_path / 'first', episode_tasks=['pick', 'place'],
        tasks=[dict(task_index=0, task='pick'), dict(task_index=1,
                                                     task='place')])
    second = _write_root(tmp_path / 'second', episodes=1,
                         tasks=[dict(task_index=0, task='wipe')],
                         action_base=5000)
    filters = {
        str(first): _write_filter(tmp_path / 'first.json', {'1': []}),
        str(second): _write_filter(tmp_path / 'second.json',
                                   {'0': [[0, ROWS]]}),
    }
    dataset = _dataset(
        [first, second], nonidle_filter_path=filters,
        task_selection={'tasks': ['place', 'wipe']})
    # 'place' is selected but the filter removed every one of its rows, so the
    # sampled indexes come only from root 1's 'wipe' episode.
    assert _episode_indices(dataset) == [0] * ROWS + [0] * ROWS
    assert len(dataset.sample_indices) == ROWS
    assert dataset.sample_indices.min() == ROWS
    sample = _sample(dataset, 0)
    assert sample['patchwam_row'] == ROWS
    assert sample['actions'][0, 0] == 5000.0


def test_mixed_roots_keep_unfiltered_history_on_the_original_clock(tmp_path):
    narrowed = _write_root(tmp_path / 'narrowed', action_base=0)
    plain = _write_root(tmp_path / 'plain', action_base=5000)
    filters = {
        str(narrowed): _write_filter(tmp_path / 'narrowed.json',
                                     {'0': [[10, 30]], '1': []}),
        str(plain): _write_filter(tmp_path / 'plain.json',
                                  {'0': [[0, ROWS]], '1': [[0, ROWS]]}),
    }
    dataset = _dataset(
        [narrowed, plain], nonidle_filter_path=filters,
        history=dict(slots=2, interval_s=1.0, jitter_s=0.0, whole_dropout=0.0,
                     slot_dropout=0.0),
        training=False)
    # Root 0 keeps 20 rows (frames 10-29), so root 1 starts at dataset row 20.
    assert list(dataset.dataset_cumulative_sizes) == [0, 20, 80]
    # A filtered root reads history from its own raw recording.
    narrowed_sample = _sample(dataset, 0)
    np.testing.assert_array_equal(
        narrowed_sample['patchwam_history_timestamps'], [9.0, 8.0])
    assert narrowed_sample['patchwam_history_valid'].all()
    # An unfiltered root beside it still resolves on its own clock and never
    # borrows rows or timestamps from the filtered root.
    plain_sample = _sample(dataset, 20)
    assert plain_sample['patchwam_episode_index'] == 0
    assert plain_sample['patchwam_frame_index'] == 0
    assert not plain_sample['patchwam_history_valid'].any()
    later = _sample(dataset, 25)
    assert later['patchwam_frame_index'] == 5
    np.testing.assert_array_equal(later['patchwam_history_timestamps'],
                                  [4.0, 3.0])
    assert later['patchwam_history_valid'].all()
    # Root 1 is filtered too, so only the original-row keys are advertised.
    np.testing.assert_array_equal(later['patchwam_raw_history_rows'],
                                  [2 * ROWS + 4, 2 * ROWS + 3])
    assert later['patchwam_raw_row'] == 2 * ROWS + 5
    assert 'patchwam_history_rows' not in later
    assert later['actions'][0, 0] == 5005.0
    # The filtered root's own window never reaches into the unfiltered root.
    tail = _sample(dataset, 19)
    assert tail['patchwam_frame_index'] == 29
    assert tail['patchwam_future_valid'] is False


def test_the_base_dataset_is_unfiltered_by_default(tmp_path):
    root = _write_root(tmp_path / 'root')
    dataset = ParquetDataset(
        data_root_path=[str(root)], transforms=[], action_window_size=4,
        statistic_name='test')
    assert len(dataset.sample_indices) == 2 * ROWS
    assert ParquetDataset._select_root_rows(
        dataset, 0, str(root), dataset.dataset) is dataset.dataset
    assert not hasattr(dataset, 'nonidle_filter_path')


def test_select_history_gathers_original_rows_not_retained_rows(tmp_path):
    """The rows path must see the recording, not the filtered sequence."""
    root = _write_root(tmp_path / 'root', episodes=1)
    dataset = _dataset(
        [root],
        nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                          {'0': [[10, 30]]}),
        history=dict(slots=3, interval_s=1.0, jitter_s=0.0, whole_dropout=0.0,
                     slot_dropout=0.0),
        training=False)
    sample, gathered = _gather_history(dataset, 0)
    assert sample['patchwam_nonidle_filtered'] is True
    assert sample['patchwam_raw_total_rows'] == ROWS
    assert sample['patchwam_raw_row'] == 10
    # Original-clock history slots, not the retained sequence start.
    np.testing.assert_array_equal(sample['patchwam_raw_history_rows'],
                                  [9, 8, 7])
    assert gathered['patchwam_history_valid'].all()
    torch.testing.assert_close(
        gathered['history_frames'][:, 0, 0, 0],
        torch.tensor([10.0, 9.0, 8.0]))
    # The retained-start row (10) would have gathered 11, 12, 13.
    assert not torch.allclose(
        gathered['history_frames'][:, 0, 0, 0], torch.tensor([11.0, 12.0,
                                                              13.0]))


def test_select_window_walks_the_gap_from_the_original_row_tensor(tmp_path):
    root = _write_root(tmp_path / 'root', episodes=1)
    dataset = _dataset(
        [root],
        nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                          {'0': [[0, 5], [10, 30]]}))
    sample = _sample(dataset, 4)
    expected = np.concatenate([[4], np.arange(10, 26)])
    # Dataset-axis ids are filtered-sequence positions; the original-row ids
    # step across the removed gap.
    np.testing.assert_array_equal(sample['patchwam_window_rows'],
                                  np.arange(4, 21))
    np.testing.assert_array_equal(sample['patchwam_raw_window_rows'], expected)
    assert sample['patchwam_future_row'] == 20
    assert sample['patchwam_raw_future_row'] == 25
    assert sample['patchwam_future_valid'] is True
    selected = PatchWAMSelectWindow(horizon=16)(dict(
        decoded_frames=_decoded_rows(ROWS),
        patchwam_raw_window_rows=sample['patchwam_raw_window_rows'],
        patchwam_window_valid=sample['patchwam_window_valid'],
        patchwam_nonidle_filtered=sample['patchwam_nonidle_filtered'],
        patchwam_raw_total_rows=sample['patchwam_raw_total_rows']))
    torch.testing.assert_close(
        selected['world_frames'][0, :, 0, 0, 0],
        torch.tensor(
            sample['patchwam_raw_window_rows'] + 1, dtype=torch.float32))
    endpoints = PatchWAMEndpoints(horizon=16)(selected)
    assert endpoints['frame_masks'].tolist() == [True, True]
    torch.testing.assert_close(endpoints['world_endpoints'][0, 1, 0, 0, 0],
                               torch.tensor(26.0))


def test_select_window_keeps_the_native_tail_rows(tmp_path):
    root = _write_root(tmp_path / 'root', episodes=1)
    dataset = _dataset(
        [root],
        nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                          {'0': [[10, 30]]}))
    sample = _sample(dataset, 19)
    assert sample['patchwam_frame_index'] == 29
    assert not bool(sample['patchwam_window_valid'][-1])
    # Dataset-axis ids repeat the last valid sampled row; the original-row key
    # repeats the last valid recording row.
    np.testing.assert_array_equal(sample['patchwam_window_rows'], [19] * 17)
    np.testing.assert_array_equal(sample['patchwam_raw_window_rows'],
                                  [29] * 17)
    selected = PatchWAMSelectWindow(horizon=16)(dict(
        decoded_frames=_decoded_rows(ROWS),
        patchwam_raw_window_rows=sample['patchwam_raw_window_rows'],
        patchwam_window_valid=sample['patchwam_window_valid'],
        patchwam_nonidle_filtered=True,
        patchwam_raw_total_rows=ROWS))
    # The tail repeats the last valid original row and stays flagged invalid.
    torch.testing.assert_close(
        selected['world_frames'][0, :, 0, 0, 0],
        torch.full((17,), 30.0))
    assert selected['world_window_valid'][-2:].tolist() == [False, False]


def test_multiroot_select_history_preserves_raw_root_offsets(tmp_path):
    first = _write_root(tmp_path / 'first', action_base=0)
    second = _write_root(tmp_path / 'second', action_base=5000)
    filters = {
        str(first): _write_filter(tmp_path / 'first.json',
                                  {'0': [[10, 30]], '1': []}),
        str(second): _write_filter(tmp_path / 'second.json',
                                   {'0': [[0, ROWS]], '1': [[0, ROWS]]}),
    }
    dataset = _dataset(
        [first, second], nonidle_filter_path=filters,
        history=dict(slots=2, interval_s=1.0, jitter_s=0.0, whole_dropout=0.0,
                     slot_dropout=0.0),
        training=False)
    assert list(dataset.dataset_cumulative_sizes) == [0, 20, 80]
    # Root 0 keeps frames 10..29 of a 2x30 recording, so root 1 (all rows
    # retained) starts at original row 60.
    second_offset = 2 * ROWS
    first_sample, first_gathered = _gather_history(dataset, 0)
    assert first_sample['patchwam_raw_total_rows'] == 2 * second_offset
    assert first_sample['patchwam_raw_row'] == 10
    np.testing.assert_array_equal(first_sample['patchwam_raw_history_rows'],
                                  [9, 8])
    assert (first_sample['patchwam_raw_window_rows'] < second_offset).all()
    torch.testing.assert_close(
        first_gathered['history_frames'][:, 0, 0, 0],
        torch.tensor([10.0, 9.0]))
    # Root 1 keeps everything; its ids stay inside its own original-row range.
    later, later_gathered = _gather_history(dataset, 25)
    assert later['patchwam_raw_row'] == second_offset + 5
    np.testing.assert_array_equal(later['patchwam_raw_history_rows'],
                                  [second_offset + 4, second_offset + 3])
    assert (later['patchwam_raw_window_rows'] >= second_offset).all()
    torch.testing.assert_close(
        later_gathered['history_frames'][:, 0, 0, 0],
        torch.tensor([float(second_offset + 5), float(second_offset + 4)]))


def test_the_row_path_fails_loudly_on_a_filtered_tensor(tmp_path):
    root = _write_root(tmp_path / 'root', episodes=1)
    dataset = _dataset(
        [root],
        nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                          {'0': [[10, 30]]}))
    sample = _sample(dataset, 0)
    filtered_only = _decoded_rows(20)
    with pytest.raises(ValueError, match='original-row'):
        PatchWAMSelectWindow(horizon=16)(dict(
            decoded_frames=filtered_only,
            patchwam_raw_window_rows=sample['patchwam_raw_window_rows'],
            patchwam_window_valid=sample['patchwam_window_valid'],
            patchwam_nonidle_filtered=True,
            patchwam_raw_total_rows=sample['patchwam_raw_total_rows']))
    with pytest.raises(ValueError, match='original-row'):
        PatchWAMSelectHistory()(dict(
            decoded_frames=filtered_only,
            patchwam_raw_history_rows=torch.tensor([9, 8, 7]),
            patchwam_history_valid=torch.tensor([True, True, True]),
            patchwam_nonidle_filtered=True,
            patchwam_raw_total_rows=sample['patchwam_raw_total_rows']))


def test_a_recording_axis_that_is_not_contiguous_zero_based_fails(tmp_path):
    import pandas as pd

    root = _write_root(tmp_path / 'root', episodes=1)
    path = root / 'data' / 'chunk-000' / 'episode_000000.parquet'
    frame = pd.read_parquet(path)
    frame['frame_index'] = frame['frame_index'] + 1
    frame.to_parquet(path)
    with pytest.raises(ValueError, match='contiguous 0-based recording axis'):
        _dataset([root],
                 nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                                  {'0': [[0, 5]]}))


def test_many_episodes_keep_only_their_filtered_rows(tmp_path):
    root = _write_root(tmp_path / 'root', episodes=40, rows=5)
    payload = {str(episode): [[1, 3]] for episode in range(40)}
    dataset = _dataset([root],
                       nonidle_filter_path=_write_filter(tmp_path / 'f.json',
                                                        payload))
    assert list(dataset.dataset_cumulative_sizes) == [0, 80]
    assert _frame_indices(dataset) == [1, 2] * 40
    assert len(dataset.sample_indices) == 80


def test_unfiltered_multi_root_raw_rows_match_the_dataset_axis(tmp_path):
    """Without any filter the raw axis must stay the dataset axis per root."""
    first = _write_root(tmp_path / 'first', episodes=1, rows=20, action_base=0)
    second = _write_root(tmp_path / 'second', episodes=1, rows=20,
                         action_base=5000)
    dataset = _dataset(
        [first, second],
        history=dict(slots=2, interval_s=1.0, jitter_s=0.0, whole_dropout=0.0,
                     slot_dropout=0.0),
        training=False)
    assert list(dataset.dataset_cumulative_sizes) == [0, 20, 40]
    sample = _sample(dataset, 20)
    assert sample['patchwam_nonidle_filtered'] is False
    assert sample['patchwam_raw_total_rows'] == 40
    assert sample['patchwam_row'] == 20
    # Root 1 starts at row 20 on both axes; adding the filtered base again
    # would report 40 and push the window out of the tensor.
    assert sample['patchwam_raw_row'] == 20
    np.testing.assert_array_equal(sample['patchwam_raw_window_rows'],
                                  np.arange(20, 37))
    np.testing.assert_array_equal(sample['patchwam_window_rows'],
                                  np.arange(20, 37))
    assert sample['patchwam_raw_future_row'] == 36
    selected = PatchWAMSelectWindow(horizon=16)(dict(
        decoded_frames=_decoded_rows(40),
        patchwam_raw_window_rows=sample['patchwam_raw_window_rows'],
        patchwam_window_valid=sample['patchwam_window_valid'],
        patchwam_nonidle_filtered=sample['patchwam_nonidle_filtered'],
        patchwam_raw_total_rows=sample['patchwam_raw_total_rows']))
    torch.testing.assert_close(
        selected['world_frames'][0, :, 0, 0, 0],
        torch.tensor(np.arange(20, 37) + 1, dtype=torch.float32))
    endpoints = PatchWAMEndpoints(horizon=16)(selected)
    assert endpoints['frame_masks'].tolist() == [True, True]
    assert sample['actions'][0, 0] == 5000.0
    # The first root keeps its own identity mapping too.
    first_sample = _sample(dataset, 0)
    assert first_sample['patchwam_raw_row'] == 0
    np.testing.assert_array_equal(first_sample['patchwam_raw_window_rows'],
                                  np.arange(0, 17))
    # History keeps the same identity mapping on the unfiltered path, and the
    # dataset-axis keys stay available for both roots.
    later = _sample(dataset, 25)
    np.testing.assert_array_equal(later['patchwam_history_rows'], [24, 23])
    np.testing.assert_array_equal(later['patchwam_raw_history_rows'], [24, 23])
    np.testing.assert_array_equal(later['patchwam_raw_vl_history_rows'],
                                  [24, 23])
    np.testing.assert_array_equal(later['patchwam_history_timestamps'],
                                  [4.0, 3.0])
    assert later['patchwam_history_valid'].all()
    gathered = PatchWAMSelectHistory()(dict(
        decoded_frames=_decoded_rows(40),
        patchwam_history_rows=later['patchwam_history_rows'],
        patchwam_history_valid=later['patchwam_history_valid'],
        patchwam_nonidle_filtered=later['patchwam_nonidle_filtered'],
        patchwam_raw_total_rows=later['patchwam_raw_total_rows']))
    torch.testing.assert_close(gathered['history_frames'][:, 0, 0, 0],
                               torch.tensor([25.0, 24.0]))
    vl_gathered = PatchWAMSelectHistory(
        rows_key='patchwam_vl_history_rows',
        valid_key='patchwam_vl_history_valid')(dict(
            decoded_frames=_decoded_rows(40),
            patchwam_vl_history_rows=later['patchwam_vl_history_rows'],
            patchwam_vl_history_valid=later['patchwam_vl_history_valid'],
            patchwam_raw_total_rows=later['patchwam_raw_total_rows']))
    torch.testing.assert_close(vl_gathered['history_frames'][:, 0, 0, 0],
                               torch.tensor([25.0, 24.0]))
