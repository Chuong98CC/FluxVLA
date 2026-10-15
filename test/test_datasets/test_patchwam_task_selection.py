# Copyright 2026 Limx Dynamics
"""Native deterministic task filtering for the PatchWAM dataset."""

import json
from pathlib import Path

import numpy as np
import pytest

from fluxvla.datasets.parquet_dataset import ParquetDataset
from fluxvla.datasets.utils.task_selection import resolve_task_selection


def tasks():
    """Two roots that both number their tasks from zero."""
    return [
        [dict(task_index=0, task='pick'), dict(task_index=1, task='place')],
        [dict(task_index=0, task='wipe')],
    ]


def episodes():
    return [
        [dict(episode_index=0, tasks=['pick']),
         dict(episode_index=1, tasks=['pick']),
         dict(episode_index=2, tasks=['place'])],
        [dict(episode_index=0, tasks=['wipe']),
         dict(episode_index=1, tasks=['wipe'])],
    ]


def test_local_indices_are_root_local_and_names_are_the_identity():
    resolved = resolve_task_selection(tasks(), episodes(), {'tasks': ['wipe']})
    assert resolved['keep'] == {(1, 0), (1, 1)}
    assert resolved['names'] == ['wipe']
    report = resolved['selection']
    assert report['tasks'][0]['task_indices'] == {'1': [0]}
    assert report['ambiguous_indices'] == [0]
    with pytest.raises(ValueError, match='different tasks across roots'):
        resolve_task_selection(tasks(), episodes(), {'tasks': [0]})


def test_episode_cap_is_per_task_name_across_roots():
    shared = [[dict(task_index=0, task='sweep')] for _ in range(3)]
    root_episodes = [
        [dict(episode_index=0, tasks=['sweep']),
         dict(episode_index=1, tasks=['sweep'])] for _ in range(3)
    ]
    capped = resolve_task_selection(shared, root_episodes,
                                    {'tasks': ['sweep'],
                                     'episodes_per_task': 4, 'seed': 3})
    assert len(capped['keep']) == 4
    assert len({root for root, _ in capped['keep']}) >= 2
    uncapped = resolve_task_selection(shared, root_episodes,
                                      {'tasks': ['sweep']})
    assert len(uncapped['keep']) == 6


def test_seeded_num_tasks_and_explicit_selection_are_deterministic():
    first = resolve_task_selection(tasks(), episodes(),
                                   {'num_tasks': 2, 'seed': 7})
    repeat = resolve_task_selection(tasks(), episodes(),
                                    {'num_tasks': 2, 'seed': 7})
    assert first['names'] == repeat['names']
    assert first['selection'] == repeat['selection']
    assert first['selection']['available_tasks'] == 3
    by_index = resolve_task_selection(tasks(), episodes(), {'tasks': [1]})
    assert by_index['names'] == ['place']
    assert by_index['keep'] == {(0, 2)}


@pytest.mark.parametrize('specification,message', [
    ({'tasks': ['missing']}, 'absent from the dataset'),
    ({'tasks': [9]}, 'absent from the dataset'),
    ({'tasks': []}, 'nonempty list'),
    ({'num_tasks': 9}, 'distinct tasks'),
    ({'num_tasks': 0}, 'positive integer'),
    ({'num_tasks': 1, 'episodes_per_task': 0}, 'positive integer'),
    ({'num_tasks': 1, 'unknown': 1}, 'Unknown task selection options'),
])
def test_invalid_requests_fail_loudly(specification, message):
    with pytest.raises(ValueError, match=message):
        resolve_task_selection(tasks(), episodes(), specification)


def test_ambiguous_metadata_and_empty_subsets_fail():
    conflicting = [[dict(episode_index=0, tasks=['pick', 'place'])]]
    with pytest.raises(ValueError, match='exactly one task name'):
        resolve_task_selection([[dict(task_index=0, task='pick'),
                                 dict(task_index=1, task='place')]],
                               conflicting, {'tasks': ['pick']})
    with pytest.raises(ValueError, match='missing from tasks.jsonl'):
        resolve_task_selection([[dict(task_index=0, task='pick')]],
                               [[dict(episode_index=0, tasks=['ghost'])]],
                               {'tasks': ['pick']})
    with pytest.raises(ValueError, match='contain no episodes'):
        resolve_task_selection([[dict(task_index=0, task='pick'),
                                 dict(task_index=1, task='unused')]],
                               [[dict(episode_index=0, tasks=['pick'])]],
                               {'tasks': ['unused']})


def _index_dataset(episode_rows, root_sizes):
    dataset = ParquetDataset.__new__(ParquetDataset)
    dataset.dataset = {
        'episode_index': np.asarray(episode_rows, dtype=np.int64)
    }
    dataset.dataset_cumulative_sizes = np.cumsum([0] + list(root_sizes))
    dataset.full_length = len(episode_rows)
    return dataset


def test_sample_indices_respect_the_resolved_selection_before_splitting():
    # Root 0 owns rows 0-5 (episodes 0,0,0,1,1,1); root 1 owns rows 6-9.
    dataset = _index_dataset([0, 0, 0, 1, 1, 1, 0, 0, 1, 1], [6, 4])
    keep = {(0, 1)}
    indices = dataset._build_sample_indices(1.0, keep=keep)
    assert indices.tolist() == [3, 4, 5]
    # The episode-fraction split then applies inside the selected subset only.
    both = {(0, 0), (0, 1)}
    assert dataset._build_sample_indices(1.0, keep=both).tolist() == \
        [0, 1, 2, 3, 4, 5]
    assert dataset._build_sample_indices(0.5, keep=both).tolist() == [0, 1, 2]
    # Without a filter the legacy behavior is unchanged.
    assert dataset._build_sample_indices(1.0).tolist() == list(range(10))
    with pytest.raises(ValueError, match='No samples left'):
        dataset._build_sample_indices(1.0, keep={(1, 7)})


def test_index_style_episode_tasks_resolve_through_the_local_table():
    """LeRobot stores names; the native fixtures also store local indices."""
    resolved = resolve_task_selection(
        [[dict(task_index=0, task='move the cup')]],
        [[dict(episode_index=0, tasks=[0]),
          dict(episode_index=1, tasks=['move the cup'])]],
        {'tasks': ['move the cup']})
    assert resolved['keep'] == {(0, 0), (0, 1)}


def test_selection_metadata_is_persisted_for_wrapped_datasets(tmp_path):
    from fluxvla.datasets.utils.task_selection import (dataset_task_selection,
                                                       save_task_selection)

    selection = {
        'source': 'seeded',
        'num_tasks': 5,
        'tasks': [{'name': 'pick', 'episodes': [{'root_index': 0,
                                                 'episode_index': 1}]}],
    }

    class Inner:

        def selection_metadata(self):
            return json.loads(json.dumps(selection))

    class Wrapper:

        def __init__(self):
            self.datasets = Inner()

    wrapper = Wrapper()
    assert dataset_task_selection(wrapper) == selection
    assert save_task_selection(object(), tmp_path / 'plain') is None
    path = save_task_selection(wrapper, tmp_path / 'run')
    assert Path(path).name == 'dataset_selection.json'
    assert json.loads(Path(path).read_text()) == selection
    # Rewriting is atomic and leaves no temporary file behind.
    assert save_task_selection(wrapper, tmp_path / 'run') == path
    assert not list((tmp_path / 'run').glob('*.tmp'))


def test_predataset_statistics_are_rejected_for_a_task_selected_recipe():
    from fluxvla.datasets.utils.task_selection import \
        reject_predataset_statistics

    selected = {'datasets': dict(task_selection=dict(tasks=['pick']))}
    reject_predataset_statistics(selected)
    for key in ('auto_compute_statistics', 'dataset_statistics_path',
                'dataset_statistics'):
        with pytest.raises(ValueError, match='cannot use'):
            reject_predataset_statistics(
                {'datasets': dict(task_selection=dict(tasks=['pick']),
                                  **{key: 'x'})})
    # Without a selection the pre-dataset statistics options stay usable.
    reject_predataset_statistics(
        {'datasets': dict(auto_compute_statistics=True)})


def release_episodes(count=6, variants=4):
    """One root whose episodes declare paraphrase instructions, not task names."""
    return [[dict(episode_index=index,
                  tasks=[f'paraphrase {index}.{variant}'
                         for variant in range(variants)])
             for index in range(count)]]


def release_paraphrases(count=6, variants=4):
    """The matching ``tasks.jsonl`` rows: every paraphrase is declared."""
    return [[dict(task_index=index * variants + variant,
                  task=f'paraphrase {index}.{variant}')
             for variant in range(variants)]
            for index in range(count)]


def test_episode_ranges_define_benchmark_identity_without_instruction_text():
    """Explicit ranges select episodes even when every episode is a paraphrase.

    The released RoboTwin 2.0 root stores dozens of instruction variants per
    episode, so exact-name identity would raise "exactly one task name". The
    map bypasses that resolution entirely.
    """
    selection = {'tasks': ['turn_switch', 'rotate_qrcode'],
                 'expected_episodes': 6,
                 'episode_ranges': {'turn_switch': [[4, 6]],
                                    'rotate_qrcode': [[0, 2]]}}
    resolved = resolve_task_selection([], release_episodes(), selection)
    assert resolved['names'] == ['turn_switch', 'rotate_qrcode']
    assert resolved['keep'] == {(0, 0), (0, 1), (0, 4), (0, 5)}
    report = resolved['selection']
    assert report['source'] == 'episode_ranges'
    assert report['episodes'] == 4
    assert report['root_episodes'] == 6
    assert report['tasks'][0]['name'] == 'turn_switch'
    assert report['tasks'][0]['episode_ranges'] == [[4, 6]]
    assert report['tasks'][0]['episode_count'] == 2
    assert report['tasks'][0]['episodes'] == [
        {'root_index': 0, 'episode_index': 4},
        {'root_index': 0, 'episode_index': 5}]
    with pytest.raises(ValueError, match='exactly one task name'):
        resolve_task_selection(release_paraphrases(), release_episodes(),
                               {'tasks': ['turn_switch']})


@pytest.mark.parametrize('specification,message', [
    ({'tasks': ['a'], 'episode_ranges': {'a': [[0, 1]], 'b': [[1, 2]]}},
     'must name exactly the requested tasks'),
    ({'tasks': ['a', 'b'], 'episode_ranges': {'a': [[0, 1]]}},
     'must name exactly the requested tasks'),
    ({'tasks': ['a'], 'episode_ranges': {'a': []}}, 'nonempty list'),
    ({'tasks': ['a'], 'episode_ranges': {'a': [[2, 2]]}}, '0 <= start < end'),
    ({'tasks': ['a', 'b'], 'episode_ranges': {'a': [[0, 2]], 'b': [[1, 3]]}},
     'overlaps another mapped range'),
    ({'tasks': ['a'], 'episode_ranges': {'a': [['x', 'y']]}},
     'must be two integers'),
    ({'tasks': ['a'], 'episode_ranges': {'a': [[0, 9]]}},
     'does not declare'),
    # A mistaken endpoint must fail the bound check, not allocate the range.
    ({'tasks': ['a'], 'episode_ranges': {'a': [[0, 10 ** 9]]}},
     'does not declare'),
    ({'tasks': ['a'], 'expected_episodes': 5, 'episode_ranges': {'a': [[0, 1]]}},
     'expected_episodes is 5'),
    ({'episode_ranges': {'a': [[0, 1]]}}, 'requires an explicit tasks list'),
    ({'tasks': ['a'], 'episode_ranges': {'a': [[0, 1]]}, 'num_tasks': 2},
     'does not take'),
    ({'tasks': ['a'], 'episode_ranges': {'a': [[0, 1]]}, 'episodes_per_task': 1},
     'does not take'),
    ({'tasks': ['a'], 'episode_ranges': {'a': [[0, 1]]}, 'seed': 1},
     'does not take'),
    ({'tasks': ['a'], 'expected_episodes': 4},
     'only applies to an episode_ranges selection'),
])
def test_invalid_episode_range_maps_fail_loudly(specification, message):
    episodes = [[dict(episode_index=index) for index in range(4)]]
    with pytest.raises(ValueError, match=message):
        resolve_task_selection([], episodes, specification)


def test_episode_ranges_require_a_single_root():
    episodes = [[dict(episode_index=0)], [dict(episode_index=0)]]
    selection = {'tasks': ['a'], 'episode_ranges': {'a': [[0, 1]]}}
    with pytest.raises(ValueError, match='exactly one dataset root'):
        resolve_task_selection([], episodes, selection)
