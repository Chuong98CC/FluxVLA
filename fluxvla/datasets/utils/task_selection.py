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

"""Deterministic native task filtering.

Task indices are root-local in LeRobot metadata, so the task *name* is the
global identity here: the same index may mean different tasks in different
roots, and the per-root tables are what resolve an explicit integer. The
resolution runs before sample-index generation and before statistics use the
selected episodes, and it returns the concrete ``(root, episode)`` pairs that
the recipe actually selected.
"""

import json
import os
from pathlib import Path
from typing import (Any, Dict, Iterator, List, Mapping, Optional, Sequence,
                    Set, Tuple)

import numpy as np

__all__ = [
    'dataset_task_selection', 'reject_predataset_statistics',
    'resolve_task_selection', 'save_task_selection'
]

#: Statistics options resolved before the dataset exists, so they cannot see a
#: task/episode selection and would normalize a narrowed run with full-data
#: statistics.
PREDATASET_STATISTICS_KEYS = ('dataset_statistics', 'dataset_statistics_path',
                              'auto_compute_statistics')


def _nested(mapping: Mapping[str, Any], *path: str):
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def reject_predataset_statistics(dataset_cfg: Mapping[str, Any]) -> None:
    """Refuse pre-dataset statistics for a task-selected recipe.

    ``dataset_statistics``/``dataset_statistics_path``/``auto_compute_statistics``
    are resolved before the dataset is built, so they cannot respect the
    resolved task/episode selection. A narrowed recipe must let the dataset
    wrapper aggregate statistics from the episodes it actually selected.
    """
    selection = (_nested(dataset_cfg, 'task_selection') or
                 _nested(dataset_cfg, 'datasets', 'task_selection'))
    if selection is None:
        return
    for key in PREDATASET_STATISTICS_KEYS:
        if (_nested(dataset_cfg, key) is not None
                or _nested(dataset_cfg, 'datasets', key) is not None):
            raise ValueError(
                f'A task-selected recipe cannot use `{key}`: those statistics '
                'are computed before the selection resolves. Let the dataset '
                'wrapper aggregate statistics from the selected episodes.')


def _iter_datasets(dataset) -> Iterator[Any]:
    """Iterate a dataset and its wrappers (``datasets``/``dataset`` links)."""
    seen, stack = set(), [dataset]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attribute in ('datasets', 'dataset'):
            inner = getattr(current, attribute, None)
            if isinstance(inner, (list, tuple)):
                stack.extend(inner)
            elif inner is not None:
                stack.append(inner)


def dataset_task_selection(dataset) -> Optional[Dict[str, Any]]:
    """Resolved task selection of a dataset or dataset wrapper, if any."""
    for candidate in _iter_datasets(dataset):
        metadata = getattr(candidate, 'selection_metadata', None)
        if callable(metadata):
            resolved = metadata()
            if resolved is not None:
                return resolved
    return None


def save_task_selection(dataset, work_dir,
                        filename: str = 'dataset_selection.json'
                        ) -> Optional[str]:
    """Persist the concrete resolved selection into the run directory.

    Traverses dataset wrappers (``DistributedRepeatingDataset`` and friends) so
    the narrowed roots/episodes the run actually trained on are recorded beside
    the resolved config and dataset statistics. Returns the written path, or
    ``None`` when the dataset declares no selection.
    """
    selection = dataset_task_selection(dataset)
    if selection is None:
        return None
    path = Path(work_dir).resolve() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    os.replace(temporary, path)
    return str(path)


def _root_name_tables(tasks: Sequence[Sequence[Mapping[str, Any]]]):
    """Per-root ``task_index -> task name`` tables."""
    tables: List[Dict[int, str]] = []
    for root_index, rows in enumerate(tasks):
        table: Dict[int, str] = {}
        for row in rows:
            index = int(row['task_index'])
            name = str(row['task'])
            if index in table and table[index] != name:
                raise ValueError(
                    f'Ambiguous task names for index {index} in root '
                    f'{root_index}: {sorted({table[index], name})}')
            table[index] = name
        tables.append(table)
    return tables


def _episode_task_name(value, table, name_to_index, root_index, episode):
    """Episode task entry as a name.

    LeRobot stores task *strings* in ``episodes.jsonl``; the native fixtures in
    this repository also write the local task index, so both spellings resolve
    through the same per-root table.
    """
    if isinstance(value, str) and value in name_to_index:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in table:
            return table[value]
    elif isinstance(value, str) and value.strip().lstrip('-').isdigit():
        if int(value) in table:
            return table[int(value)]
    raise ValueError(
        f'Episode {episode} in root {root_index} declares task {value!r} '
        'which is missing from tasks.jsonl')


def _episode_names(episodes: Sequence[Sequence[Mapping[str, Any]]],
                   tables: Sequence[Dict[int, str]]):
    """Per-root ``episode_index -> (task name, local task index)``."""
    resolved: List[Dict[int, Tuple[str, int]]] = []
    for root_index, rows in enumerate(episodes):
        table = tables[root_index]
        name_to_index = {}
        for index, name in table.items():
            name_to_index.setdefault(name, index)
        entries: Dict[int, Tuple[str, int]] = {}
        for row in rows:
            episode = int(row['episode_index'])
            values = row.get('tasks')
            if values is None:
                single = row.get('task')
                values = [] if single in (None, '') else [single]
            elif not isinstance(values, (list, tuple)):
                values = [values]
            distinct = sorted({
                _episode_task_name(value, table, name_to_index, root_index,
                                   episode) for value in values
            })
            if len(distinct) != 1:
                raise ValueError(
                    f'Episode {episode} in root {root_index} must declare '
                    f'exactly one task name, got {distinct}')
            name = distinct[0]
            entries[episode] = (name, int(name_to_index[name]))
        resolved.append(entries)
    return resolved


def resolve_task_selection(
    tasks: Sequence[Sequence[Mapping[str, Any]]],
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    selection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve a task/episode subset against the per-root metadata.

    Args:
        tasks: Per-root rows of ``tasks.jsonl`` (``task_index``/``task``).
        episodes: Per-root rows of ``episodes.jsonl`` (``episode_index`` and
            ``tasks``).
        selection: Mapping with optional ``tasks`` (names or local indices),
            ``num_tasks`` (default 5), ``episodes_per_task`` and ``seed``.

    Returns:
        ``{'selection': <concrete resolved selection>, 'keep': set of
        (root index, episode index), 'names': [selected task names]}``.
    """
    specification = dict(selection)
    unknown = sorted(set(specification) -
                     {'tasks', 'num_tasks', 'episodes_per_task', 'seed'})
    if unknown:
        raise ValueError(f'Unknown task selection options: {unknown}')
    seed = specification.get('seed', 42)
    if type(seed) is not int:
        raise ValueError('The task selection seed must be an integer')
    tables = _root_name_tables(tasks)
    per_root = _episode_names(episodes, tables)
    index_to_names: Dict[int, Set[str]] = {}
    name_to_episodes: Dict[str, List[Tuple[int, int, int]]] = {}
    for root_index, entries in enumerate(per_root):
        for episode, (name, local_index) in sorted(entries.items()):
            index_to_names.setdefault(local_index, set()).add(name)
            name_to_episodes.setdefault(name, []).append(
                (root_index, episode, local_index))
    if not name_to_episodes:
        raise ValueError('The dataset roots declare no tasks')
    ambiguous = {index: sorted(names)
                 for index, names in index_to_names.items() if len(names) > 1}
    requested = specification.get('tasks')
    if requested is not None:
        if (isinstance(requested, (str, int))
                and not isinstance(requested, bool)):
            requested = [requested]
        if not isinstance(requested, (list, tuple)) or not requested:
            raise ValueError(
                'Explicit tasks must be a nonempty list of names or indices')
        chosen: List[str] = []
        source = 'explicit'
        for item in requested:
            if isinstance(item, bool) or not isinstance(item, (str, int)):
                raise TypeError(
                    'Task entries must be names or indices, received '
                    f'{item!r}')
            if isinstance(item, int):
                names = index_to_names.get(item)
                if names and len(names) > 1:
                    raise ValueError(
                        f'Task index {item} means different tasks across '
                        f'roots ({sorted(names)}); use a task name instead')
                if not names:
                    # A local index may be absent from some roots; accept it if
                    # every root that declares it agrees on the name.
                    names = {table[item] for table in tables if item in table}
                if len(names) > 1:
                    raise ValueError(
                        f'Task index {item} means different tasks across '
                        f'roots ({sorted(names)}); use a task name instead')
            else:
                declared = any(item in table.values() for table in tables)
                names = ({item} if (declared or item in name_to_episodes)
                         else set())
            if not names:
                raise ValueError(
                    f'Requested task is absent from the dataset: {item!r}')
            name = min(names)
            if name not in chosen:
                chosen.append(name)
    else:
        num_tasks = specification.get('num_tasks', 5)
        if type(num_tasks) is not int or num_tasks < 1:
            raise ValueError('num_tasks must be a positive integer')
        names = sorted(name_to_episodes)
        if len(names) < num_tasks:
            raise ValueError(
                f'The dataset has {len(names)} distinct tasks; {num_tasks} '
                'were requested')
        order = np.random.default_rng(seed).permutation(len(names)).tolist()
        chosen = sorted(names[position] for position in order[:num_tasks])
        source = 'seeded'
    episodes_per_task = specification.get('episodes_per_task')
    if episodes_per_task is not None and (type(episodes_per_task) is not int
                                          or episodes_per_task < 1):
        raise ValueError('episodes_per_task must be a positive integer')
    empty = sorted(name for name in chosen if not name_to_episodes.get(name))
    if empty:
        raise ValueError(f'Selected tasks contain no episodes: {empty}')
    if episodes_per_task is not None:
        generator = np.random.default_rng(seed)
        for name in sorted(chosen):
            rows = name_to_episodes[name]
            order = generator.permutation(len(rows)).tolist()
            name_to_episodes[name] = [rows[position]
                                      for position in
                                      order[:episodes_per_task]]
    keep: Set[Tuple[int, int]] = set()
    tasks_report = []
    for name in chosen:
        rows = sorted(name_to_episodes[name])
        keep.update((root_index, episode)
                    for root_index, episode, _ in rows)
        tasks_report.append({
            'name': name,
            'episodes': [{'root_index': root_index, 'episode_index': episode}
                         for root_index, episode, _ in rows],
            'task_indices': {str(root_index): sorted({
                local_index for root, _, local_index in rows
                if root == root_index}) for root_index in sorted(
                    {root for root, _, _ in rows})},
        })
    selection_report = {
        'source': source,
        'seed': seed,
        'num_tasks': len(chosen),
        'episodes_per_task': episodes_per_task,
        'available_tasks': len(name_to_episodes),
        'ambiguous_indices': sorted(ambiguous),
        'episodes': len(keep),
        'tasks': tasks_report,
    }
    return {'selection': selection_report, 'keep': keep, 'names': chosen}
