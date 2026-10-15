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

import json
import os
from typing import Any

import numpy as np

from fluxvla.engines import initialize_overwatch

overwatch = initialize_overwatch(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def save_dataset_statistics(dataset_statistics, run_dir):
    """Save dataset statistics to `dataset_statistics.json`."""
    out_path = os.path.join(run_dir, 'dataset_statistics.json')
    with open(out_path, 'w') as f_json:
        json.dump(_jsonable(dataset_statistics), f_json, indent=2)
    overwatch.info(f'Saved dataset statistics file at path {out_path}')


def save_grouped_dataset_statistics(grouped_dataset_statistics, run_dir):
    """Save grouped dataset statistics to one JSON file per group."""
    for group_name, group_stats in grouped_dataset_statistics.items():
        out_path = os.path.join(run_dir,
                                f'dataset_statistics_{group_name}.json')
        with open(out_path, 'w') as f_json:
            json.dump(_jsonable(group_stats), f_json, indent=2)
        overwatch.info(f'Saved dataset statistics for group {group_name} '
                       f'at path {out_path}')


def robotwin_statistics_from_native(
    payload: Any,
    *,
    statistic_name: Any = None,
    state_key: str = 'proprio',
    action_key: str = 'action',
    field_key: str = 'default',
) -> dict:
    """Convert native ``dataset_statistics.json`` to the released RoboTwin schema.

    ``scripts/train.py`` writes the native
    ``{statistic_name: {field: {min, max, mean, std, ...}}}`` payload (``field``
    already renamed by ``name_mappings``), while the released RoboTwin asset
    consumes
    ``{state: {default: {global_mean, global_std}}, action: {...}}``. This
    helper performs that explicit, lossless-for-z-score conversion so a newly
    trained run evaluates with its own training statistics.

    Args:
        payload: Native statistics mapping, grouped or already unwrapped.
        statistic_name: Group to convert. When omitted the single group that
            carries ``state_key`` is used; several groups must be disambiguated
            explicitly.
        state_key / action_key: Native field names (``proprio`` and ``action``
            for the PatchWAM/LIT RobotWin recipes).
        field_key: Released-schema field key (``default``).
    """
    if not isinstance(payload, dict) or not payload:
        raise ValueError('Native dataset statistics must be a non-empty mapping')
    group = payload
    if statistic_name is not None:
        if statistic_name not in payload:
            raise KeyError(
                f'Native dataset statistics have no {statistic_name!r} group; '
                f'available groups: {sorted(payload)}')
        group = payload[statistic_name]
    elif state_key not in payload:
        candidates = [
            value for value in payload.values()
            if isinstance(value, dict) and state_key in value
        ]
        if len(candidates) != 1:
            raise KeyError(
                f'Cannot pick the native statistics group holding '
                f'{state_key!r} from groups {sorted(payload)}; pass '
                '`statistic_name` explicitly')
        group = candidates[0]
    if not isinstance(group, dict):
        raise TypeError('Native dataset statistics group must be a mapping')
    converted = {}
    for field, source_key in (('state', state_key), ('action', action_key)):
        if source_key not in group:
            raise KeyError(
                f'Native dataset statistics have no {source_key!r} field; '
                f'available fields: {sorted(group)}')
        fields = group[source_key]
        if not isinstance(fields, dict) or 'mean' not in fields \
                or 'std' not in fields:
            raise KeyError(
                f'Native {source_key!r} statistics need `mean` and `std`')
        mean = np.asarray(fields['mean'], dtype=np.float64).reshape(-1)
        std = np.asarray(fields['std'], dtype=np.float64).reshape(-1)
        if mean.size == 0 or mean.shape != std.shape:
            raise ValueError(
                f'Native {source_key!r} statistics need matching, non-empty '
                f'mean/std vectors, got {mean.shape} and {std.shape}')
        converted[field] = {
            field_key: {
                'global_mean': mean.tolist(),
                'global_std': std.tolist(),
            }
        }
    return converted


def convert_robotwin_statistics_file(
    native_path: str,
    out_path: str,
    *,
    statistic_name: Any = None,
    state_key: str = 'proprio',
    action_key: str = 'action',
) -> dict:
    """Convert a native statistics file on disk to the released schema.

    Explicit one-off command::

        python -c "from fluxvla.datasets.utils.statistics import \\
          convert_robotwin_statistics_file as c; \\
          c('work_dirs/run/dataset_statistics.json', 'run/robotwin_stats.json')"
    """
    with open(native_path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    converted = robotwin_statistics_from_native(
        payload, statistic_name=statistic_name, state_key=state_key,
        action_key=action_key)
    with open(out_path, 'w', encoding='utf-8') as stream:
        json.dump(_jsonable(converted), stream, indent=2)
    return converted
