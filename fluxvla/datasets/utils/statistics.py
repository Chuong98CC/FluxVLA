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


def reject_conversion_overwrite(source_path: str, out_path: str) -> None:
    """Refuse to write a conversion output over its own source file.

    Conversion helpers promise that the released/native asset they read is left
    untouched. They write in place, so the guarantee has to be enforced: a
    caller passing the source as the output would otherwise truncate it. Both
    an identical path and an existing alias (symlink or hardlink) are rejected
    before anything is opened for writing.

    Args:
        source_path: File the conversion reads.
        out_path: File the conversion would write.

    Raises:
        ValueError: When the two refer to the same file.
    """
    if os.path.realpath(source_path) == os.path.realpath(out_path):
        raise ValueError(
            f'Refusing to convert {source_path!r} onto itself; choose a '
            'different output path so the source stays untouched')
    if os.path.exists(out_path) and os.path.exists(source_path) \
            and os.path.samefile(source_path, out_path):
        raise ValueError(
            f'Refusing to convert {source_path!r} onto its own alias '
            f'{out_path!r} (symlink or hardlink); choose a different output '
            'path so the source stays untouched')


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
    reject_conversion_overwrite(native_path, out_path)
    converted = robotwin_statistics_from_native(
        payload, statistic_name=statistic_name, state_key=state_key,
        action_key=action_key)
    with open(out_path, 'w', encoding='utf-8') as stream:
        json.dump(_jsonable(converted), stream, indent=2)
    return converted


def released_robotwin_statistics_to_native(
    payload: Any,
    *,
    statistic_name: str = 'robotwin',
    state_key: str = 'proprio',
    action_key: str = 'action',
    field_key: str = 'default',
) -> dict:
    """Convert the released RoboTwin ``dataset_stats.json`` to the native schema.

    The published PatchWAM RoboTwin asset stores
    ``{state: {default: {global_mean, global_std}, ...}, action: {...}}`` while
    the FluxVLA training and evaluation transforms consume
    ``{statistic_name: {proprio: {mean, std}, action: {mean, std}}}``. The
    released z-score contract is defined by the ``global_*`` fields, which are
    copied verbatim; the per-horizon ``stepwise_*`` fields are not part of the
    released z-score and are ignored.

    The conversion is exact (no rounding or rescaling) so a converted file
    reproduces the released normalization bit-for-bit.

    Args:
        payload: Parsed released statistics file.
        statistic_name: Native group name written into the result.
        state_key / action_key: Native field names for the two groups.
        field_key: Released-schema field key (``default``).
    """
    if not isinstance(payload, dict) or not payload:
        raise ValueError('Released dataset statistics must be a non-empty '
                         'mapping')
    converted: dict = {statistic_name: {}}
    for target, source in ((state_key, 'state'), (action_key, 'action')):
        group = payload.get(source)
        if not isinstance(group, dict) or field_key not in group:
            raise KeyError(
                f'Released dataset statistics have no {source}.{field_key}; '
                f'available top-level keys: {sorted(payload)}')
        fields = group[field_key]
        if not isinstance(fields, dict):
            raise TypeError(
                f'Released {source}.{field_key} statistics must be a mapping')
        for required in ('global_mean', 'global_std'):
            if required not in fields or fields[required] is None:
                raise KeyError(
                    f'Released {source}.{field_key} statistics need '
                    f'{required!r}; available fields: {sorted(fields)}')
        mean = np.asarray(fields['global_mean'], dtype=np.float64).reshape(-1)
        std = np.asarray(fields['global_std'], dtype=np.float64).reshape(-1)
        if mean.size == 0 or mean.shape != std.shape:
            raise ValueError(
                f'Released {source} statistics need matching, non-empty '
                f'global_mean/global_std vectors, got {mean.shape} and '
                f'{std.shape}')
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise ValueError(
                f'Released {source} statistics contain non-finite values')
        converted[statistic_name][target] = {
            'mean': mean.tolist(),
            'std': std.tolist(),
        }
    return converted


def convert_released_robotwin_statistics_file(
    released_path: str,
    out_path: str,
    *,
    statistic_name: str = 'robotwin',
    state_key: str = 'proprio',
    action_key: str = 'action',
) -> dict:
    """Convert a released RoboTwin statistics file on disk to the native schema.

    ``out_path`` is always a new file; the released asset is never modified::

        python -c "from fluxvla.datasets.utils.statistics import \\
          convert_released_robotwin_statistics_file as c; \\
          c('<assets>/dataset_stats.json', '<run>/dataset_statistics_robotwin.json')"
    """
    with open(released_path, 'r', encoding='utf-8') as stream:
        payload = json.load(stream)
    reject_conversion_overwrite(released_path, out_path)
    converted = released_robotwin_statistics_to_native(
        payload, statistic_name=statistic_name, state_key=state_key,
        action_key=action_key)
    with open(out_path, 'w', encoding='utf-8') as stream:
        json.dump(_jsonable(converted), stream, indent=2)
    return converted
