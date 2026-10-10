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

from typing import Any, Dict, Optional

import numpy as np

from fluxvla.engines import DATASETS
from ..models.third_party_models.patchwam.codec import (ACTION_MODES,
                                                        normalize_action_mode)
from ..models.third_party_models.patchwam.data.history import PastFrameSelector
from .parquet_dataset import ParquetDataset

__all__ = ['PatchWAMParquetDataset']


def _scalar(value, default):
    if value is None:
        return default
    array = np.asarray(value).reshape(-1)
    return array[0] if array.size else default


def _optional_dim(name: str, value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return int(value)


def _combine_action_statistics(sources) -> Dict[str, Any]:
    """Concatenate per-source statistics in packed coordinate order.

    Every statistic key must be present (and non-null) in **all** selected
    sources: fabricating a shorter vector when one source lacks a quantile would
    silently mis-align coordinates. Arrays are concatenated along the last axis
    so any leading (e.g. per-key) prefix shape is preserved.
    """
    keys = set()
    for source in sources:
        keys.update(source)
    combined: Dict[str, Any] = {}
    for key in sorted(keys):
        present = [source.get(key) for source in sources]
        if any(value is None for value in present):
            raise ValueError(
                f'Statistic {key!r} is missing for at least one selected '
                'action source; every selected source must provide the same '
                'statistics (coordinates are never fabricated)')
        arrays = [np.asarray(value, dtype=np.float64) for value in present]
        prefixes = {array.shape[:-1] for array in arrays}
        if len(prefixes) != 1:
            raise ValueError(
                f'Statistic {key!r} has inconsistent prefix shapes '
                f'{sorted(prefixes)}; cannot concatenate the selected action '
                'sources')
        combined[key] = np.concatenate(arrays, axis=-1).tolist()
    return combined


@DATASETS.register_module()
class PatchWAMParquetDataset(ParquetDataset):
    """Native Parquet dataset with PatchWAM temporal metadata.

    Adds episode/root-bounded current/future endpoint rows and original-clock
    causal history rows via the single protected
    :meth:`ParquetDataset._pre_transform_hook`, without duplicating the native
    storage, statistics, or transform execution. A 17-row sampled window yields
    16 action targets, the current frame at the window start, and the future
    endpoint at ``start + future_offset`` (default ``16``), matching the source
    LIBERO/RobotWin recipe.

    Args:
        future_offset: Rows between the current and future endpoint. Defaults to
            ``window_start_idx + action_window_size`` (16 for LIBERO).
        history: Optional mapping enabling causal history. Recognised keys:
            ``slots``, ``interval_s``, ``jitter_s``, ``whole_dropout``,
            ``slot_dropout``, ``truncate_recent``, ``vl_jitter_frames``,
            ``enabled``.
    """

    def __init__(self,
                 *args,
                 future_offset: Optional[int] = None,
                 horizon: Optional[int] = None,
                 history: Optional[Dict[str, Any]] = None,
                 training: bool = True,
                 action_mode: str = 'joint_only',
                 joint_action_key: Optional[str] = None,
                 end_effector_action_key: Optional[str] = None,
                 joint_action_dim: Optional[int] = None,
                 end_effector_action_dim: Optional[int] = None,
                 joint_action_stats_key: Optional[str] = None,
                 end_effector_action_stats_key: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.future_offset = int(future_offset) if future_offset is not None else int(
            self.window_start_idx + self.action_window_size)
        if self.future_offset < 1:
            raise ValueError('future_offset must be positive')
        self.horizon = int(horizon) if horizon is not None else self.future_offset
        if self.horizon < 1:
            raise ValueError('horizon must be positive')
        options = dict(history or {})
        self.history_enabled = bool(options.pop('enabled', bool(history)))
        self.history_selector = PastFrameSelector(
            slots=options.pop('slots', 20),
            interval_s=options.pop('interval_s', 1.0),
            jitter_s=options.pop('jitter_s', 0.4),
            whole_dropout=options.pop('whole_dropout', 0.2),
            slot_dropout=options.pop('slot_dropout', 0.2),
            truncate_recent=options.pop('truncate_recent', True),
            vl_jitter_frames=options.pop('vl_jitter_frames', 1),
        )
        if options:
            raise ValueError(f'Unknown history options: {sorted(options)}')
        self.training = bool(training)
        # Action representation selection. ``joint_only`` with no extra
        # settings is exactly the legacy single-source path.
        self.action_mode = normalize_action_mode(action_mode)
        self.joint_action_key = (str(self.action_key) if joint_action_key is None
                                 else str(joint_action_key))
        self.end_effector_action_key = (
            None if end_effector_action_key is None
            else str(end_effector_action_key))
        if (self.action_mode != 'joint_only'
                and not self.end_effector_action_key):
            raise ValueError(
                f'action_mode={self.action_mode!r} requires '
                '`end_effector_action_key`; the end-effector source is never '
                'inferred from the joint action field')
        if (self.action_mode == 'dual'
                and self.end_effector_action_key == self.joint_action_key):
            raise ValueError(
                'dual action mode requires distinct `joint_action_key` and '
                '`end_effector_action_key` sources')
        self.joint_action_dim = _optional_dim('joint_action_dim',
                                              joint_action_dim)
        self.end_effector_action_dim = _optional_dim('end_effector_action_dim',
                                                     end_effector_action_dim)
        self.joint_action_stats_key = (
            self.joint_action_key if joint_action_stats_key is None
            else str(joint_action_stats_key))
        self.end_effector_action_stats_key = (
            self.end_effector_action_key
            if end_effector_action_stats_key is None
            else str(end_effector_action_stats_key))
        if (self.action_mode == 'dual'
                and (self.joint_action_stats_key
                     == self.end_effector_action_stats_key)):
            raise ValueError(
                'dual action mode requires distinct '
                '`joint_action_stats_key` and `end_effector_action_stats_key`')

    # ------------------------------------------------------------------
    # Action representation
    # ------------------------------------------------------------------
    @staticmethod
    def _row_action_field(row_data: Dict[str, Any], key: str):
        if key not in row_data:
            raise ValueError(
                f'Action field {key!r} is missing from the dataset row; '
                'configure `joint_action_key`/`end_effector_action_key` to '
                'match the data layout')
        return row_data[key]

    def _get_action_value(self, row: int):
        """Select/concatenate the configured action source field(s)."""
        mode = getattr(self, 'action_mode', 'joint_only')
        if mode == 'joint_only':
            joint_key = (getattr(self, 'joint_action_key', None) or
                         self.action_key)
            joint_dim = getattr(self, 'joint_action_dim', None)
            if joint_key == self.action_key and joint_dim is None:
                # No override: exact legacy value and type.
                return super()._get_action_value(row)
            value = np.asarray(
                self._row_action_field(self.dataset[row], joint_key),
                dtype=np.float64).reshape(-1)
            if joint_dim is not None and value.shape[0] != joint_dim:
                raise ValueError(
                    f'Action field {joint_key!r} has width {value.shape[0]} but '
                    f'joint_action_dim={joint_dim} is configured')
            return value
        end_effector_key = getattr(self, 'end_effector_action_key', None)
        if not end_effector_key:
            raise ValueError(
                f'action_mode={mode!r} requires `end_effector_action_key`')
        row_data = self.dataset[row]
        if mode == 'end_effector_only':
            fields = ((end_effector_key, 'end_effector_action_dim'),)
        else:
            joint_key = (getattr(self, 'joint_action_key', None) or
                         self.action_key)
            fields = ((joint_key, 'joint_action_dim'),
                      (end_effector_key, 'end_effector_action_dim'))
        parts = []
        for key, dim_name in fields:
            value = np.asarray(self._row_action_field(row_data, key),
                               dtype=np.float64).reshape(-1)
            width = getattr(self, dim_name, None)
            if width is not None and value.shape[0] != width:
                raise ValueError(
                    f'Action field {key!r} has width {value.shape[0]} but '
                    f'{dim_name}={width} is configured')
            parts.append(value)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def _apply_action_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Derive ``stats['action']`` for the selected coordinate set."""
        mode = getattr(self, 'action_mode', 'joint_only')
        if mode == 'joint_only':
            stats_key = (getattr(self, 'joint_action_stats_key', None) or
                         'action')
            joint_key = (getattr(self, 'joint_action_key', None) or
                         self.action_key)
            if (stats_key == 'action' and joint_key == self.action_key
                    and getattr(self, 'joint_action_dim', None) is None):
                # No override: leave the legacy statistics untouched.
                return data
            sources = [('joint_action_dim', stats_key)]
        else:
            end_effector_key = getattr(self, 'end_effector_action_stats_key',
                                       None)
            if not end_effector_key:
                raise ValueError(
                    f'action_mode={mode!r} requires '
                    '`end_effector_action_stats_key`')
            if mode == 'end_effector_only':
                sources = [('end_effector_action_dim', end_effector_key)]
            else:
                joint_key = (getattr(self, 'joint_action_stats_key', None) or
                             'action')
                sources = [('joint_action_dim', joint_key),
                           ('end_effector_action_dim', end_effector_key)]
        stats = data.get(getattr(self, 'statistics_key', 'stats'))
        if not isinstance(stats, dict):
            raise ValueError(
                f'action_mode={mode!r} requires an action statistics mapping')
        missing = [key for _, key in sources if key not in stats]
        if missing:
            raise KeyError(
                f'action_mode={mode!r} requires dataset statistics for '
                f'{missing}; configure `statistic_keys`/`name_mappings` so the '
                'selected action sources are emitted')
        selected = []
        for dim_name, key in sources:
            entry = stats[key]
            if not isinstance(entry, dict):
                raise ValueError(
                    f'Dataset statistics for {key!r} must be a mapping')
            width = getattr(self, dim_name, None)
            for stat_name, value in entry.items():
                if value is None:
                    continue
                array = np.asarray(value)
                if array.ndim == 0:
                    raise ValueError(
                        f'Statistic {stat_name!r} for {key!r} must have '
                        'an action-coordinate axis')
                length = array.shape[-1]
                if width is not None and length != width:
                    raise ValueError(
                        f'Statistic {stat_name!r} for {key!r} has length '
                        f'{length} but {dim_name}={width} is configured')
            selected.append(entry)
        derived = (_combine_action_statistics(selected)
                   if len(selected) > 1 else selected[0])
        assembled = dict(stats)
        assembled['action'] = derived
        data[getattr(self, 'statistics_key', 'stats')] = assembled
        return data

    def train(self, mode: bool = True) -> 'PatchWAMParquetDataset':
        self.training = bool(mode)
        return self

    def eval(self) -> 'PatchWAMParquetDataset':
        return self.train(False)

    def _row_timestamp(self, dataset_idx: int, row: int, data: Dict[str, Any],
                       fallback: float):
        """Return ``(timestamp, valid)`` with explicit bounds/episode checks.

        Guards ``self.dataset[row]`` against out-of-range rows and rows that
        belong to a different episode or data root.
        """
        if row < 0 or row >= len(self.dataset):
            return fallback, False
        if not self._same_episode_and_dataset(int(row), dataset_idx, data):
            return fallback, False
        return float(_scalar(self.dataset[int(row)].get('timestamp'),
                             fallback)), True

    def _pre_transform_hook(self, data: Dict[str, Any], index: int,
                            dataset_idx: int) -> Dict[str, Any]:
        data = super()._pre_transform_hook(data, index, dataset_idx)
        info = self.info[dataset_idx]
        fps = float(info.get('fps', 0) or 0)
        frame_index = int(_scalar(data.get('frame_index'), 0))
        timestamp = float(
            _scalar(data.get('timestamp'), frame_index / fps if fps > 0 else 0.0))
        episode_index = int(_scalar(data.get('episode_index'), -1))
        data['patchwam_row'] = int(index)
        data['patchwam_episode_index'] = episode_index
        data['patchwam_frame_index'] = frame_index
        data['patchwam_timestamp'] = timestamp
        data['patchwam_fps'] = fps
        data['patchwam_reset'] = bool(frame_index == 0)
        data['patchwam_augment'] = bool(self.training)

        # Endpoint window bounded to the episode/root: an out-of-episode row
        # repeats the last valid row and is marked invalid.
        window_rows = [int(index)]
        window_valid = [True]
        window_timestamps = [timestamp]
        for step in range(1, self.horizon + 1):
            row = index + step
            step_timestamp, valid = self._row_timestamp(
                dataset_idx, row, data, window_timestamps[-1])
            window_valid.append(bool(valid))
            window_rows.append(int(row) if valid else window_rows[-1])
            window_timestamps.append(step_timestamp if valid
                                     else window_timestamps[-1])
        data['patchwam_window_rows'] = np.asarray(window_rows, dtype=np.int64)
        data['patchwam_window_valid'] = np.asarray(window_valid, dtype=bool)
        data['patchwam_window_timestamps'] = np.asarray(window_timestamps,
                                                        dtype=np.float64)
        future_row = index + self.future_offset
        future_valid = bool(self._same_episode_and_dataset(
            future_row, dataset_idx, data))
        data['patchwam_future_row'] = int(future_row) if future_valid else int(
            index)
        data['patchwam_future_valid'] = future_valid

        if self.history_enabled:
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(
                    'PatchWAM history requires a positive dataset fps in '
                    'meta/info.json')
            local_rows, valid = self.history_selector.select(
                frame_index, fps, training=self.training,
                vision_language=False)
            absolute = index - frame_index + local_rows
            data['patchwam_history_rows'] = absolute.astype(np.int64)
            history_valid, history_timestamps = self._history_clocks(
                absolute, valid.numpy(), index, dataset_idx, data, timestamp)
            data['patchwam_history_valid'] = history_valid
            data['patchwam_history_timestamps'] = history_timestamps
            # VL history uses its own jitter contract (no world dropout).
            vl_rows, vl_valid = self.history_selector.select(
                frame_index, fps, training=self.training, vision_language=True)
            vl_absolute = index - frame_index + vl_rows
            data['patchwam_vl_history_rows'] = vl_absolute.astype(np.int64)
            vl_history_valid, vl_timestamps = self._history_clocks(
                vl_absolute, vl_valid.numpy(), index, dataset_idx, data,
                timestamp)
            data['patchwam_vl_history_valid'] = vl_history_valid
            data['patchwam_vl_history_timestamps'] = vl_timestamps
        return self._apply_action_statistics(data)

    def _history_clocks(self, rows, valid, index, dataset_idx, data, fallback):
        """Validate each history row and resolve its in-episode timestamp."""
        rows = np.asarray(rows, dtype=np.int64)
        valid = np.asarray(valid, dtype=bool)
        timestamps = []
        for position, row in enumerate(rows):
            row_timestamp, ok = self._row_timestamp(dataset_idx, int(row), data,
                                                    fallback)
            valid[position] = bool(valid[position] and ok)
            timestamps.append(row_timestamp)
        return valid, np.asarray(timestamps, dtype=np.float64)
