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
from typing import Any, Dict, Optional, Union

import numpy as np

from fluxvla.engines import DATASETS
from ..models.third_party_models.patchwam.codec import (ACTION_MODES,
                                                        normalize_action_mode)
from ..models.third_party_models.patchwam.data.history import PastFrameSelector
from .parquet_dataset import ParquetDataset

__all__ = ['PatchWAMParquetDataset']


def _nonidle_entries(payload, payload_path):
    """Return the episode -> row-range mapping from a filter payload."""
    if not isinstance(payload, dict):
        raise ValueError(
            f'Nonidle filter {payload_path} must contain a JSON object')
    entries = payload.get('episodes', payload)
    if not isinstance(entries, dict):
        raise ValueError(
            f'Nonidle filter {payload_path} "episodes" must be a mapping of '
            'episode index to [start, end) row ranges')
    return entries


def _nonidle_intervals(raw_ranges, payload_path, episode):
    """Validate one episode's half-open [start, end) row intervals."""
    if not isinstance(raw_ranges, list):
        raise ValueError(
            f'Nonidle filter {payload_path} episode {episode} must be a list '
            'of [start, end) row ranges')
    intervals = []
    for item in raw_ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f'Nonidle filter {payload_path} episode {episode} range '
                f'{item!r} is not a [start, end) start/end pair')
        start, end = item
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)):
            raise ValueError(
                f'Nonidle filter {payload_path} episode {episode} range '
                f'{item!r} must contain integer start/end offsets')
        intervals.append((int(start), int(end)))
    return intervals


def _nonidle_root_selection(hf_dataset, payload, root_path, payload_path):
    """Resolve the retained rows of one root and its original-clock layout.

    Mirrors the source reader: per episode the kept rows are the sorted union
    of the clipped half-open intervals, an absent episode key keeps the whole
    episode, and an explicit ``[]`` removes every row of that episode.
    """
    entries = _nonidle_entries(payload, payload_path)
    if 'frame_index' not in hf_dataset.column_names:
        raise ValueError(
            f'Nonidle filtering requires a frame_index column: {root_path}')
    episode_index = np.asarray(hf_dataset['episode_index'], dtype=np.int64)
    frame_index = np.asarray(hf_dataset['frame_index'], dtype=np.int64)
    total = len(episode_index)
    starts = [0] + (np.flatnonzero(np.diff(episode_index) != 0) + 1).tolist()
    blocks: Dict[int, Any] = {}
    for position, start in enumerate(starts):
        end = (starts[position + 1] if position + 1 < len(starts) else total)
        episode = int(episode_index[start])
        if episode in blocks:
            raise ValueError(
                f'Episode {episode} is not one contiguous block in '
                f'{root_path}; nonidle filtering needs one block per episode')
        frames = frame_index[start:end]
        # Original-clock history is addressed by episode-local frame index, so
        # the recording axis must be the contiguous 0-based one LeRobot writes.
        if not np.array_equal(frames, np.arange(len(frames), dtype=np.int64)):
            raise ValueError(
                'Nonidle filtering requires the contiguous 0-based recording '
                'axis (frame_index = 0..length-1) for episode '
                f'{episode}: {root_path}')
        blocks[episode] = (start, end)

    keep = np.ones(total, dtype=bool)
    for episode, (start, end) in blocks.items():
        if str(episode) not in entries:
            continue
        length = end - start
        local = np.zeros(length, dtype=bool)
        for first, last in _nonidle_intervals(entries[str(episode)],
                                              payload_path, episode):
            first, last = max(0, first), min(length, last)
            if last > first:
                local[first:last] = True
        keep[start:end] = local

    kept = np.flatnonzero(keep)
    if not len(kept):
        raise ValueError(
            f'Nonidle filtering left no rows for root {root_path} '
            f'(filter {payload_path})')
    # One vectorized search over the episode edges keeps startup linear in the
    # number of episodes instead of scanning every retained row per episode.
    edges = np.asarray(starts + [total], dtype=np.int64)
    positions = np.searchsorted(kept, edges)
    episodes = {
        int(episode_index[start]): int(start)
        for start in starts
    }
    layout = dict(table=hf_dataset, size=total, kept=kept, episodes=episodes,
                  kept_per_episode=np.diff(positions).tolist(),
                  payload=str(payload_path))
    return kept, layout


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
        nonidle_filter_path: Optional non-idle row filter for this dataset.
            Either one JSON path applied to every root, or a mapping of data
            root path to JSON path. ``None`` (the default) keeps the native
            unfiltered behavior. See :meth:`_select_root_rows`.
        require_nonidle_filter: When True, a missing ``nonidle_filter_path``
            fails at construction instead of silently training unfiltered.
            Used only by the source-matching RobotWin recipe.
    """

    def __init__(self,
                 *args,
                 future_offset: Optional[int] = None,
                 horizon: Optional[int] = None,
                 history: Optional[Dict[str, Any]] = None,
                 training: bool = True,
                 nonidle_filter_path: Optional[Union[str, Dict[str, str]]] = None,  # noqa: E501
                 require_nonidle_filter: bool = False,
                 action_mode: str = 'joint_only',
                 joint_action_key: Optional[str] = None,
                 end_effector_action_key: Optional[str] = None,
                 joint_action_dim: Optional[int] = None,
                 end_effector_action_dim: Optional[int] = None,
                 joint_action_stats_key: Optional[str] = None,
                 end_effector_action_stats_key: Optional[str] = None,
                 **kwargs) -> None:
        # The per-root selection hook runs inside ``super().__init__``, so the
        # filter settings and their accumulators must exist beforehand.
        self.nonidle_filter_path = self._validate_nonidle_filter_path(
            nonidle_filter_path)
        self.require_nonidle_filter = bool(require_nonidle_filter)
        if self.require_nonidle_filter and self.nonidle_filter_path is None:
            raise ValueError(
                'require_nonidle_filter=True requires a nonidle_filter_path '
                '(set PATCHWAM_NONIDLE_FILTER to the non-idle row ranges '
                'JSON) so training cannot silently run unfiltered')
        self._nonidle_raw_sizes: list = []
        self._nonidle_roots: Dict[int, Any] = {}
        self._nonidle_filtered = False
        self._nonidle_raw_offsets = None
        self._nonidle_raw_total_rows = None
        super().__init__(*args, **kwargs)
        # Any filtered root makes the whole sample pool use the recording axis
        # for its raw keys (and only then are the dataset-axis keys dropped).
        self._nonidle_filtered = bool(self._nonidle_roots)
        if self._nonidle_raw_sizes:
            self._nonidle_raw_offsets = np.cumsum(
                [0] + self._nonidle_raw_sizes).tolist()
            self._nonidle_raw_total_rows = int(
                sum(self._nonidle_raw_sizes))
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
    # Non-idle row filtering (opt-in, PatchWAM-only)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_nonidle_filter_path(nonidle_filter_path):
        if nonidle_filter_path is None:
            return None
        if isinstance(nonidle_filter_path, dict):
            for root, path in nonidle_filter_path.items():
                if not isinstance(path, (str, os.PathLike)):
                    raise ValueError(
                        'nonidle_filter_path mapping values must be JSON '
                        f'paths, got {path!r} for root {root!r}')
            return dict(nonidle_filter_path)
        if isinstance(nonidle_filter_path, (str, os.PathLike)):
            return str(nonidle_filter_path)
        raise ValueError(
            'nonidle_filter_path must be None, a JSON path, or a mapping of '
            f'root path to JSON path, got {nonidle_filter_path!r}')

    def _nonidle_path_for(self, root_path: str):
        """Resolve the configured filter file for one data root."""
        configured = self.nonidle_filter_path
        if configured is None:
            return None
        if not isinstance(configured, dict):
            return configured
        candidates = (
            str(root_path),
            os.path.abspath(os.path.expanduser(str(root_path))),
            os.path.realpath(str(root_path)),
        )
        for candidate in candidates:
            if candidate in configured:
                return configured[candidate]
        raise KeyError(
            f'No nonidle filter configured for root {candidates[1]}; '
            f'configured roots: {sorted(configured)}')

    @staticmethod
    def _load_nonidle_payload(payload_path):
        path = os.path.abspath(os.path.expanduser(str(payload_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Nonidle filter file not found: {path}')
        with open(path, 'r', encoding='utf-8') as handle:
            raw = handle.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f'Nonidle filter {path} is not valid JSON: {error}') from error

    def _select_root_rows(self, root_index: int, root_path: str, hf_dataset):
        """Drop non-idle rows of one root before concatenation.

        The raw per-root table and the retained-row map are kept so original
        clock history, including rows the filter removed, still resolves.
        """
        self._nonidle_raw_sizes.append(len(hf_dataset))
        payload_path = self._nonidle_path_for(root_path)
        if payload_path is None:
            return hf_dataset
        payload = self._load_nonidle_payload(payload_path)
        kept, layout = _nonidle_root_selection(hf_dataset, payload, root_path,
                                              payload_path)
        self._nonidle_roots[root_index] = layout
        return hf_dataset.select(kept.tolist())

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

    def _filter_layout(self, dataset_idx: int):
        """Non-idle layout for one root, or None when it is unfiltered."""
        return getattr(self, '_nonidle_roots', {}).get(dataset_idx)

    def _filtered_root_base(self, dataset_idx: int) -> int:
        return int(self.dataset_cumulative_sizes[dataset_idx])

    def _raw_root_offset(self, dataset_idx: int) -> int:
        """First original-recording row of one root.

        Filtering removes rows from the dataset axis only, so the original rows
        of a root keep their concat offsets.
        """
        offsets = getattr(self, '_nonidle_raw_offsets', None)
        if offsets is None:
            return self._filtered_root_base(dataset_idx)
        return int(offsets[dataset_idx])

    def _raw_total_rows(self) -> int:
        """Total rows of the unfiltered recording across every root."""
        total = getattr(self, '_nonidle_raw_total_rows', None)
        return len(self.dataset) if total is None else int(total)

    def _original_rows(self, dataset_idx: int, dataset_rows) -> np.ndarray:
        """Map global dataset rows of one root onto the recording axis.

        ``dataset_rows`` are dataset rows; the result is recording rows. The
        two axes differ by the root's retained-row shift only, so the local
        offset is applied exactly once: an unfiltered root maps
        ``raw_base + (row - dataset_base)`` instead of adding a base twice.
        """
        rows = np.asarray(dataset_rows, dtype=np.int64)
        local = rows - self._filtered_root_base(dataset_idx)
        layout = self._filter_layout(dataset_idx)
        if layout is None:
            return (self._raw_root_offset(dataset_idx)
                    + local).astype(np.int64)
        kept = layout['kept']
        return (self._raw_root_offset(dataset_idx)
                + kept[np.clip(local, 0, len(kept) - 1)]).astype(np.int64)

    def _history_raw_rows(self, dataset_idx: int, episode_index: int,
                          frame_index: int, global_row: int, clock_rows):
        """Original-recording rows for original-clock history slots.

        ``clock_rows`` are episode-local frame indexes on the recording clock,
        the axis ``PastFrameSelector`` samples. They resolve to exact original
        rows (never a substituted neighbour), so rows the filter removed remain
        addressable for timestamp lookup and row gathering. An unfiltered root
        keeps the dataset axis, which is identical to its recording.
        """
        layout = self._filter_layout(dataset_idx)
        rows = np.asarray(clock_rows, dtype=np.int64)
        if layout is None:
            dataset_rows = int(global_row) - frame_index + rows
            return (self._raw_root_offset(dataset_idx)
                    + (dataset_rows
                       - self._filtered_root_base(dataset_idx))).astype(
                           np.int64)
        episode_start = layout['episodes'].get(int(episode_index))
        if episode_start is None:
            raise ValueError(
                f'Episode {episode_index} is missing from the non-idle layout '
                f'of root {dataset_idx}')
        return (self._raw_root_offset(dataset_idx) + int(episode_start)
                + rows).astype(np.int64)

    def _raw_row_timestamp(self, dataset_idx: int, raw_row: int,
                           episode_index: int, data: Dict[str, Any],
                           fallback: float):
        """Resolve a timestamp for an original-recording row.

        A filtered root reads its raw table directly, so rows the filter
        removed still resolve; an unfiltered root reads the dataset row, which
        is the same recording row.
        """
        layout = self._filter_layout(dataset_idx)
        if layout is None:
            dataset_row = (self._filtered_root_base(dataset_idx)
                           + (int(raw_row)
                              - self._raw_root_offset(dataset_idx)))
            return self._row_timestamp(dataset_idx, dataset_row, data,
                                       fallback)
        local_row = int(raw_row) - self._raw_root_offset(dataset_idx)
        if local_row < 0 or local_row >= layout['size']:
            return fallback, False
        row = layout['table'][local_row]
        if int(_scalar(row.get('episode_index'), -1)) != int(episode_index):
            return fallback, False
        return float(_scalar(row.get('timestamp'), fallback)), True

    def _history_clocks(self, clock_rows, valid, episode_index, dataset_idx,
                        frame_index, global_row, data, fallback):
        """Validate each history slot and resolve its original-clock stamp."""
        valid = np.asarray(valid, dtype=bool).copy()
        raw_rows = self._history_raw_rows(dataset_idx, episode_index,
                                         frame_index, global_row, clock_rows)
        timestamps = []
        for position, raw_row in enumerate(raw_rows):
            row_timestamp, ok = self._raw_row_timestamp(
                dataset_idx, int(raw_row), episode_index, data, fallback)
            valid[position] = bool(valid[position] and ok)
            timestamps.append(row_timestamp)
        return valid, np.asarray(timestamps, dtype=np.float64)

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
        # Original-recording row metadata. ``patchwam_row`` and the window /
        # future keys stay dataset rows for the filtered-sequence pipeline;
        # the ``patchwam_raw_*`` keys are exact recording rows (per-root
        # offsets preserved) for a predecoded original-row tensor. Both axes
        # coincide when no non-idle filter is active.
        filtered = bool(getattr(self, '_nonidle_filtered', False))
        raw_total = self._raw_total_rows()
        data['patchwam_nonidle_filtered'] = bool(filtered)
        data['patchwam_raw_total_rows'] = int(raw_total)
        data['patchwam_raw_row'] = int(
            self._original_rows(dataset_idx, [int(index)])[0])
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
        data['patchwam_raw_window_rows'] = self._original_rows(
            dataset_idx, window_rows)
        data['patchwam_window_valid'] = np.asarray(window_valid, dtype=bool)
        data['patchwam_window_timestamps'] = np.asarray(window_timestamps,
                                                        dtype=np.float64)
        future_row = index + self.future_offset
        future_valid = bool(self._same_episode_and_dataset(
            future_row, dataset_idx, data))
        data['patchwam_future_row'] = int(future_row) if future_valid else int(
            index)
        data['patchwam_future_valid'] = future_valid
        data['patchwam_raw_future_row'] = int(self._original_rows(
            dataset_idx, [future_row if future_valid else int(index)])[0])

        if self.history_enabled:
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(
                    'PatchWAM history requires a positive dataset fps in '
                    'meta/info.json')
            local_rows, valid = self.history_selector.select(
                frame_index, fps, training=self.training,
                vision_language=False)
            raw_rows = self._history_raw_rows(
                dataset_idx, episode_index, frame_index, index, local_rows)
            data['patchwam_raw_history_rows'] = raw_rows.astype(np.int64)
            if not filtered:
                # Without a filter the recording axis is the dataset axis.
                data['patchwam_history_rows'] = raw_rows.astype(np.int64)
            history_valid, history_timestamps = self._history_clocks(
                local_rows, valid.numpy(), episode_index, dataset_idx,
                frame_index, index, data, timestamp)
            data['patchwam_history_valid'] = history_valid
            data['patchwam_history_timestamps'] = history_timestamps
            # VL history uses its own jitter contract (no world dropout).
            vl_rows, vl_valid = self.history_selector.select(
                frame_index, fps, training=self.training, vision_language=True)
            vl_raw_rows = self._history_raw_rows(
                dataset_idx, episode_index, frame_index, index, vl_rows)
            data['patchwam_raw_vl_history_rows'] = vl_raw_rows.astype(np.int64)
            if not filtered:
                data['patchwam_vl_history_rows'] = vl_raw_rows.astype(np.int64)
            vl_history_valid, vl_timestamps = self._history_clocks(
                vl_rows, vl_valid.numpy(), episode_index, dataset_idx,
                frame_index, index, data, timestamp)
            data['patchwam_vl_history_valid'] = vl_history_valid
            data['patchwam_vl_history_timestamps'] = vl_timestamps
        return self._apply_action_statistics(data)
