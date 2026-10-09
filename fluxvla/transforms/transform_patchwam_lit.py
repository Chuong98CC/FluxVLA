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
"""LIT goal transform: normalize the terminal state with state statistics."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from fluxvla.engines import TRANSFORMS
from .normalize import NormalizeStatesAndActions

__all__ = ['PatchWAMLITGoal']


@TRANSFORMS.register_module()
class PatchWAMLITGoal:
    """Expose the normalized terminal goal state for the LIT objective.

    Reads the raw terminal ``observation.state`` carried by
    ``PatchWAMLITParquetDataset`` and normalizes it with the **state/proprio**
    statistics (never the action statistics), reusing the native
    ``NormalizeStatesAndActions`` arithmetic so the current state and the goal
    share one normalization. Invalid or out-of-episode goals are emitted as
    zeros with ``goal_valid=False`` and therefore excluded from the
    reconstruction loss.
    """

    def __init__(self,
                 state_key: str = 'proprio',
                 state_dim: Optional[int] = None,
                 norm_type: str = 'mean_std',
                 normalization_epsilon: float = 1e-6,
                 statistics_key: str = 'stats',
                 raw_key: str = 'patchwam_goal_state_raw',
                 valid_key: str = 'patchwam_goal_valid',
                 goal_key: str = 'goal_state',
                 goal_valid_key: str = 'goal_valid') -> None:
        self.state_key = str(state_key)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.norm_type = str(norm_type)
        if self.norm_type not in ('mean_std', 'min_max', 'quantile', 'none'):
            raise ValueError(f'Unsupported norm_type {norm_type!r}')
        self.normalization_epsilon = float(normalization_epsilon)
        if not np.isfinite(self.normalization_epsilon) or \
                self.normalization_epsilon < 0:
            raise ValueError('normalization_epsilon must be finite and >= 0')
        self.statistics_key = str(statistics_key)
        self.raw_key = str(raw_key)
        self.valid_key = str(valid_key)
        self.goal_key = str(goal_key)
        self.goal_valid_key = str(goal_valid_key)
        self._normalizer = NormalizeStatesAndActions(
            state_key=self.state_key, action_key=None, norm_type=self.norm_type,
            normalization_epsilon=self.normalization_epsilon,
            statistics_key=self.statistics_key)

    def _statistics(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        statistics = data.get(self.statistics_key)
        if not isinstance(statistics, dict) or self.state_key not in statistics:
            return None
        return statistics[self.state_key]

    def _pad_or_truncate(self, values: np.ndarray) -> np.ndarray:
        if self.state_dim is None or values.shape[-1] == self.state_dim:
            return values
        if values.shape[-1] > self.state_dim:
            return values[..., :self.state_dim]
        padded = np.zeros((self.state_dim,), dtype=values.dtype)
        padded[:values.shape[-1]] = values
        return padded

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raw = data.get(self.raw_key)
        valid = bool(data.get(self.valid_key, False))
        goal: Optional[np.ndarray] = None
        if raw is not None and valid:
            # A valid terminal goal must be a real, finite state vector; a
            # missing statistics entry is a configuration error rather than an
            # excuse to silently disable goal supervision.
            values = np.asarray(raw, dtype=np.float32).reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError(
                    f'{self.raw_key} contains non-finite goal values')
            if self.norm_type != 'none':
                statistics = self._statistics(data)
                if statistics is None:
                    raise ValueError(
                        f'Missing state statistics {self.state_key!r} under '
                        f'{self.statistics_key!r}; a valid terminal goal '
                        'cannot be normalized without them')
                expected = np.asarray(statistics['mean']).reshape(-1).shape[0]
                if expected != values.shape[0]:
                    raise ValueError(
                        f'Goal width {values.shape[0]} does not match the '
                        f'{self.state_key!r} statistics width {expected}')
            normalized = self._normalizer._normalize_by_type(
                values, statistics if self.norm_type != 'none' else None,
                self.norm_type)
            goal = np.asarray(normalized, dtype=np.float32).reshape(-1)
        if goal is None:
            if self.state_dim is None:
                raise ValueError(
                    'A zero-length or invalid goal needs an explicit '
                    '`state_dim` to be materialized')
            goal = np.zeros((self.state_dim,), dtype=np.float32)
            valid = False
        elif self.state_dim is not None:
            goal = self._pad_or_truncate(goal)
        data[self.goal_key] = goal.astype(np.float32, copy=False)
        data[self.goal_valid_key] = np.asarray(valid, dtype=bool)
        return data
