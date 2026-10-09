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
from .patchwam_dataset import PatchWAMParquetDataset

__all__ = ['PatchWAMLITParquetDataset']


@DATASETS.register_module()
class PatchWAMLITParquetDataset(PatchWAMParquetDataset):
    """PatchWAM dataset adding the terminal LIT goal state.

    The LIT goal is ``observation.state`` at the exact same episode/root-bounded
    row as the future endpoint (``index + goal_offset``, default = horizon 16),
    never the last action, never ``t + 15``, and never a row from another
    episode or data root. The raw vector and its validity are carried through
    the existing transform pipeline so
    :class:`~fluxvla.transforms.transform_patchwam_lit.PatchWAMLITGoal` can
    normalize them with the state/proprio statistics.
    """

    def __init__(self,
                 *args,
                 goal_offset: Optional[int] = None,
                 goal_state_key: str = 'observation.state',
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        offset = self.horizon if goal_offset is None else int(goal_offset)
        if offset < 1:
            raise ValueError('goal_offset must be positive')
        self.goal_offset = offset
        self.goal_state_key = str(goal_state_key)

    def _goal_state(self, row: int, dataset_idx: int, data: Dict[str, Any]):
        if row < 0 or row >= len(self.dataset):
            return None
        if not self._same_episode_and_dataset(int(row), dataset_idx, data):
            return None
        record = self.dataset[int(row)]
        if self.goal_state_key not in record:
            # The row is inside the episode/root window but the field is
            # missing: a schema error, not a boundary. Marking it invalid would
            # silently remove the goal supervision the recipe promises.
            raise KeyError(
                f'Record {row} has no {self.goal_state_key!r}; the LIT goal '
                'requires the terminal observation.state column')
        return np.asarray(record[self.goal_state_key], dtype=np.float32)

    def _pre_transform_hook(self, data: Dict[str, Any], index: int,
                            dataset_idx: int) -> Dict[str, Any]:
        data = super()._pre_transform_hook(data, index, dataset_idx)
        goal_row = int(index) + self.goal_offset
        goal_state = self._goal_state(goal_row, dataset_idx, data)
        data['patchwam_goal_offset'] = self.goal_offset
        data['patchwam_goal_valid'] = goal_state is not None
        data['patchwam_goal_row'] = goal_row if goal_state is not None else int(
            index)
        data['patchwam_goal_state_raw'] = goal_state
        return data
