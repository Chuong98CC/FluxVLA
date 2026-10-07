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

from typing import Any, Dict

import numpy as np
import torch

from fluxvla.engines import DATASETS
from .parquet_dataset import LiberoParquetEvalDataset

__all__ = ['PatchWAMLiberoEvalDataset']


@DATASETS.register_module()
class PatchWAMLiberoEvalDataset(LiberoParquetEvalDataset):
    """Native LIBERO eval dataset retaining raw prompts for PatchWAM.

    The base dataset builds a tokenizer-oriented batch and drops
    ``task_description``. PatchWAM conditions on the raw instruction, so this
    subclass runs the same configured transform chain but emits
    ``images``/``img_masks``/``states``/``task_description`` and uses a
    configurable device instead of a hard ``.cuda()``.
    """

    def __init__(self, *args, device: str = 'cuda', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = str(device)

    def __call__(self, inputs: Dict[str, Any]):
        data: Dict[str, Any] = dict(inputs)
        is_new_episode = bool(data.get('is_new_episode', False))
        if is_new_episode:
            self._reset_img_buffer()
        norm_stats = (None if self.norm_stats is None else
                      self.norm_stats[self.norm_stats_key])
        data['norm_stats'] = norm_stats
        for transform in self.transforms:
            data = transform(data)
        images = data.get('pixel_values')
        if images is None:
            raise KeyError('PatchWAM eval transforms must produce `pixel_values`')
        images = torch.as_tensor(images).float()
        mask = data.get('img_masks', [True])
        batch = dict(
            images=images.to(self.device).unsqueeze(0),
            img_masks=torch.tensor([list(mask)], device=self.device),
        )
        if 'frame_masks' in data:
            batch['frame_masks'] = torch.as_tensor(
                data['frame_masks'], dtype=torch.bool,
                device=self.device).unsqueeze(0)
        if 'states' in data:
            batch['states'] = torch.as_tensor(
                np.asarray(data['states'], dtype=np.float32),
                device=self.device).unsqueeze(0)
        instruction = data.get('task_description',
                               inputs.get('task_description',
                                          'No task description provided'))
        batch['task_description'] = [instruction]
        batch['reset_history'] = is_new_episode
        return batch, data.get('replay_img')
