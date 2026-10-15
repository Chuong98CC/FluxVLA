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
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from fluxvla.engines import DATASETS

__all__ = ['PatchWAMRoboTwinEvalDataset']


@DATASETS.register_module()
class PatchWAMRoboTwinEvalDataset:
    """RoboTwin evaluation observations for PatchWAM heads.

    The canonical RoboTwin evaluator (the pull request's
    ``RobotwinEvalRunner`` and its in-process ``LocalFluxVLAPolicy``) feeds the
    policy one online observation per prediction. Its ``_build_observation``
    emits the three RoboTwin cameras as top-level ``uint8 [H, W, 3]`` arrays,
    the 14-D ``qpos`` and the raw instruction text. This dataset adapts that
    dict into the native PatchWAM prediction batch, reusing the shared FluxVLA
    transforms named by the config for mosaicking, state normalization and
    action denormalization.

    The trainer composes the same three cameras into the compact 288x256
    mosaic and z-scores proprio with the run's ``dataset_statistics.json``, so
    the config mirrors the training transform chain minus augmentation. The
    released legacy ``dataset_stats.json`` asset uses a different schema and is
    not accepted here; convert it or evaluate a newly trained checkpoint.

    Args:
        transforms (List[Dict]): Transform configs applied to each observation.
            Must produce ``pixel_values`` (the mosaic) and ``states``.
        norm_stats (str | Dict | None): Dataset statistics, either a path or an
            already-loaded payload keyed by ``statistic_name``.
        statistic_name (str): Top-level statistics key written by training.
        img_keys (Sequence[str]): Observation keys holding the camera frames.
        device (str): Device the emitted batch tensors live on.
        state_key (str): Observation key holding the raw proprioception vector.
        model_path (str | None): Accepted for parity with the canonical policy's
            dataset injection; unused by this adapter's transforms.
    """

    def __init__(self,
                 transforms: List[Dict],
                 norm_stats: Union[str, Dict[str, Any], None] = None,
                 statistic_name: str = 'robotwin',
                 img_keys: Optional[Sequence[str]] = None,
                 device: str = 'cuda',
                 state_key: str = 'qpos',
                 model_path: Optional[str] = None) -> None:
        from fluxvla.engines import build_transform_from_cfg

        self.transforms = [
            build_transform_from_cfg(dict(transform))
            for transform in transforms
        ]
        if isinstance(norm_stats, str):
            with open(norm_stats, 'r', encoding='utf-8') as stream:
                norm_stats = json.load(stream)
        self.norm_stats = norm_stats
        self.statistic_name = str(statistic_name)
        self.img_keys = list(
            img_keys
            or ('cam_high', 'cam_left_wrist', 'cam_right_wrist'))
        self.device = str(device)
        self.state_key = str(state_key)
        self.model_path = model_path
        # Delta-action recipes add this back; absolute recipes ignore it.
        self.last_raw_state: Optional[np.ndarray] = None

    def _statistics(self) -> Optional[Dict[str, Any]]:
        if self.norm_stats is None:
            return None
        if self.statistic_name not in self.norm_stats:
            raise KeyError(
                f'Dataset statistics have no {self.statistic_name!r} entry; '
                f'available keys: {sorted(self.norm_stats)}')
        return self.norm_stats[self.statistic_name]

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        for img_key in self.img_keys:
            if img_key not in inputs:
                raise KeyError(
                    f'Missing RoboTwin camera {img_key!r} in the observation')
        if self.state_key not in inputs:
            raise KeyError(
                f'Missing proprioception {self.state_key!r} in the observation')

        self.last_raw_state = np.asarray(inputs[self.state_key],
                                         dtype=np.float32).copy()
        data: Dict[str, Any] = dict(inputs)
        data['norm_stats'] = self._statistics()
        for transform in self.transforms:
            data = transform(data)

        pixel_values = data.get('pixel_values')
        if pixel_values is None:
            raise KeyError(
                'PatchWAM RoboTwin eval transforms must produce `pixel_values`')
        if 'states' not in data:
            raise KeyError(
                'PatchWAM RoboTwin eval transforms must produce `states`')

        images = torch.as_tensor(np.asarray(pixel_values), dtype=torch.float32)
        mask = data.get('img_masks', [True])
        batch = dict(
            images=images.to(self.device).unsqueeze(0),
            img_masks=torch.tensor([list(mask)], dtype=torch.bool,
                                   device=self.device),
            states=torch.as_tensor(
                np.asarray(data['states'], dtype=np.float32),
                device=self.device).unsqueeze(0),
        )
        if 'frame_masks' in data:
            batch['frame_masks'] = torch.as_tensor(
                data['frame_masks'], dtype=torch.bool,
                device=self.device).unsqueeze(0)
        instruction = data.get('task_description',
                               inputs.get('task_description',
                                          'No task description provided'))
        batch['task_description'] = [instruction]
        return batch
