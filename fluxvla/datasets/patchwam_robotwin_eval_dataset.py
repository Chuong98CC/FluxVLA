"""RoboTwin evaluation observations for PatchWAM heads.

The canonical RoboTwin evaluator (the pull request's ``RobotwinEvalRunner`` and
its in-process ``LocalFluxVLAPolicy``) feeds the policy one online observation
per prediction. Its ``_build_observation`` emits the three RoboTwin cameras as
top-level ``uint8 [H, W, 3]`` arrays, the 14-D ``qpos`` and the raw instruction
text. This module adapts that dict into the native PatchWAM prediction batch and
hosts the one evaluation-only frame transform the training pipeline has no
equivalent for.

The trainer composes the same three cameras into the compact 288x256 mosaic,
snapshots an unaugmented head view for VL conditioning and z-scores proprio
with the run's ``dataset_statistics.json``, so the evaluation config mirrors
the training transform chain minus augmentation.

``PatchWAMEvalFrames`` lives here rather than in ``transform_patchwam`` so the
training transform module stays byte-identical; it is still a registered
``TRANSFORMS`` entry and is named directly by the evaluation configs.
"""

import json
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from fluxvla.engines import DATASETS, TRANSFORMS

__all__ = ['PatchWAMEvalFrames', 'PatchWAMRoboTwinEvalDataset']


@TRANSFORMS.register_module()
class PatchWAMEvalFrames:
    """Scale online camera frames to unambiguous ``[0, 1]`` float views.

    ``ProcessEvalInputs`` yields PIL/``uint8`` frames. Scaling must be decided
    by dtype, not by intensity: a frame whose maximum sample is 1 is a dark
    frame, not already-normalized white. ``PatchWAMEvalPack`` alone cannot tell
    those apart, so the conversion happens here first.

    Writes:
        ``pixel_values``: the per-camera float ``CHW`` frames consumed by
        ``PatchWAMEvalPack``.
        ``world_frames``: ``[V, 1, 3, H, W]`` in ``[0, 1]``, the unaugmented
        current views ``PatchWAMCleanVL`` snapshots for VL conditioning.
    """

    def __init__(self, frames_key: str = 'pixel_values') -> None:
        self.frames_key = str(frames_key)

    @staticmethod
    def _to_float_chw(frame: Any) -> torch.Tensor:
        array = np.asarray(frame)
        if array.dtype == np.uint8:
            tensor = torch.tensor(array, dtype=torch.float32) / 255.0
        else:
            tensor = torch.as_tensor(np.asarray(array, dtype=np.float32))
            if tensor.numel() and (float(tensor.min()) < 0.0
                                   or float(tensor.max()) > 1.0):
                raise ValueError(
                    'Evaluation frames must be uint8 or float in [0, 1], got '
                    f'[{float(tensor.min())}, {float(tensor.max())}]')
        if tensor.ndim == 3 and tensor.shape[-1] == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError('Evaluation frames must be CHW or HWC RGB')
        return tensor.clamp(0.0, 1.0).contiguous()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frames = data.get(self.frames_key)
        if frames is None or len(frames) == 0:
            raise ValueError(
                f'PatchWAMEvalFrames needs per-camera frames in '
                f'{self.frames_key!r}')
        clips = [self._to_float_chw(frame) for frame in frames]
        data[self.frames_key] = clips
        data['world_frames'] = torch.stack(
            [clip.unsqueeze(0) for clip in clips], dim=0)
        return data


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
    mosaic, snapshots the unaugmented head view for VL conditioning and
    z-scores proprio with the run's ``dataset_statistics.json``, so the config
    mirrors the training transform chain minus augmentation. The released
    legacy ``dataset_stats.json`` asset uses a different schema and is not
    supported here.

    Args:
        transforms (List[Dict]): Transform configs applied to each observation.
            Must produce ``pixel_values`` (the mosaic), ``states`` and, for
            image-conditioned encoders, ``vl_current``.
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

        raw_state = np.asarray(inputs[self.state_key], dtype=np.float32)
        self.last_raw_state = raw_state.copy()
        data: Dict[str, Any] = dict(inputs)
        # ``NormalizeStatesAndActions`` reads the raw sample from ``states``
        # (its ``state_key`` selects the matching statistics entry) and the
        # statistics from ``statistics_key``; both are primed here.
        data['states'] = raw_state
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
        # Image-conditioned text encoders require the unaugmented current view;
        # ``PatchWAMCleanVL`` emits it together with empty history fields. The
        # canonical policy forwards this batch straight to ``predict_action``,
        # so every tensor carries its leading batch axis here.
        for key in ('vl_current', 'vl_history', 'vl_history_valid'):
            if key in data:
                batch[key] = torch.as_tensor(data[key]).unsqueeze(0).to(
                    self.device)
        if 'frame_masks' in data:
            batch['frame_masks'] = torch.as_tensor(
                data['frame_masks'], dtype=torch.bool,
                device=self.device).unsqueeze(0)
        instruction = data.get('task_description',
                               inputs.get('task_description',
                                          'No task description provided'))
        batch['task_description'] = [instruction]
        return batch
