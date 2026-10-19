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
"""PatchWAM image/packing/augmentation transforms.

Expected input contract (all tensors float unless noted):

* ``world_frames``: ``[V, T, 3, H, W]`` decoded camera clips in ``[0, 1]``,
  row 0 is the current observation and row ``horizon`` the future endpoint.
* ``history_frames``: ``[K, 3, H, W]`` head-camera history in ``[0, 1]``
  (optional).
* ``patchwam_future_valid``: bool scalar - the future endpoint row is inside the
  episode/root window.
* ``patchwam_history_valid``: ``[K]`` bool - history slots sampled on the
  original recording clock.
* ``patchwam_episode_index`` / ``patchwam_timestamp`` / ``patchwam_reset``:
  episode identity, seconds, and reset signal for online history buffering.

Produced fields: ``world_endpoints`` ``[V, 2, 3, H, W]``, ``camera_video``
``[V, 3, 2, H, W]`` and/or ``video`` mosaic ``[3, 2, H, W]`` in ``[-1, 1]``,
``vl_current`` / ``vl_history`` (clean, pre-augmentation), ``history_video`` /
``history_valid``, ``frame_masks`` ``[2]`` and ``img_masks`` ``[V]``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from fluxvla.engines import TRANSFORMS
from ..datasets.utils.video_decode import decode_video_frames
from ..models.third_party_models.patchwam.data import (AppearanceRandomizer,
                                                       CausalObservationBuffer,
                                                       ClipAugment,
                                                       compose_cameras,
                                                       resize_clip)

__all__ = [
    'PatchWAMCleanVL',
    'PatchWAMComposeWorld',
    'PatchWAMDecodeFrames',
    'PatchWAMEndpoints',
    'PatchWAMEvalPack',
    'PatchWAMFkActionMetadata',
    'PatchWAMHistoryBuffer',
    'PatchWAMKeep',
    'PatchWAMRawInputs',
    'PatchWAMResizeCameras',
    'PatchWAMSelectHistory',
    'PatchWAMSelectWindow',
    'PatchWAMWorldAugment',
    'PatchWAMWorldHistory',
]


def _as_views(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f'{name} must be a torch tensor')
    return value


def _scalar(value, default):
    if value is None:
        return default
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f'Expected a scalar row value, got shape {tuple(tensor.shape)}')
    return tensor.reshape(()).item()


def _require_original_row_tensor(data: Dict[str, Any], frames: torch.Tensor,
                                 name: str) -> None:
    """Require the unfiltered original-row tensor when a filter is active.

    Row metadata is emitted on the raw recording axis (per-root offsets
    preserved), so a non-idle-filtered sample can only be gathered from a
    tensor that spans every original row. Anything shorter is the filtered
    sequence and would silently select the wrong frames.
    """
    if not data.get('patchwam_nonidle_filtered'):
        return
    total = data.get('patchwam_raw_total_rows')
    if total is None:
        raise ValueError(
            f'{name} must be the unfiltered original-row tensor when a '
            'non-idle filter is active, but the sample carries no '
            'patchwam_raw_total_rows')
    if int(frames.shape[0]) != int(total):
        raise ValueError(
            f'{name} must be the unfiltered original-row tensor when a '
            f'non-idle filter is active: got {int(frames.shape[0])} rows but '
            f'the recording has {int(total)}')


def _first_present(data: Dict[str, Any], *keys, default=None):
    """First key that is present and not None (explicit raw keys win)."""
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return default


# Canonical dataset-axis history keys and their original-row companions. Only
# these are paired: an arbitrary custom ``rows_key`` is never rewritten onto a
# raw key the caller did not ask for.
_RAW_HISTORY_ROW_KEYS = {
    'patchwam_history_rows': 'patchwam_raw_history_rows',
    'patchwam_vl_history_rows': 'patchwam_raw_vl_history_rows',
}


@TRANSFORMS.register_module()
class PatchWAMKeep:
    """Prune the sample dict to the configured collator/model contract."""

    def __init__(self, keys: Sequence[str], require: bool = True):
        self.keys = list(keys)
        self.require = bool(require)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        missing = [key for key in self.keys if key not in data]
        if self.require and missing:
            raise ValueError(f'Missing PatchWAM sample keys: {missing}')
        return {key: data[key] for key in self.keys if key in data}


@TRANSFORMS.register_module()
class PatchWAMRawInputs:
    """Map raw parquet state to the native ``states`` key in place.

    Unlike ``ProcessParquetInputs`` this preserves the dataset's temporal
    metadata (window/history rows and timestamps, statistics, info) so the
    PatchWAM decoder and selectors can consume it. ``actions`` /
    ``action_masks`` / ``stats`` / ``info`` / ``task_description`` are already
    produced by ``ParquetDataset``.
    """

    def __init__(self,
                 state_key: str = 'observation.state',
                 out_state_key: str = 'states'):
        self.state_key = state_key
        self.out_state_key = out_state_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.state_key not in data:
            raise KeyError(f'Raw sample is missing {self.state_key!r}')
        data[self.out_state_key] = np.asarray(data[self.state_key],
                                              dtype=np.float32)
        data.setdefault('task_description', data.get('task_description', ''))
        return data


@TRANSFORMS.register_module()
class PatchWAMFkActionMetadata:
    """Attach FK normalization and episode-arm metadata to a sample.

    The mean/std values come from the same effective action statistics the
    immediately preceding :class:`NormalizeStatesAndActions` transform used, so
    the FK objective can invert normalization exactly. This transform is
    training-only and is absent from inference recipes.
    """

    def __init__(
        self,
        action_key: str = 'action',
        action_dim: int = 14,
        statistics_key: str = 'stats',
        normalization_epsilon: float = 1e-8,
        require_active_arm: bool = True,
        mean_key: str = 'fk_action_mean',
        std_key: str = 'fk_action_std',
        epsilon_key: str = 'fk_action_epsilon',
    ) -> None:
        if not isinstance(action_dim, int) or isinstance(action_dim, bool):
            raise ValueError('action_dim must be a positive integer')
        if action_dim < 1:
            raise ValueError('action_dim must be a positive integer')
        epsilon = float(normalization_epsilon)
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError(
                'normalization_epsilon must be finite and nonnegative')
        self.action_key = str(action_key)
        self.action_dim = int(action_dim)
        self.statistics_key = str(statistics_key)
        self.normalization_epsilon = epsilon
        self.require_active_arm = bool(require_active_arm)
        self.mean_key = str(mean_key)
        self.std_key = str(std_key)
        self.epsilon_key = str(epsilon_key)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        active = data.get('active_arm')
        if active is None:
            if self.require_active_arm:
                raise ValueError(
                    'FK-enabled samples require an `active_arm` label from '
                    'the dataset episode lookup')
        else:
            active = np.asarray(active)
            if active.shape != (2,):
                raise ValueError('active_arm must contain two booleans')
            if active.dtype != np.bool_:
                raise ValueError(
                    'active_arm must be a boolean [2] array, got '
                    f'{active.dtype}')
            data['active_arm'] = active.astype(bool, copy=False)

        statistics = data.get(self.statistics_key)
        if not isinstance(statistics, Mapping):
            raise ValueError(
                f'FK action metadata requires {self.statistics_key!r} '
                'statistics')
        entry = statistics.get(self.action_key)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f'FK action metadata requires statistics for '
                f'{self.action_key!r}')
        for field in ('mean', 'std'):
            if field not in entry or entry[field] is None:
                raise ValueError(
                    f'FK action metadata requires statistics.{self.action_key}'
                    f'.{field}')
        mean = np.asarray(entry['mean'], dtype=np.float32).reshape(-1)
        std = np.asarray(entry['std'], dtype=np.float32).reshape(-1)
        for name, value in (('mean', mean), ('std', std)):
            if value.shape != (self.action_dim,):
                raise ValueError(
                    f'Action statistic {name} has width {value.shape[0]}; '
                    f'expected {self.action_dim}')
            if not np.isfinite(value).all():
                raise ValueError(
                    f'Action statistic {name} contains non-finite values')
        if (std < 0).any():
            raise ValueError('Action standard deviation must be nonnegative')
        data[self.mean_key] = mean.copy()
        data[self.std_key] = std.copy()
        data[self.epsilon_key] = np.asarray(
            self.normalization_epsilon, dtype=np.float32)
        return data


@TRANSFORMS.register_module()
class PatchWAMResizeCameras:
    """Resize decoded world/history cameras to their source geometry.

    Applied before augmentation so LIBERO/RobotWin/RoboDojo each augment at
    their recipe geometry (224x224, 240x320, 256x256). Clean VL frames are left
    at full resolution for ``PatchWAMCleanVL`` to resize.
    """

    def __init__(self, camera_size: Sequence[int], history: bool = True):
        if len(camera_size) != 2:
            raise ValueError('camera_size must be [height, width]')
        self.camera_size = (int(camera_size[0]), int(camera_size[1]))
        self.history = bool(history)

    def _resize(self, frames: torch.Tensor) -> torch.Tensor:
        # [..., C, H, W]
        if frames.numel() == 0:
            return frames
        lead = frames.shape[:-3]
        flat = frames.reshape(-1, *frames.shape[-3:]).float()
        resized = F.interpolate(flat, size=self.camera_size, mode='bilinear',
                                align_corners=False, antialias=True)
        return resized.reshape(*lead, *resized.shape[-3:])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        world = data.get('world_frames')
        if world is not None:
            data['world_frames'] = self._resize(world).contiguous()
        if self.history:
            history = data.get('history_frames')
            if history is not None:
                data['history_frames'] = self._resize(history).contiguous()
        return data


@TRANSFORMS.register_module()
class PatchWAMSelectWindow:
    """Slice the ``[V, T, 3, H, W]`` world clip from a decoded episode tensor.

    ``decoded_frames`` is ``[N, V, 3, H, W]`` (or ``[N, 3, H, W]``) indexed by
    absolute *original-recording* row: the dataset row when no non-idle filter
    is active, and the unfiltered per-root row (offsets preserved across roots)
    when one is. The explicit ``patchwam_raw_window_rows`` /
    ``patchwam_raw_row`` keys are preferred and take precedence over the
    dataset-axis keys. ``horizon`` is the endpoint offset.
    """

    def __init__(self, horizon: int = 16, decoded_key: str = 'decoded_frames',
                 out_key: str = 'world_frames'):
        if horizon < 1:
            raise ValueError('horizon must be positive')
        self.horizon = int(horizon)
        self.decoded_key = decoded_key
        self.out_key = out_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        frames = data.get(self.decoded_key)
        if frames is None:
            raise ValueError(f'{self.decoded_key} is required to select a window')
        _require_original_row_tensor(data, frames, self.decoded_key)
        row = int(_scalar(
            _first_present(data, 'patchwam_raw_row', 'patchwam_row'), 0))
        window_rows = _first_present(data, 'patchwam_raw_window_rows',
                                     'patchwam_window_rows')
        if window_rows is not None:
            # Episode/root-bounded rows chosen by the dataset hook; a tail
            # repeats the last valid row and is flagged invalid.
            window_rows = torch.as_tensor(window_rows, dtype=torch.long)
            if window_rows.numel() != self.horizon + 1:
                raise ValueError(
                    f'patchwam_window_rows must have {self.horizon + 1} rows')
            if window_rows.max() >= frames.shape[0]:
                raise ValueError('window rows fall outside the decoded tensor')
            window = frames[window_rows]
            data['world_window_valid'] = torch.as_tensor(
                data.get('patchwam_window_valid',
                         torch.ones_like(window_rows, dtype=torch.bool)),
                dtype=torch.bool)
        else:
            stop = row + self.horizon + 1
            if stop > frames.shape[0]:
                raise ValueError(
                    f'decoded_frames has {frames.shape[0]} rows but rows '
                    f'[{row}, {stop}) are required')
            window = frames[row:stop]
        if window.ndim == 4:
            window = window.unsqueeze(1)
        if window.ndim != 5:
            raise ValueError('decoded_frames must be [N, V, 3, H, W] or [N, 3, H, W]')
        data[self.out_key] = window.permute(1, 0, 2, 3, 4).contiguous()
        return data


@TRANSFORMS.register_module()
class PatchWAMDecodeFrames:
    """Decode only the requested episode-local frames from LeRobot videos.

    Uses the native ``decode_video_frames`` helper and the same path template as
    ``ProcessParquetInputs`` so no concatenated dataset tensor is required. The
    dataset hook provides ``patchwam_window_timestamps`` / validity and the
    world + VL history timestamps; invalid tail rows repeat the last valid frame
    and stay masked.
    """

    def __init__(self,
                 video_keys: Sequence[str],
                 tolerance_s: float = 0.1,
                 backend: Optional[str] = None,
                 head_index: int = 0,
                 window_key: str = 'patchwam_window_timestamps',
                 history_key: str = 'patchwam_history_timestamps',
                 vl_history_key: str = 'patchwam_vl_history_timestamps',
                 vl_history_valid_key: str = 'patchwam_vl_history_valid',
                 vl_out_key: str = 'vl_history_frames'):
        if not video_keys:
            raise ValueError('video_keys must contain at least one camera key')
        self.video_keys = list(video_keys)
        self.tolerance_s = float(tolerance_s)
        self.backend = backend
        self.head_index = int(head_index)
        self.window_key = window_key
        self.history_key = history_key
        self.vl_history_key = vl_history_key
        self.vl_history_valid_key = vl_history_valid_key
        self.vl_out_key = vl_out_key

    def _video_path(self, data: Dict[str, Any], video_key: str) -> str:
        info = data['info']
        episode_chunk = int(data['episode_index']) // int(info['chunks_size'])
        path = os.path.join(
            data['data_root'],
            info['video_path'].format(episode_chunk=episode_chunk,
                                      video_key=video_key,
                                      episode_index=int(
                                          data['episode_index'])))
        if not os.path.exists(path):
            raise FileNotFoundError(f'Video file not found: {path}')
        return path

    @staticmethod
    def _to_unit(frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float()
        if frames.numel() and frames.max() > 1.001:
            frames = frames / 255.0
        return frames.clamp(0, 1)

    def _decode(self, path: str, timestamps: Sequence[float]) -> torch.Tensor:
        unique = sorted(set(float(ts) for ts in timestamps))
        frames = self._to_unit(
            decode_video_frames(path, unique, self.tolerance_s,
                                backend=self.backend))
        index = {ts: position for position, ts in enumerate(unique)}
        positions = [
            index[min(unique, key=lambda candidate: abs(candidate - ts))]
            for ts in timestamps
        ]
        return frames[torch.tensor(positions, dtype=torch.long)]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        window_ts = [float(ts) for ts in data[self.window_key]]
        views = []
        for video_key in self.video_keys:
            path = self._video_path(data, video_key)
            views.append(self._decode(path, window_ts))
        data['world_frames'] = torch.stack(views)
        data['world_window_valid'] = torch.as_tensor(
            data.get('patchwam_window_valid',
                     torch.ones(len(window_ts), dtype=torch.bool)),
            dtype=torch.bool)
        data['world_camera_valid'] = torch.as_tensor(
            data.get('world_camera_valid',
                     torch.ones(len(self.video_keys), dtype=torch.bool)),
            dtype=torch.bool)
        history_ts = data.get(self.history_key)
        if history_ts is not None and len(history_ts):
            history_path = self._video_path(data,
                                            self.video_keys[self.head_index])
            history = self._decode(history_path,
                                   [float(ts) for ts in history_ts])
            valid = torch.as_tensor(data.get('patchwam_history_valid'),
                                    dtype=torch.bool)
            if valid.shape != (history.shape[0],):
                raise ValueError('patchwam_history_valid must match history rows')
            data['history_frames'] = history.masked_fill(
                ~valid[:, None, None, None], 0).contiguous()
        # VL history uses its own (distinct) sampling clock and decode.
        vl_ts = data.get(self.vl_history_key)
        if vl_ts is not None and len(vl_ts):
            vl_path = self._video_path(data,
                                       self.video_keys[self.head_index])
            vl_frames = self._decode(vl_path, [float(ts) for ts in vl_ts])
            vl_valid = torch.as_tensor(data.get(self.vl_history_valid_key),
                                       dtype=torch.bool)
            if vl_valid.shape != (vl_frames.shape[0],):
                raise ValueError(
                    f'{self.vl_history_valid_key} must match VL history rows')
            data[self.vl_out_key] = vl_frames.masked_fill(
                ~vl_valid[:, None, None, None], 0).contiguous()
        return data


@TRANSFORMS.register_module()
class PatchWAMSelectHistory:
    """Gather head-camera history frames from a decoded episode tensor.

    Rows are original-recording rows: ``patchwam_raw_history_rows`` wins over
    the dataset-axis ``patchwam_history_rows``. ``decoded_frames`` must
    therefore span the unfiltered recording when a non-idle filter is active.

    ``raw_rows_key`` defaults to the companion of the canonical world/VL row
    keys only; a custom ``rows_key`` keeps its own key unless the caller passes
    an explicit ``raw_rows_key``.
    """

    def __init__(self,
                 head_index: int = 0,
                 decoded_key: str = 'decoded_frames',
                 out_key: str = 'history_frames',
                 rows_key: str = 'patchwam_history_rows',
                 raw_rows_key: Optional[str] = None,
                 valid_key: str = 'patchwam_history_valid'):
        self.head_index = int(head_index)
        self.decoded_key = decoded_key
        self.out_key = out_key
        self.rows_key = rows_key
        self.raw_rows_key = (raw_rows_key if raw_rows_key is not None
                             else _RAW_HISTORY_ROW_KEYS.get(str(rows_key)))
        self.valid_key = valid_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        preferred = ([self.raw_rows_key] if self.raw_rows_key is not None
                     else []) + [self.rows_key]
        rows = _first_present(data, *preferred)
        valid = data.get(self.valid_key)
        if rows is None:
            return data
        frames = data.get(self.decoded_key)
        if frames is None:
            raise ValueError(f'{self.decoded_key} is required to gather history')
        _require_original_row_tensor(data, frames, self.decoded_key)
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        if valid is None:
            valid = torch.ones_like(rows, dtype=torch.bool)
        valid = torch.as_tensor(valid, dtype=torch.bool).reshape(-1)
        if valid.shape != rows.shape:
            raise ValueError(f'{self.valid_key} must match {self.rows_key}')
        if rows.numel() and (rows.min() < 0 or rows.max() >= frames.shape[0]):
            raise ValueError('history rows fall outside the decoded tensor')
        gathered = frames[rows] if rows.numel() else frames[:0]
        if gathered.ndim == 5:
            gathered = gathered[:, self.head_index]
        history = torch.where(valid[:, None, None, None], gathered,
                              torch.zeros_like(gathered))
        data[self.out_key] = history.contiguous()
        data[self.valid_key] = valid
        return data


@TRANSFORMS.register_module()
class PatchWAMEndpoints:
    """Select the current/future endpoint pair from a decoded world clip."""

    def __init__(self, horizon: int = 16, out_key: str = 'world_endpoints'):
        if horizon < 1:
            raise ValueError('horizon must be positive')
        self.horizon = int(horizon)
        self.out_key = out_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        world = _as_views(data['world_frames'], 'world_frames')
        if world.ndim != 5 or world.shape[2] != 3:
            raise ValueError('world_frames must be [V, T, 3, H, W]')
        if world.shape[1] <= self.horizon:
            raise ValueError(
                f'world_frames has T={world.shape[1]} rows but the endpoint '
                f'horizon is {self.horizon}')
        endpoints = world[:, (0, self.horizon)].contiguous()
        data[self.out_key] = endpoints
        views = endpoints.shape[0]
        window_valid = data.get('world_window_valid')
        if window_valid is not None:
            window_valid = torch.as_tensor(window_valid, dtype=torch.bool)
            if window_valid.numel() != self.horizon + 1:
                raise ValueError(
                    f'world_window_valid must have {self.horizon + 1} entries')
            future_valid = bool(window_valid[-1])
        else:
            future_valid = bool(data.get('patchwam_future_valid', True))
        camera_valid = data.get('world_camera_valid')
        if camera_valid is None:
            camera_valid = torch.ones(views, dtype=torch.bool)
        else:
            camera_valid = torch.as_tensor(camera_valid, dtype=torch.bool)
            if camera_valid.shape != (views,):
                raise ValueError('world_camera_valid must be [V]')
        data['img_masks'] = camera_valid
        data['frame_masks'] = torch.tensor([True, future_valid],
                                           dtype=torch.bool)
        return data


@TRANSFORMS.register_module()
class PatchWAMCleanVL:
    """Snapshot clean VL current/history images before world augmentation."""

    def __init__(self,
                 image_size: int = 448,
                 head_index: int = 0,
                 history_key: str = 'history_frames',
                 valid_key: str = 'patchwam_history_valid'):
        if image_size < 1:
            raise ValueError('image_size must be positive')
        self.image_size = int(image_size)
        self.head_index = int(head_index)
        self.history_key = history_key
        self.valid_key = valid_key

    def _clean(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.unsqueeze(0) if frame.ndim == 3 else frame
        resized = F.interpolate(frame.float(), size=(self.image_size,) * 2,
                                mode='bilinear', align_corners=False,
                                antialias=True)
        return (resized * 2 - 1).contiguous()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        world = _as_views(data['world_frames'], 'world_frames')
        if self.head_index >= world.shape[0]:
            raise ValueError('head_index is outside the camera views')
        current = world[self.head_index, 0]
        data['vl_current'] = self._clean(current)[0]
        history = data.get(self.history_key)
        if history is None:
            data['vl_history'] = current.new_empty((0, 3, self.image_size,
                                                    self.image_size))
            data['vl_history_valid'] = torch.zeros(0, dtype=torch.bool)
            return data
        history = _as_views(history, self.history_key)
        if history.ndim != 4 or history.shape[1] != 3:
            raise ValueError(f'{self.history_key} must be [K, 3, H, W]')
        valid = data.get(self.valid_key)
        if valid is None:
            valid = torch.ones(history.shape[0], dtype=torch.bool)
        valid = torch.as_tensor(valid, dtype=torch.bool)
        if valid.shape != (history.shape[0],):
            raise ValueError(f'{self.valid_key} must be [K]')
        safe = torch.where(valid[:, None, None, None], history,
                           torch.zeros_like(history))
        vl_history = self._clean(safe)
        vl_history = vl_history.masked_fill(~valid[:, None, None, None], 0)
        data['vl_history'] = vl_history
        data['vl_history_valid'] = valid
        return data


@TRANSFORMS.register_module()
class PatchWAMWorldAugment:
    """Source clip/appearance augmentation with the original draw order.

    One probability draw per sample, independent parameter draws per camera, and
    shared temporal parameters within a camera (the clips are whole TCHW
    sequences). Clean ``vl_*`` images are never passed to the augmenter.
    """

    _APPEARANCE_KEYS = {'photometric', 'style', 'fourier', 'background'}

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 kind: Optional[str] = None,
                 enabled: bool = True):
        self.enabled = bool(enabled)
        self.config = dict(config or {})
        if kind is None:
            kind = ('appearance'
                    if self._APPEARANCE_KEYS & set(self.config) else 'clip')
        if kind not in {'clip', 'appearance'}:
            raise ValueError('kind must be clip or appearance')
        self.kind = kind
        if self.enabled:
            self.augment = (AppearanceRandomizer(**self.config)
                            if kind == 'appearance' else
                            ClipAugment(**self.config))
        else:
            self.augment = None

    def _collect(self, data: Dict[str, Any]):
        endpoints = _as_views(data['world_endpoints'], 'world_endpoints')
        if endpoints.ndim != 5 or endpoints.shape[2] != 3:
            raise ValueError('world_endpoints must be [V, 2, 3, H, W]')
        clips: List[torch.Tensor] = [
            endpoints[index] for index in range(endpoints.shape[0])
        ]
        labels = [('world', index) for index in range(endpoints.shape[0])]
        history = data.get('history_frames')
        if history is not None:
            history = _as_views(history, 'history_frames')
            if history.ndim != 4 or history.shape[1] != 3:
                raise ValueError('history_frames must be [K, 3, H, W]')
            for index in range(history.shape[0]):
                clips.append(history[index].unsqueeze(0))
                labels.append(('history', index))
        return clips, labels

    def _scatter(self, data, augmented, labels, views: int, history_count: int):
        world = []
        history = []
        for clip, label in zip(augmented, labels):
            kind, index = label
            if kind == 'world':
                world.append((index, clip))
            else:
                history.append((index, clip))
        if world:
            ordered = [clip for _, clip in sorted(world)]
            data['world_endpoints'] = torch.stack(ordered)
        if history_count:
            ordered = [clip[0] for _, clip in sorted(history)]
            data['history_frames'] = torch.stack(ordered)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.augment is None or not data.get('patchwam_augment', True):
            return data
        clips, labels = self._collect(data)
        history_count = sum(1 for kind, _ in labels if kind == 'history')
        views = len(labels) - history_count
        augmented = self.augment(clips)
        if len(augmented) != len(clips):
            raise ValueError('The augmenter changed the number of clips')
        self._scatter(data, augmented, labels, views, history_count)
        return data


@TRANSFORMS.register_module()
class PatchWAMComposeWorld:
    """Mosaic or separate-view world video in ``[-1, 1]`` from endpoints."""

    def __init__(self,
                 layout: str = 'horizontal',
                 video_size: Sequence[int] = (224, 448),
                 robotwin_layout: str = 'compact_288x256',
                 separate_views: bool = False):
        self.layout = layout
        self.video_size = tuple(video_size)
        self.robotwin_layout = robotwin_layout
        self.separate_views = bool(separate_views)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        endpoints = _as_views(data['world_endpoints'],
                              'world_endpoints')
        if endpoints.ndim != 5 or endpoints.shape[2] != 3:
            raise ValueError('world_endpoints must be [V, 2, 3, H, W]')
        if self.separate_views:
            data['camera_video'] = torch.stack([
                (clip * 2 - 1).permute(1, 0, 2, 3).contiguous()
                for clip in endpoints
            ])
        else:
            # The mosaic is a single fused view.
            data['img_masks'] = torch.ones(1, dtype=torch.bool)
        data['video'] = compose_cameras(list(endpoints), self.layout,
                                        self.video_size,
                                        self.robotwin_layout)
        return data


@TRANSFORMS.register_module()
class PatchWAMEvalPack:
    """Pack per-camera eval observations into a PatchWAM mosaic batch.

    Reads ``images`` (a list of per-camera ``CHW``/``HWC`` frames from the
    native eval transforms), resizes each to ``camera_size``, composes the
    source mosaic in ``[-1, 1]``, and writes ``pixel_values`` / ``img_masks``
    for the native eval dataset to batch. ``lang_tokens`` / ``lang_masks`` are
    placeholders because PatchWAM conditions on ``task_description`` text.
    """

    def __init__(self,
                 layout: str = 'horizontal',
                 video_size: Sequence[int] = (224, 448),
                 camera_size: Sequence[int] = (224, 224),
                 robotwin_layout: str = 'compact_288x256',
                 head_index: int = 0):
        self.layout = layout
        self.video_size = tuple(video_size)
        self.camera_size = tuple(camera_size)
        self.robotwin_layout = robotwin_layout
        self.head_index = int(head_index)

    @staticmethod
    def _to_chw(image) -> torch.Tensor:
        tensor = (image if isinstance(image, torch.Tensor) else
                  torch.as_tensor(np.asarray(image)))
        tensor = tensor.float()
        if tensor.numel() and tensor.max() > 1.001:
            tensor = tensor / 255.0
        if tensor.ndim == 3 and tensor.shape[-1] == 3 and tensor.shape[
                0] != 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError('Eval images must be CHW or HWC RGB frames')
        return tensor.clamp(0, 1)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        images = data.get('images', data.get('pixel_values'))
        if not images:
            raise ValueError('Eval inputs must provide per-camera images')
        clips = []
        for image in images:
            frame = self._to_chw(image)
            resized = F.interpolate(frame.unsqueeze(0),
                                    size=self.camera_size,
                                    mode='bilinear',
                                    align_corners=False,
                                    antialias=True)
            clips.append(resized[0].unsqueeze(0))
        mosaic = compose_cameras(clips, self.layout, self.video_size,
                                 self.robotwin_layout)
        data['pixel_values'] = mosaic
        data['img_masks'] = [True]
        data.setdefault('lang_tokens', [0])
        data.setdefault('lang_masks', [True])
        return data


@TRANSFORMS.register_module()
class PatchWAMWorldHistory:
    """Assemble world-model history ``[K, 3, H, W]`` in ``[-1, 1]``."""

    def __init__(self, head_index: int = 0, history_key: str = 'history_frames'):
        self.head_index = int(head_index)
        self.history_key = history_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        history = data.get(self.history_key)
        if history is None:
            endpoints = data['world_endpoints']
            template = endpoints[self.head_index, 0]
            data['history_video'] = template.new_empty((0, 3, *template.shape[-2:]))
            data['history_valid'] = torch.zeros(0, dtype=torch.bool)
            return data
        history = _as_views(history, self.history_key)
        if history.ndim != 4 or history.shape[1] != 3:
            raise ValueError(f'{self.history_key} must be [K, 3, H, W]')
        valid = data.get('patchwam_history_valid')
        if valid is None:
            valid = torch.ones(history.shape[0], dtype=torch.bool)
        valid = torch.as_tensor(valid, dtype=torch.bool)
        if valid.shape != (history.shape[0],):
            raise ValueError('patchwam_history_valid must be [K]')
        world = (history * 2 - 1).masked_fill(~valid[:, None, None, None], 0)
        data['history_video'] = world.contiguous()
        data['history_valid'] = valid
        return data


@TRANSFORMS.register_module()
class PatchWAMHistoryBuffer:
    """Online causal head-camera history with episode reset.

    Must be called once per recorded observation step on the original clock.
    Requires ``patchwam_episode_index``, ``patchwam_timestamp``, and the current
    head frame (``image`` or ``world_frames[head_index, 0]``). Raises clearly if
    the cadence/reset signals are missing rather than returning stale history.

    ``frame_layout`` selects the input layout: ``c_h_w``, ``t_c_h_w``,
    ``v_c_h_w``, ``v_t_c_h_w``, or ``auto``. When ``vl_image_size`` is set the
    clean resized VL history is emitted too, matching the training VL contract.
    """

    def __init__(self,
                 slots: int = 20,
                 interval_s: float = 1.0,
                 tolerance_s: float = 0.05,
                 head_index: int = 0,
                 image_key: str = 'world_frames',
                 frame_layout: str = 'auto',
                 vl_image_size: Optional[int] = None,
                 world_size: Optional[Sequence[int]] = None,
                 input_range: str = 'auto'):
        self.buffer = CausalObservationBuffer(slots=slots,
                                              interval_s=interval_s,
                                              tolerance_s=tolerance_s)
        self.head_index = int(head_index)
        self.image_key = image_key
        if frame_layout not in {'auto', 'c_h_w', 't_c_h_w', 'v_c_h_w',
                                'v_t_c_h_w'}:
            raise ValueError(f'Unsupported frame_layout: {frame_layout!r}')
        self.frame_layout = frame_layout
        self.vl_image_size = (None if vl_image_size is None else
                              int(vl_image_size))
        self.world_size = (None if world_size is None else
                           (int(world_size[0]), int(world_size[1])))
        if input_range not in {'auto', 'unit', 'minus_one_one'}:
            raise ValueError(f'Unsupported input_range: {input_range!r}')
        self.input_range = input_range

    def _to_signed(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.float()
        if self.input_range == 'minus_one_one':
            return frame
        if self.input_range == 'unit':
            return frame.clamp(0, 1) * 2 - 1
        if frame.numel() and frame.max() > 1.001:
            frame = frame / 255.0
        if frame.numel() and frame.min() >= -1e-3:
            frame = frame * 2 - 1
        return frame.clamp(-1, 1)

    def _current_frame(self, data: Dict[str, Any]) -> torch.Tensor:
        if 'image' in data:
            frame = data['image']
        else:
            frames = data.get(self.image_key)
            if frames is None:
                raise ValueError(
                    f'Online history requires `image` or `{self.image_key}`')
            layout = self.frame_layout
            if layout == 'auto':
                if frames.ndim == 3:
                    layout = 'c_h_w'
                elif frames.ndim == 4:
                    raise ValueError(
                        'Ambiguous 4-D frame layout; pass frame_layout='
                        "'t_c_h_w' or 'v_c_h_w' explicitly")
                elif frames.ndim == 5:
                    layout = 'v_t_c_h_w'
                else:
                    raise ValueError(
                        f'Unsupported head-frame shape {tuple(frames.shape)}')
            if layout == 'c_h_w':
                frame = frames
            elif layout == 't_c_h_w':
                frame = frames[0]
            elif layout == 'v_c_h_w':
                frame = frames[self.head_index]
            else:  # v_t_c_h_w
                frame = frames[self.head_index, 0]
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError('Head frame must be CHW RGB')
        frame = self._to_signed(frame)
        if self.world_size is not None:
            frame = F.interpolate(frame.unsqueeze(0), size=self.world_size,
                                  mode='bilinear', align_corners=False,
                                  antialias=True)[0]
        return frame

    def _vl_history(self, history: Dict[str, Any]) -> None:
        if self.vl_image_size is None:
            return
        video = history['history_video']
        valid = history['history_valid']
        if video.shape[0] == 0:
            history['vl_history'] = video.new_empty(
                (0, 3, self.vl_image_size, self.vl_image_size))
            history['vl_history_valid'] = valid
            return
        resized = F.interpolate(video.float(),
                                size=(self.vl_image_size,) * 2,
                                mode='bilinear', align_corners=False,
                                antialias=True)
        history['vl_history'] = resized.contiguous()
        history['vl_history_valid'] = valid

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if data.get('patchwam_reset', False):
            self.buffer.reset()
        episode = data.get('patchwam_episode_index')
        timestamp = data.get('patchwam_timestamp')
        if episode is None or timestamp is None:
            raise ValueError(
                'Online history requires `patchwam_episode_index` and '
                '`patchwam_timestamp` at every observation step')
        frame = self._current_frame(data)
        self.buffer.append(frame, float(timestamp), episode_id=int(episode))
        history = self.buffer.history(float(timestamp))
        self._vl_history(history)
        data.update(history)
        return data
