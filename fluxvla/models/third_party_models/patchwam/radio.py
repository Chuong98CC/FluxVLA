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
"""C-RADIOv4-H targets, the training-only teacher and the auxiliary objective.

:class:`RadioAuxFlowPolicy` keeps the original 128-wide VAE future flow and adds
a weighted *clean* C-RADIO feature MSE from the selected transformer's hidden
future states (the last block by default) through one ``radio_projection``.

The teacher is a training dependency only: it is held outside the module tree,
is always frozen/eval/no-grad, never enters ``state_dict`` and is never built
for inference.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .flow import masked_sample_mse
from .policy import PatchFlowPolicy

__all__ = [
    'DEFAULT_RADIO_FEATURE_DIM',
    'DEFAULT_RADIO_PATCH_SIZE',
    'LocalRadioTeacher',
    'RADIO_V4_VERSION',
    'RadioAuxFlowPolicy',
    'align_supervised_features',
    'load_local_radio_teacher',
    'radio_feature_targets',
    'radio_spatial_features',
    'radio_target_encoder',
    'resolve_capture_layer',
]

RADIO_V4_VERSION = 'c-radio_v4-h'
DEFAULT_RADIO_FEATURE_DIM = 1280
DEFAULT_RADIO_PATCH_SIZE = 16
ADAPTER_LAYER_IDEX_NAME = 'adapter_layer_idex'

_SPATIAL_KEYS = ('spatial_features', 'last_hidden_state', 'features', 'spatial')


def _normalize_version(value: Any) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def radio_spatial_features(result: Any) -> Tensor:
    """Extract the ``[N,S,D]`` spatial stream from an official RADIO return.

    The upstream API yields ``(summary, spatial_features)``, a ``RadioOutput``
    dataclass, a Hugging Face ``ModelOutput`` or a plain mapping. Summary and
    adaptor outputs are never used as supervision.
    """
    if torch.is_tensor(result):
        return result
    for name in _SPATIAL_KEYS:
        value = getattr(result, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(result, Mapping):
        for name in _SPATIAL_KEYS:
            value = result.get(name)
            if torch.is_tensor(value):
                return value
    if isinstance(result, (tuple, list)):
        if len(result) == 1 and torch.is_tensor(result[0]):
            return result[0]
        if len(result) == 2 and torch.is_tensor(result[1]):
            return result[1]
    raise TypeError(
        'The C-RADIO teacher must return spatial features [N,S,D], a '
        '(summary, spatial_features) pair, a RadioOutput or a mapping with a '
        f'spatial key; got {type(result)!r}')


def _teacher_call(teacher: Any, images: Tensor) -> Tensor:
    """Run an injected teacher or an ``nn.Module`` forward."""
    encode = getattr(teacher, 'encode', None)
    result = encode(images) if callable(encode) else teacher(images)
    return radio_spatial_features(result)


class LocalRadioTeacher(nn.Module):
    """Frozen ``c-radio_v4-h`` spatial features from a local HF export.

    ``images`` are ``[N,3,H,W]`` in ``[0,1]``: the export's own input
    conditioner performs the normalization, so no CLIP/image processor is
    applied here. Calling code holds this module outside the trained module
    tree.
    """

    def __init__(self, model: nn.Module, *, feature_dim: int, patch_size: int,
                 minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
                 min_resolution_step: Optional[int] = None,
                 version: str = RADIO_V4_VERSION,
                 model_path: Optional[str] = None) -> None:
        super().__init__()
        if feature_dim < 1 or patch_size < 1:
            raise ValueError('RADIO feature_dim and patch_size must be positive')
        self.model = model
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.minimum_resolution = int(minimum_resolution)
        self.min_resolution_step = (None if not min_resolution_step else
                                    int(min_resolution_step))
        self.version = str(version)
        self.model_path = model_path
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        self.model.eval()
        return _teacher_call(self.model, images)


def load_local_radio_teacher(
    model_path: str,
    *,
    version: str = RADIO_V4_VERSION,
    feature_dim: int = DEFAULT_RADIO_FEATURE_DIM,
    patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
    minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
    device: Optional[Any] = None,
    dtype: Optional[torch.dtype] = None,
) -> LocalRadioTeacher:
    """Strictly load a local Hugging Face C-RADIOv4-H export.

    The official export is resolved with ``AutoConfig``/``AutoModel`` using
    ``local_files_only=True`` and ``trust_remote_code=True`` so locally supplied
    model code can run. Nothing is downloaded, no package is installed and there
    is no fallback to an older RADIO implementation: a missing or mismatched
    asset fails with the offending path.
    """
    path = Path(model_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            'The C-RADIOv4-H teacher must be a local Hugging Face export '
            f'directory: {path}')
    config_path = path / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(
            f'The C-RADIOv4-H export is missing config.json: {config_path}')
    config = json.loads(config_path.read_text())
    asset_version = config.get('version')
    if _normalize_version(asset_version) != _normalize_version(version):
        raise ValueError(
            f'The local RADIO export reports version {asset_version!r}; this '
            f'variant requires {version!r}')
    asset_patch = config.get('patch_size')
    if asset_patch is not None and int(asset_patch) != int(patch_size):
        raise ValueError(
            f'The local RADIO export uses patch_size={asset_patch}; the model '
            f'is configured for {patch_size}')
    asset_width = config.get('hidden_size') or config.get('embed_dim')
    if asset_width is not None and int(asset_width) != int(feature_dim):
        raise ValueError(
            f'The local RADIO export produces {int(asset_width)}-wide features;'
            f' the model is configured for radio_dim={feature_dim}')
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as error:  # pragma: no cover - transformers is required
        raise RuntimeError(
            'transformers is required to load the local C-RADIOv4-H export'
        ) from error
    try:
        hf_config = AutoConfig.from_pretrained(str(path), local_files_only=True,
                                               trust_remote_code=True)
        model = AutoModel.from_pretrained(str(path), config=hf_config,
                                          local_files_only=True,
                                          trust_remote_code=True)
    except Exception as error:  # noqa: BLE001 - one clear failure
        raise RuntimeError(
            'Unable to build the local C-RADIOv4-H export at '
            f'{path} with AutoModel/AutoConfig (local_files_only=True, '
            'trust_remote_code=True). Confirm the directory is a complete '
            f'official export. Original error: {error!r}') from error
    declared_version = getattr(hf_config, 'version', None)
    if declared_version is not None and (
            _normalize_version(declared_version)
            != _normalize_version(version)):
        raise ValueError(
            f'The loaded RADIO config reports version {declared_version!r}; '
            f'this variant requires {version!r}')
    declared_width = (getattr(model, 'embed_dim', None)
                      or getattr(hf_config, 'hidden_size', None))
    if declared_width is not None and int(declared_width) != int(feature_dim):
        raise ValueError(
            f'The loaded RADIO model produces {int(declared_width)}-wide '
            f'features; the model is configured for radio_dim={feature_dim}')
    step = (getattr(model, 'min_resolution_step', None)
            or getattr(hf_config, 'min_resolution_step', None)
            or getattr(hf_config, 'window_size', None))
    if step is not None:
        # A window grid multiplies the patch size; keep it in pixel units.
        step = int(step)
        if step < patch_size:
            step *= patch_size
    if device is not None:
        model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return LocalRadioTeacher(model, feature_dim=feature_dim,
                             patch_size=patch_size,
                             minimum_resolution=minimum_resolution,
                             min_resolution_step=step,
                             version=version, model_path=str(path))


def _teacher_geometry(teacher: Any, *, patch_size: int,
                      minimum_resolution: int) -> tuple[int, int]:
    """Patch size and padded-input step honoured for this teacher."""
    patch = int(getattr(teacher, 'patch_size', None) or patch_size)
    if patch < 1:
        raise ValueError('The RADIO patch size must be positive')
    step = max(patch, int(minimum_resolution or 0))
    teacher_step = getattr(teacher, 'min_resolution_step', None)
    if teacher_step:
        step = max(step, int(teacher_step))
    return patch, step


def radio_feature_targets(
    video: Tensor,
    *,
    teacher: Any,
    raster: tuple[int, int],
    patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
    minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
    expected_dim: int = DEFAULT_RADIO_FEATURE_DIM,
) -> Tensor:
    """Align per-view RADIO spatial features to the existing future raster.

    ``video`` is ``[N,3,H,W]`` in the native [-1,1] world endpoint range; each
    element is one camera view, so views are never mixed. Valid nonfinite values
    raise; invalid frames/views must already be sanitized to zero by the caller.
    The endpoint is converted to [0,1], padded to the teacher's step grid,
    encoded, and the teacher grid is spatially interpolated to ``raster`` only
    when the two differ. Returns ``[N, raster_h*raster_w, D]``.
    """
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError('RADIO future images must be [N,3,H,W]')
    raster_h, raster_w = int(raster[0]), int(raster[1])
    if raster_h < 1 or raster_w < 1:
        raise ValueError('The future raster must be positive')
    height, width = video.shape[-2:]
    if not torch.isfinite(video).all():
        raise ValueError(
            'RADIO future images must be finite on valid coordinates; '
            'sanitize invalid frames/views before the teacher runs')
    # The world model feeds the explicit [-1,1] endpoint; RADIO consumes [0,1]
    # and applies its own input conditioner exactly once.
    images = ((video.float() + 1) / 2).clamp_(0, 1)
    patch, step = _teacher_geometry(teacher, patch_size=patch_size,
                                    minimum_resolution=minimum_resolution)
    padded_h = math.ceil(height / step) * step
    padded_w = math.ceil(width / step) * step
    if (padded_h, padded_w) != (height, width):
        images = F.pad(images, (0, padded_w - width, 0, padded_h - height))
    parameter = (next(teacher.parameters(), None)
                 if isinstance(teacher, nn.Module) else None)
    if parameter is not None:
        # Match the teacher's device/dtype explicitly so mixed precision never
        # silently promotes or mismatches the export's parameters.
        images = images.to(device=parameter.device, dtype=parameter.dtype)
    # The teacher is frozen: targets never carry gradient into its weights or
    # into the encoder, even when a caller injects a plain ``nn.Module``.
    with torch.no_grad():
        features = _teacher_call(teacher, images)
    if features.ndim == 4 and features.shape[1] == int(expected_dim):
        features = features.flatten(2).transpose(1, 2)
    if features.ndim != 3 or features.shape[0] != images.shape[0]:
        raise ValueError('The RADIO teacher must return [N,S,D] spatial features')
    if features.shape[-1] != int(expected_dim):
        raise ValueError(
            f'The RADIO teacher produced {features.shape[-1]} features; the '
            f'model is configured for radio_dim={expected_dim}')
    grid_h, grid_w = padded_h // patch, padded_w // patch
    if features.shape[1] != grid_h * grid_w:
        raise ValueError(
            f'The RADIO teacher returned {features.shape[1]} tokens; the '
            f'{grid_h}x{grid_w} grid derived from patch_size={patch} and step='
            f'{step} on a {padded_h}x{padded_w} input expects '
            f'{grid_h * grid_w}')
    # Crop the tokens that only cover padding back to the real image extent.
    needed_h = math.ceil(height / patch)
    needed_w = math.ceil(width / patch)
    if (needed_h, needed_w) != (grid_h, grid_w):
        features = features.reshape(-1, grid_h, grid_w, features.shape[-1])
        features = features[:, :needed_h, :needed_w].reshape(
            -1, needed_h * needed_w, features.shape[-1])
    if (needed_h, needed_w) != (raster_h, raster_w):
        features = F.interpolate(
            features.transpose(1, 2).reshape(-1, features.shape[-1],
                                             needed_h, needed_w),
            size=(raster_h, raster_w), mode='bilinear', align_corners=False,
        ).flatten(2).transpose(1, 2)
    return features


def radio_target_encoder(
    teacher_provider: Callable[[torch.device], Any],
    *,
    patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
    minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
    expected_dim: int = DEFAULT_RADIO_FEATURE_DIM,
) -> Callable[[Tensor, tuple[int, int]], Tensor]:
    """Build the teacher hook consumed by ``encode_observations``."""

    def encode_future(video: Tensor, raster: tuple[int, int]) -> Tensor:
        teacher = teacher_provider(video.device)
        return radio_feature_targets(
            video, teacher=teacher, raster=raster, patch_size=patch_size,
            minimum_resolution=minimum_resolution, expected_dim=expected_dim)

    return encode_future


def align_supervised_features(tokens: Tensor, valid: Optional[Tensor], *,
                              name: str, device: Optional[torch.device] = None,
                              dtype: Optional[torch.dtype] = None) -> Tensor:
    """Detach, align and finiteness-check a supervised RADIO target.

    ``valid`` may arrive on another device (it is not stored with the cached
    features); invalid padding is allowed to hold anything, but every supervised
    coordinate must be finite so a NaN target fails clearly instead of turning
    the loss into NaN.
    """
    if not torch.is_tensor(tokens):
        raise TypeError(f'`{name}` must be a tensor')
    target = tokens.detach()
    if device is not None or dtype is not None:
        target = target.to(device=device, dtype=dtype)
    if valid is None:
        supervised = target
    else:
        if not torch.is_tensor(valid):
            raise TypeError(f'`{name}` validity must be a tensor')
        valid = valid.to(device=target.device, dtype=torch.bool)
        if valid.shape != target.shape[:2]:
            raise ValueError(f'`{name}` validity must be [B,S]')
        supervised = target[valid]
    if not torch.isfinite(supervised).all():
        raise ValueError(
            f'`{name}` must be finite on valid coordinates; masked padding may '
            'hold any value, supervised coordinates may not')
    return target


def _denoiser_reference(denoiser: nn.Module) -> Optional[torch.Tensor]:
    """First denoiser parameter used as the device/dtype reference."""
    return next(denoiser.parameters(), None)


def resolve_capture_layer(adapter_layer_idex: int, depth: int, *,
                          name: str = ADAPTER_LAYER_IDEX_NAME) -> int:
    """Translate the public zero-based layer selector into capture order.

    ``adapter_layer_idex`` indexes block outputs from zero (``0..depth-1``) with
    ``-1`` meaning the last block. Transformer order is all double blocks
    followed by all single blocks, which is exactly the order the existing
    capture walks. Returns the 1-based ``representation_layer`` value the
    transformer capture expects.
    """
    if isinstance(adapter_layer_idex, bool) or not isinstance(
            adapter_layer_idex, int):
        raise ValueError(
            f'`{name}` must be an integer block index in [-1, {depth - 1}] '
            f'(-1 selects the last block); got {adapter_layer_idex!r}')
    if adapter_layer_idex < -1 or adapter_layer_idex >= depth:
        raise ValueError(
            f'`{name}` must be -1 (last block) or a zero-based block index in '
            f'[0, {depth - 1}]; got {adapter_layer_idex}')
    return depth if adapter_layer_idex == -1 else adapter_layer_idex + 1


class RadioAuxFlowPolicy(PatchFlowPolicy):
    """Original VAE flow plus clean C-RADIO feature supervision.

    The future flow state stays ``[B,S,token_dim]`` (the original VAE latents)
    and every original parameter keeps its name and meaning. A single extra
    ``radio_projection`` maps the transformer hidden future tokens of the block
    selected by ``adapter_layer_idex`` to C-RADIO width for a weighted, masked
    clean-feature MSE. All blocks still run in the same single pass so the
    original 128-wide future/action velocities are unchanged. The auxiliary
    target is detached and never noised, and ``radio_weight == 0``
    short-circuits to the untouched parent objective.
    """

    def __init__(self, denoiser: nn.Module, *, action_dim: int, text_dim: int,
                 radio_dim: int = DEFAULT_RADIO_FEATURE_DIM,
                 radio_weight: float = 0.5,
                 adapter_layer_idex: int = -1, **policy_options) -> None:
        super().__init__(denoiser, action_dim=action_dim, text_dim=text_dim,
                         **policy_options)
        if not isinstance(radio_dim, int) or radio_dim < 1:
            raise ValueError('radio_dim must be a positive integer')
        if not math.isfinite(float(radio_weight)) or radio_weight < 0:
            raise ValueError('radio_weight must be finite and non-negative')
        width = getattr(denoiser, 'representation_dim', None)
        if width is None:
            raise ValueError(
                'The RADIO auxiliary head requires a feature-capable denoiser '
                'exposing representation_dim')
        depth = getattr(denoiser, 'representation_layers', None)
        if not depth:
            raise ValueError(
                'The RADIO auxiliary head requires a denoiser exposing '
                'representation_layers so `adapter_layer_idex` can be '
                'validated')
        # Public zero-based selector, kept as a plain configuration attribute
        # (never a parameter or buffer) next to the derived 1-based layer.
        self.adapter_layer_idex = adapter_layer_idex
        self._capture_layer = resolve_capture_layer(adapter_layer_idex,
                                                    int(depth))
        self.radio_dim = int(radio_dim)
        self.radio_weight = float(radio_weight)
        self.radio_projection = nn.Linear(int(width), int(radio_dim))
        reference = _denoiser_reference(denoiser)
        if reference is not None:
            # Match the transformer's device/dtype instead of staying on the
            # default device in float32.
            self.radio_projection.to(device=reference.device,
                                     dtype=reference.dtype)

    @property
    def capture_layer(self) -> Optional[int]:
        """1-based transformer layer tapped for the auxiliary feature loss."""
        if self.radio_weight == 0:
            return None
        return self._capture_layer

    def forward(self, batch, **kwargs):
        if kwargs.get('representation_only') or kwargs.get('teacher_sampling'):
            raise ValueError(
                'The RADIO auxiliary loss is only defined for the training '
                'flow forward')
        if self.radio_weight == 0:
            # Exact original objective: no capture, no target, no teacher.
            result = super().forward(batch, **kwargs)
            result['loss_radio'] = batch['action'].new_zeros(())
            result['loss_vae'] = result['loss_video']
            return result
        capture_layer = self.capture_layer
        if capture_layer is None:
            raise ValueError(
                'The RADIO auxiliary loss requires a denoiser exposing '
                'representation_layers')
        result = super().forward(batch, capture_layer=capture_layer, **kwargs)
        hidden = result.pop('velocity_features', None)
        target = batch.get('radio_future_tokens')
        if hidden is None or not torch.is_tensor(target):
            raise ValueError(
                'radio_weight>0 requires `radio_future_tokens` and the captured '
                'transformer hidden future states')
        future_length = int(batch['future_tokens'].shape[1])
        if hidden.ndim != 3 or hidden.shape[1] < future_length:
            raise ValueError(
                'The captured transformer states must cover the future tokens')
        hidden = hidden[:, :future_length]
        target = align_supervised_features(
            target, batch.get('future_valid'), name='radio_future_tokens',
            device=hidden.device)
        prediction = self.radio_projection(
            hidden.to(self.radio_projection.weight.dtype))
        if prediction.shape != target.shape:
            raise ValueError(
                f'The RADIO feature prediction {tuple(prediction.shape)} does '
                f'not match the target {tuple(target.shape)}')
        valid = batch.get('future_valid')
        if valid is not None:
            valid = valid.to(device=target.device, dtype=torch.bool)
            # Broadcast the per-token mask over the feature coordinates.
            valid = valid[..., None]
        elif not torch.isfinite(target).all():
            raise ValueError('RADIO feature targets must be finite')
        loss_radio = masked_sample_mse(prediction, target, valid).mean()
        result['loss_radio'] = loss_radio
        result['loss_vae'] = result['loss_video']
        result['loss'] = (self.video_weight * result['loss_video']
                          + self.radio_weight * loss_radio
                          + self.action_weight * result['loss_action'])
        return result
