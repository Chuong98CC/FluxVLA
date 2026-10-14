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

"""Frozen C-RADIO spatial features as a swappable visual backbone.

The official NVIDIA C-RADIO exports consume float RGB in ``[0, 1]``, honour
a patch-aligned input resolution (``get_nearest_supported_resolution``) and
return a summary stream plus a spatial stream. This module presents that
encoder through :class:`FrozenVisionFeatures`, so a consumer only sees
``[B, N, D]`` tokens and an ``(h, w)`` grid.

Local assets only. :meth:`RadioViTBackbone.from_local_export` reuses the
repository's strict C-RADIOv4-H export loader
(``fluxvla.models.third_party_models.patchwam.radio``), which resolves a
complete local Hugging Face directory with ``local_files_only=True`` and
rejects a missing, mismatched, or wrong-version asset instead of falling
back to another implementation. Nothing is downloaded and no alternative
model is substituted.
"""

import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from fluxvla.engines import VISION_BACKBONES
from fluxvla.models.third_party_models.patchwam.radio import (
    DEFAULT_RADIO_FEATURE_DIM, DEFAULT_RADIO_PATCH_SIZE, RADIO_V4_VERSION,
    load_local_radio_teacher, radio_spatial_features)

from .frozen_vision import FrozenVisionFeatures, SpatialFeatureOutput, \
    to_unit_range

#: Feature widths of the official C-RADIOv4 exports. The loader validates
#: the resolved model against the configured width and fails on mismatch.
RADIO_V4_FEATURE_DIMS: Dict[str, int] = {
    'c-radio_v4-h': DEFAULT_RADIO_FEATURE_DIM,
    'c-radio_v4-so400m': 1152,
}


def radio_feature_dim(version: str) -> int:
    """Expected feature width of an official C-RADIOv4 version."""
    if version not in RADIO_V4_FEATURE_DIMS:
        raise ValueError(
            f'Unsupported C-RADIO version {version!r}; expected one of '
            f'{sorted(RADIO_V4_FEATURE_DIMS)}')
    return RADIO_V4_FEATURE_DIMS[version]


def _returns_feature_format(model: Any) -> bool:
    """Whether the RADIO callable accepts ``feature_fmt``."""
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - non-introspectable
        return False
    return 'feature_fmt' in parameters


def normalize_spatial_tokens(features: torch.Tensor,
                             feature_dim: int,
                             layout: str = 'auto') -> torch.Tensor:
    """Normalize an official C-RADIO spatial stream to ``[B, N, D]`` tokens.

    Args:
        features: ``[B, N, D]`` tokens directly, a channels-first
            ``[B, D, h, w]`` raster, or a channels-last ``[B, h, w, D]``
            raster, exactly as the official exports return them.
        feature_dim: Width ``D`` the configured encoder must produce.
        layout: ``'auto'`` accepts any of the layouts above; ``'NCHW'`` and
            ``'NHWC'`` assert one raster layout explicitly, which is what an
            export wrapper knows after choosing it at graph-build time.
    """
    if not torch.is_tensor(features):
        raise ValueError('The C-RADIO spatial stream must be a tensor; got '
                         f'{type(features).__name__}')
    if features.ndim == 3:
        return features
    if features.ndim != 4:
        raise ValueError('The C-RADIO spatial stream must be [B, N, D] tokens '
                         f'or a [B, *, *, *] raster; got {tuple(features.shape)}')
    if layout == 'auto':
        if features.shape[1] == feature_dim:
            layout = 'NCHW'
        elif features.shape[-1] == feature_dim:
            layout = 'NHWC'
        else:
            raise ValueError(
                'The C-RADIO spatial stream is neither NCHW nor NHWC; got '
                f'{tuple(features.shape)} but {feature_dim} was configured')
    if layout == 'NCHW':
        return features.flatten(2).transpose(1, 2)
    if layout == 'NHWC':
        return features.flatten(1, 2)
    raise ValueError(f'Unsupported C-RADIO spatial layout {layout!r}')


@VISION_BACKBONES.register_module()
class RadioViTBackbone(FrozenVisionFeatures):
    """Frozen C-RADIO features over patch-aligned input geometry.

    Args:
        vision_backbone_id: Registered variant, e.g. ``c-radio_v4-h``.
        model: A ready RADIO callable. Mutually exclusive with
            ``model_path``; injected models are used for tests and for
            callers that own the loading policy.
        model_path: Complete local official export directory.
        version: Official C-RADIOv4 version. Defaults to the identifier.
        feature_dim: Expected feature width. Defaults to the version's
            documented width and is validated against the encoder.
        patch_size: Encoder patch size in pixels.
        resolution_step: Patch-aligned resolution multiple. Resolved from
            the loaded export when omitted.
        minimum_resolution: Smallest accepted input edge in pixels.
        feature_format: Requested spatial stream layout. ``NCHW`` matches
            the official repository API; the tokens are returned as
            ``[B, N, D]`` either way.
        dtype: Encoder parameter dtype.
        device: Optional placement device.
        default_image_resolution: ``(C, H, W)`` used for ``num_patches``.
    """

    def __init__(
        self,
        vision_backbone_id: Optional[str] = None,
        *,
        model: Optional[Any] = None,
        model_path: Optional[str] = None,
        version: Optional[str] = None,
        feature_dim: Optional[int] = None,
        patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
        resolution_step: Optional[int] = None,
        minimum_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
        feature_format: str = 'NCHW',
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[Any] = None,
        default_image_resolution: Tuple[int, int, int] = (3, 256, 256),
    ) -> None:
        version = version or vision_backbone_id or RADIO_V4_VERSION
        if version not in RADIO_V4_FEATURE_DIMS:
            raise ValueError(
                f'Unsupported C-RADIO version {version!r}; expected one of '
                f'{sorted(RADIO_V4_FEATURE_DIMS)}')
        expected_dim = (radio_feature_dim(version) if feature_dim is None else
                        int(feature_dim))
        if int(patch_size) < 1 or int(minimum_resolution) < 1:
            raise ValueError('The RADIO patch and minimum resolution must be '
                             'positive')
        if feature_format != 'NCHW':
            raise ValueError('C-RADIO spatial features are requested as NCHW')
        if (model is None) == (model_path is None):
            raise ValueError('Provide exactly one of model or model_path')
        self.version = version
        self.patch_size = int(patch_size)
        self.minimum_resolution = int(minimum_resolution)
        self.feature_format = feature_format
        self.dtype = dtype
        step = resolution_step
        if model is None:
            teacher = load_local_radio_teacher(
                model_path,
                version=version,
                feature_dim=expected_dim,
                patch_size=patch_size,
                minimum_resolution=minimum_resolution,
                device=device,
                dtype=dtype)
            model = teacher.model
            self.model_path = teacher.model_path
            declared = getattr(model, 'embed_dim', None)
            if declared is not None and int(declared) != expected_dim:
                raise ValueError(
                    f'The loaded C-RADIO model produces {int(declared)}-wide '
                    f'features; {expected_dim} was configured')
            step = teacher.min_resolution_step or max(
                int(patch_size), int(minimum_resolution))
        else:
            self.model_path = None
            if device is not None:
                model = model.to(device=device)
            if dtype is not None:
                model = model.to(dtype=dtype)
        if step is None:
            step = max(int(patch_size), int(minimum_resolution))
        super().__init__(
            vision_backbone_id or version,
            feature_dim=expected_dim,
            resolution_step=int(step),
            default_image_resolution=default_image_resolution)
        self.model = model
        self.freeze()

    @classmethod
    def from_local_export(cls, model_path: str,
                          **kwargs) -> 'RadioViTBackbone':
        """Build from a complete local official C-RADIO export directory."""
        kwargs.setdefault('model_path', model_path)
        return cls(**kwargs)

    # -- geometry ---------------------------------------------------------
    def _nearest_supported_resolution(self, height: int,
                                      width: int) -> Tuple[int, int]:
        """Patch-aligned input size the encoder will actually consume.

        The official helper is authoritative, including when it resolves to a
        *smaller* resolution than the requested one. Without it, the size
        rounds down to the patch-aligned step so unsupported geometry is never
        assumed.
        """
        nearest = getattr(self.model, 'get_nearest_supported_resolution', None)
        if callable(nearest):
            try:
                resolved = nearest(int(height), int(width))
            except TypeError:
                resolved = nearest((int(height), int(width)))
            if torch.is_tensor(resolved):
                resolved = resolved.tolist()
            resolved = tuple(int(value) for value in resolved)
            if len(resolved) == 2:
                height, width = resolved
        else:
            step = self.resolution_step
            height = max(step, int(height) // step * step)
            width = max(step, int(width) // step * step)
        if int(height) < 1 or int(width) < 1:
            raise ValueError('The RADIO resolution must be positive')
        if int(height) % self.patch_size or int(width) % self.patch_size:
            raise ValueError(
                'The RADIO resolution must be a patch-size multiple: '
                f'{int(height)}x{int(width)} for patch {self.patch_size}')
        return int(height), int(width)

    def _prepare_input(self, unit: torch.Tensor,
                       target: Tuple[int, int]) -> torch.Tensor:
        """Resample down or pad up to the supported geometry.

        Shrinking uses bilinear resampling because padding cannot remove
        pixels; growing pads with zeros after the ``[0, 1]`` conversion. Axes
        are handled independently, so a mixed request stays exact.
        """
        height, width = unit.shape[-2:]
        target_h, target_w = target
        keep_h, keep_w = min(height, target_h), min(width, target_w)
        if (keep_h, keep_w) != (height, width):
            unit = F.interpolate(unit, size=(keep_h, keep_w), mode='bilinear',
                                 align_corners=False, antialias=True)
        pad_h, pad_w = target_h - keep_h, target_w - keep_w
        if pad_h or pad_w:
            unit = F.pad(unit, (0, pad_w, 0, pad_h))
        return unit

    def feature_grid(self, height: int, width: int) -> Tuple[int, int]:
        if int(height) < 1 or int(width) < 1:
            raise ValueError('The RADIO input resolution must be positive')
        padded_h, padded_w = self._nearest_supported_resolution(height, width)
        return padded_h // self.patch_size, padded_w // self.patch_size

    # -- encoding ---------------------------------------------------------
    def _call_model(self, images: torch.Tensor) -> Any:
        if _returns_feature_format(self.model):
            return self.model(images, feature_fmt=self.feature_format)
        return self.model(images)

    def spatial_tokens(self, images: torch.Tensor,
                       *, layout: str = 'auto') -> torch.Tensor:
        """Normalize the encoder's spatial stream to ``[B, N, D]`` tokens.

        ``images`` must already be in the encoder's ``[0, 1]`` range and at a
        supported geometry: this method calls the encoder directly and applies
        no resampling. It is the graph boundary the offline ONNX export
        wraps, which is why it is public and separate from
        :meth:`encode_features`.
        """
        features = normalize_spatial_tokens(
            radio_spatial_features(self._call_model(images)),
            self.feature_dim, layout)
        if features.ndim != 3 or features.shape[0] != images.shape[0]:
            raise ValueError('C-RADIO must return [B, N, D] spatial features')
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f'The C-RADIO encoder produced {features.shape[-1]}-wide '
                f'features; {self.feature_dim} was configured')
        return features

    @torch.no_grad()
    def encode_features(self, images: torch.Tensor) -> SpatialFeatureOutput:
        """Encode ``[B, 3, H, W]`` RGB in ``[-1, 1]`` to ``[B, N, D]``."""
        unit = to_unit_range(images, 'C-RADIO')
        height, width = unit.shape[-2:]
        target = self._nearest_supported_resolution(height, width)
        unit = self._prepare_input(unit, target)
        padded_h, padded_w = target
        self.model.eval()
        features = self.spatial_tokens(
            unit.to(device=self._parameter_device(),
                    dtype=self._parameter_dtype()))
        grid = (padded_h // self.patch_size, padded_w // self.patch_size)
        if features.shape[1] != grid[0] * grid[1]:
            raise ValueError(
                'The C-RADIO feature count does not match the patch-aligned '
                f'grid: {features.shape[1]} tokens for {grid[0]}x{grid[1]}')
        return SpatialFeatureOutput(tokens=features, grid=grid)

    def _parameter_device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        return torch.device('cpu') if parameter is None else parameter.device

    def _parameter_dtype(self) -> torch.dtype:
        parameter = next(self.model.parameters(), None)
        return self.dtype if parameter is None else parameter.dtype


def build_radio_backbone(version: str = RADIO_V4_VERSION,
                         **kwargs) -> RadioViTBackbone:
    """Convenience factory for the documented C-RADIOv4 versions."""
    return RadioViTBackbone(version, **kwargs)


__all__ = [
    'RADIO_V4_FEATURE_DIMS',
    'RadioViTBackbone',
    'build_radio_backbone',
    'normalize_spatial_tokens',
    'radio_feature_dim',
]
