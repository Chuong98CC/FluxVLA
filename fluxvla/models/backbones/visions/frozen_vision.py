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

"""Shared frozen visual-feature contract for swappable encoders.

``VisionBackbone`` is the token-embedding interface used by the VLMs.
Feature-space consumers (world-action models, auxiliary feature targets,
latent reconstruction) additionally need a spatial view of the same
encoder: ``[B, N, D]`` tokens over an ``(h, w)`` raster, produced by a
frozen encoder from finite ``[B, 3, H, W]`` RGB. ``FrozenVisionFeatures``
adds that contract so a VAE, C-RADIO, or future encoder can be swapped
without changing the consumer.

The input range is part of each encoder's contract, not of the shared
interface. C-RADIO consumes ``[0, 1]`` and uses :func:`to_unit_range` to
convert the world-model endpoint range; the VAE latent wrapper consumes the
native ``[-1, 1]`` endpoint range unchanged.

Encoders in this package are frozen by construction: they are placed in
eval mode, ``requires_grad`` is cleared, and encoding runs under
``torch.no_grad()``. Consumers own any trainable projection.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from .base_vision import VisionBackbone


@dataclass(frozen=True)
class SpatialFeatureOutput:
    """Frozen spatial features and the raster they tile.

    Attributes:
        tokens: ``[B, N, D]`` finite feature tokens with ``N == h * w``.
            Batch entries are independent images, never mixed views.
        grid: ``(h, w)`` feature raster of a single input image, used by
            consumers to build per-view positional geometry.
        raw: The encoder's own return value, for callers that need the
            summary stream as well. Never required by the token contract.
    """

    tokens: torch.Tensor
    grid: Tuple[int, int]
    raw: object = None

    def __post_init__(self) -> None:
        height, width = self.grid
        if height < 1 or width < 1:
            raise ValueError('The feature grid must be positive')
        if self.tokens.ndim != 3:
            raise ValueError('Spatial features must be [B, N, D] tokens')
        if self.tokens.shape[1] != height * width:
            raise ValueError(
                'Spatial feature count does not match the declared grid: '
                f'{self.tokens.shape[1]} tokens for {height}x{width}')

    @property
    def feature_dim(self) -> int:
        return int(self.tokens.shape[-1])


def to_unit_range(images: torch.Tensor,
                  name: str = 'visual encoder') -> torch.Tensor:
    """Validate ``[B, 3, H, W]`` RGB in ``[-1, 1]`` and rescale to ``[0, 1]``.

    World-action datasets carry endpoint frames in ``[-1, 1]`` and the
    ``[0, 1]`` encoders in this package consume that converted range. The
    conversion is applied exactly once, so callers must pass the native
    range; encoders whose contract keeps ``[-1, 1]`` do not call this helper.
    """
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError(f'{name} images must be [B,3,H,W] tensors')
    if images.shape[1] != 3:
        raise ValueError(f'{name} images must have three RGB channels')
    if images.numel() and not torch.isfinite(images).all():
        raise ValueError(f'{name} images must be finite; sanitize invalid '
                         'frames before encoding')
    if images.numel() and (float(images.min()) < -1.001
                           or float(images.max()) > 1.001):
        raise ValueError(f'{name} images must be in [-1,1]')
    return ((images.float() + 1) / 2).clamp_(0, 1)


class FrozenVisionFeatures(VisionBackbone):
    """Frozen visual encoder with an explicit spatial-feature contract.

    Subclasses provide ``feature_grid`` and ``encode_features``; the
    remaining ``VisionBackbone`` members are satisfied here so every
    implementation is usable as a token-embedding backbone too.

    Args:
        vision_backbone_id: Identifier of the concrete encoder variant.
        feature_dim: Width ``D`` of the encoded spatial features.
        resolution_step: Input resolution multiple the encoder honours,
            in pixels. Used to pad inputs so geometry is exact.
        default_image_resolution: ``(C, H, W)`` resolution used for
            ``num_patches`` and ``default_image_resolution``.
    """

    def __init__(self,
                 vision_backbone_id: str,
                 *,
                 feature_dim: int,
                 resolution_step: int,
                 default_image_resolution: Tuple[int, int, int] = (
                     3, 256, 256)) -> None:
        super().__init__(vision_backbone_id)
        if int(feature_dim) < 1:
            raise ValueError('The visual feature dimension must be positive')
        if int(resolution_step) < 1:
            raise ValueError('The visual resolution step must be positive')
        channels, height, width = default_image_resolution
        if channels != 3 or height < 1 or width < 1:
            raise ValueError('The default image resolution must be (3,H,W)')
        self.feature_dim = int(feature_dim)
        self.resolution_step = int(resolution_step)
        self._default_image_resolution = (int(channels), int(height),
                                         int(width))

    # -- shared spatial contract -----------------------------------------
    def train(self, mode: bool = True) -> 'FrozenVisionFeatures':
        """Keep the encoder and its children frozen in evaluation mode."""
        super().train(False)
        return self

    def freeze(self) -> 'FrozenVisionFeatures':
        """Freeze the encoder and keep it in evaluation mode."""
        self.requires_grad_(False)
        self.train(False)
        return self

    @abstractmethod
    def feature_grid(self, height: int, width: int) -> Tuple[int, int]:
        """Feature raster ``(h, w)`` produced for an ``H x W`` input."""
        raise NotImplementedError

    @abstractmethod
    def encode_features(self, images: torch.Tensor) -> SpatialFeatureOutput:
        """Encode ``[B, 3, H, W]`` RGB in ``[-1, 1]`` to spatial tokens."""
        raise NotImplementedError

    # -- VisionBackbone interface ----------------------------------------
    @property
    def embed_dim(self) -> int:
        return self.feature_dim

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self._default_image_resolution

    @property
    def num_patches(self) -> int:
        _, height, width = self._default_image_resolution
        grid_h, grid_w = self.feature_grid(height, width)
        return grid_h * grid_w

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return ``[B, N, D]`` spatial tokens for ``[B, 3, H, W]`` input."""
        return self.encode_features(pixel_values).tokens

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Wrap this encoder as one unit; its weights are frozen."""
        from fluxvla.engines.utils.fsdp_wrapping import (
            build_module_wrap_policy)
        return build_module_wrap_policy({type(self)})
