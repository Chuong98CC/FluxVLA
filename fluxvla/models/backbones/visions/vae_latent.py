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

"""Frozen VAE latents as a swappable visual backbone.

The latent recipe stays the default visual path. This wrapper gives a VAE
the same ``[B, N, D]`` tokens and ``(h, w)`` grid contract as C-RADIO, so a
consumer can switch between reconstructable latents and frozen foundation
features without touching its own code.

The autoencoder is injected rather than re-implemented here: callers pass
the module they already build for their latent recipe (for example the
FLUX.2 autoencoder used by PatchWAM), which keeps a single loading policy
per encoder family. Latent inputs stay in the native ``[-1, 1]`` endpoint
range; no range conversion is applied on this path.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from fluxvla.engines import VISION_BACKBONES

from .frozen_vision import FrozenVisionFeatures, SpatialFeatureOutput


@VISION_BACKBONES.register_module()
class VAELatentVisionBackbone(FrozenVisionFeatures):
    """Frozen autoencoder latents exposed as spatial feature tokens.

    Args:
        vision_backbone_id: Identifier of the latent recipe, e.g.
            ``vae-flux2``.
        autoencoder: A module with ``encode(images) -> [B, C, h, w]``.
            Required; the wrapper never builds or downloads one.
        feature_dim: Latent channel width ``C``. Validated on every encode.
        resolution_step: Spatial downsampling factor of the autoencoder,
            used only for default geometry bookkeeping.
        dtype: Preferred dtype; ignored when the autoencoder is fp32-only.
        device: Optional placement device.
        default_image_resolution: ``(C, H, W)`` used for ``num_patches``.
    """

    def __init__(
        self,
        vision_backbone_id: str = 'vae-latents',
        *,
        autoencoder: Optional[nn.Module] = None,
        feature_dim: int,
        resolution_step: int = 8,
        dtype: torch.dtype = torch.float32,
        device: Optional[Any] = None,
        default_image_resolution: Tuple[int, int, int] = (3, 256, 256),
    ) -> None:
        if autoencoder is None or not callable(
                getattr(autoencoder, 'encode', None)):
            raise ValueError('A frozen autoencoder with encode(images) is '
                             'required; this wrapper never loads one')
        if int(feature_dim) < 1:
            raise ValueError('The latent channel width must be positive')
        super().__init__(
            vision_backbone_id,
            feature_dim=int(feature_dim),
            resolution_step=int(resolution_step),
            default_image_resolution=default_image_resolution)
        self.dtype = dtype
        self.autoencoder = autoencoder
        if device is not None:
            self.autoencoder = self.autoencoder.to(device=device)
        self.freeze()

    def feature_grid(self, height: int, width: int) -> Tuple[int, int]:
        """Latent grid for an ``H x W`` input, using the downsampling step."""
        if int(height) < 1 or int(width) < 1:
            raise ValueError('The latent input resolution must be positive')
        step = self.resolution_step
        return (max(1, int(height) // step), max(1, int(width) // step))

    @torch.no_grad()
    def encode_features(self, images: torch.Tensor) -> SpatialFeatureOutput:
        """Encode ``[B, 3, H, W]`` RGB in ``[-1, 1]`` to ``[B, N, C]``."""
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError('VAE observations must be [B,3,H,W] tensors')
        if images.shape[1] != 3:
            raise ValueError('VAE observations must have three RGB channels')
        if images.numel() and not torch.isfinite(images).all():
            raise ValueError('VAE observations must be finite; sanitize '
                             'invalid frames before encoding')
        self.autoencoder.eval()
        dtype = self._parameter_dtype()
        latents = self.autoencoder.encode(images.to(device=self._device(),
                                                    dtype=dtype))
        if not torch.is_tensor(latents) or latents.ndim != 4:
            raise ValueError(
                'The autoencoder must return [B, C, h, w] latents')
        if latents.shape[1] != self.feature_dim:
            raise ValueError(
                f'The autoencoder produced {int(latents.shape[1])}-wide '
                f'latents; {self.feature_dim} was configured')
        if latents.shape[0] != images.shape[0]:
            raise ValueError('The autoencoder changed the batch size')
        tokens = latents.flatten(2).transpose(1, 2)
        return SpatialFeatureOutput(tokens=tokens,
                                    grid=(int(latents.shape[-2]),
                                          int(latents.shape[-1])))

    def _device(self) -> torch.device:
        parameter = next(self.autoencoder.parameters(), None)
        return torch.device('cpu') if parameter is None else parameter.device

    def _parameter_dtype(self) -> torch.dtype:
        parameter = next(self.autoencoder.parameters(), None)
        return self.dtype if parameter is None else parameter.dtype


__all__ = ['VAELatentVisionBackbone']
