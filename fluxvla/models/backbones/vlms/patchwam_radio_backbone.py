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

from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from fluxvla.engines import VLM_BACKBONES, initialize_overwatch
from ...third_party_models.patchwam.radio import (DEFAULT_RADIO_FEATURE_DIM,
                                                  DEFAULT_RADIO_PATCH_SIZE,
                                                  RADIO_V4_VERSION,
                                                  load_local_radio_teacher,
                                                  radio_target_encoder)
from .patchwam_backbone import PatchWAMBackbone

overwatch = initialize_overwatch(__name__)

__all__ = ['PatchWAMRadioAuxBackbone']


class _RadioTeacherMixin:
    """Shared, out-of-tree lifecycle for the frozen C-RADIOv4-H teacher."""

    def _configure_radio_teacher(
        self,
        *,
        radio_teacher: Optional[Any],
        radio_model_path: Optional[str],
        radio_dim: int,
        radio_version: str,
        radio_patch_size: int,
        radio_min_resolution: int,
        radio_dtype: Optional[Union[str, torch.dtype]],
    ) -> None:
        if radio_dim < 1:
            raise ValueError('radio_dim must be positive')
        if radio_patch_size < 1:
            raise ValueError('radio_patch_size must be positive')
        self.radio_dim = int(radio_dim)
        self.radio_version = str(radio_version)
        self.radio_patch_size = int(radio_patch_size)
        self.radio_min_resolution = int(radio_min_resolution)
        self.radio_model_path = (None if radio_model_path is None else
                                 str(radio_model_path))
        self.radio_dtype = _resolve_dtype(radio_dtype)
        if radio_teacher is not None and self.radio_model_path is not None:
            raise ValueError(
                'Provide either `radio_teacher` or `radio_model_path`, not both')
        # Keep the teacher out of the module tree: frozen weights must never
        # enter state_dict, the optimizer or FSDP wrapping.
        object.__setattr__(self, '_radio_teacher', radio_teacher)

    @property
    def radio_teacher(self) -> Optional[Any]:
        """The attached teacher, if it has been built or injected."""
        return getattr(self, '_radio_teacher', None)

    @property
    def radio_configured(self) -> bool:
        return (self.radio_teacher is not None
                or self.radio_model_path is not None)

    def _ensure_teacher(self, device: torch.device) -> Any:
        """Return a frozen, eval, no-grad teacher on the encoding device."""
        teacher = self.radio_teacher
        if teacher is None:
            if self.radio_model_path is None:
                raise ValueError(
                    'The RADIO variants need training targets: configure '
                    '`radio_model_path` with a local C-RADIOv4-H export or '
                    'inject `radio_teacher`.')
            teacher = load_local_radio_teacher(
                self.radio_model_path,
                version=self.radio_version,
                feature_dim=self.radio_dim,
                patch_size=self.radio_patch_size,
                minimum_resolution=self.radio_min_resolution,
                device=device,
                dtype=self.radio_dtype,
            )
            object.__setattr__(self, '_radio_teacher', teacher)
        declared = getattr(teacher, 'feature_dim', None)
        if declared is not None and int(declared) != self.radio_dim:
            raise ValueError(
                f'The RADIO teacher provides {int(declared)} features but '
                f'radio_dim={self.radio_dim}')
        if isinstance(teacher, nn.Module):
            teacher.requires_grad_(False)
            teacher.eval()
            parameter = next(teacher.parameters(), None)
            if parameter is not None:
                needs_device = parameter.device != device
                needs_dtype = (self.radio_dtype is not None
                               and parameter.dtype != self.radio_dtype)
                if needs_device or needs_dtype:
                    # Move the lazy teacher to the encoding device and the
                    # configured dtype at first training use.
                    teacher = teacher.to(
                        device=device,
                        dtype=self.radio_dtype if needs_dtype else parameter.dtype)
                    object.__setattr__(self, '_radio_teacher', teacher)
        return teacher

    def _radio_feature_encoder(self):
        """Future-endpoint hook producing C-RADIO spatial features."""
        return radio_target_encoder(
            lambda device: self._ensure_teacher(device),
            patch_size=self.radio_patch_size,
            minimum_resolution=self.radio_min_resolution,
            expected_dim=self.radio_dim,
        )

    def _freeze_radio_teacher(self) -> None:
        teacher = self.radio_teacher
        if isinstance(teacher, nn.Module):
            teacher.requires_grad_(False)
            teacher.eval()

    def apply_trainable_policy(self):
        """Freeze the AE, encoder base and the RADIO teacher."""
        super().apply_trainable_policy()
        self._freeze_radio_teacher()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # The teacher is always frozen and in eval mode.
        self._freeze_radio_teacher()
        return self

    def _radio_checkpoint_contract(self) -> Dict[str, Any]:
        return {
            'radio_dim': self.radio_dim,
            'radio_version': self.radio_version,
            'radio_patch_size': self.radio_patch_size,
            'radio_min_resolution': self.radio_min_resolution,
        }


def _validate_radio_targets(encoded: Mapping[str, Any], key: str,
                            radio_dim: int, *, zero_invalid: bool) -> None:
    """Validate detached RADIO targets and zero invalid padding in place."""
    tokens = encoded[key]
    if tokens.ndim != 3 or tokens.shape[-1] != radio_dim:
        raise ValueError(
            f'{key} must be [B,S,{radio_dim}], got {tuple(tokens.shape)}')
    valid = encoded.get('future_valid')
    if valid is None:
        if not torch.isfinite(tokens).all():
            raise ValueError(f'{key} must be finite')
        return
    if valid.shape != tokens.shape[:2]:
        raise ValueError('future_valid must match the future token grid')
    valid = valid.to(device=tokens.device, dtype=torch.bool)
    if not torch.isfinite(tokens[valid]).all():
        raise ValueError(
            f'{key} must be finite on valid coordinates; invalid frames/views '
            'must be sanitized before the teacher runs')
    if zero_invalid:
        encoded[key] = torch.where(valid[..., None], tokens,
                                   torch.zeros_like(tokens))


@VLM_BACKBONES.register_module()
class PatchWAMRadioAuxBackbone(_RadioTeacherMixin, PatchWAMBackbone):
    """Original VAE future latents plus detached RADIO targets.

    Everything about the original observation encoding is preserved, including
    the frozen future VAE encode. When RADIO targets are requested, the same
    future RGB endpoint additionally produces detached ``radio_future_tokens``
    aligned to the reference raster for the training-only representation loss.
    """

    def __init__(
        self,
        autoencoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        *,
        token_dim: int = 128,
        text_dim: int = 32,
        history_pool_size: int = 4,
        radio_teacher: Optional[Any] = None,
        radio_model_path: Optional[str] = None,
        radio_dim: int = DEFAULT_RADIO_FEATURE_DIM,
        radio_version: str = RADIO_V4_VERSION,
        radio_patch_size: int = DEFAULT_RADIO_PATCH_SIZE,
        radio_min_resolution: int = DEFAULT_RADIO_PATCH_SIZE,
        radio_dtype: Optional[Union[str, torch.dtype]] = None,
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__(
            autoencoder=autoencoder,
            text_encoder=text_encoder,
            token_dim=token_dim,
            text_dim=text_dim,
            history_pool_size=history_pool_size,
            **backbone_kwargs,
        )
        self._configure_radio_teacher(
            radio_teacher=radio_teacher,
            radio_model_path=radio_model_path,
            radio_dim=radio_dim,
            radio_version=radio_version,
            radio_patch_size=radio_patch_size,
            radio_min_resolution=radio_min_resolution,
            radio_dtype=radio_dtype,
        )
        self.apply_trainable_policy()

    def encode(self, batch: Mapping[str, Any],
               target: bool) -> Dict[str, Any]:
        """Return the original VAE batch plus optional RADIO targets."""
        request = dict(batch)
        need_radio = request.pop('need_radio', None)
        if need_radio is None:
            # Default: produce targets whenever a teacher is configured. The
            # VLA passes an explicit flag so ``radio_weight == 0`` never loads
            # the teacher.
            need_radio = self.radio_configured
        encoded = super().encode(
            request, target=target,
            radio_encoder=self._radio_feature_encoder()
            if (target and need_radio) else None)
        if target and 'radio_future_tokens' in encoded:
            _validate_radio_targets(encoded, 'radio_future_tokens',
                                    self.radio_dim, zero_invalid=True)
        return encoded

    def checkpoint_contract(self) -> Dict[str, Any]:
        contract = super().checkpoint_contract()
        contract.update(self._radio_checkpoint_contract())
        return contract


def _resolve_dtype(dtype: Optional[Union[str, torch.dtype]]
                   ) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in {'float32', 'float16', 'bfloat16'}:
        raise ValueError('Unsupported RADIO dtype')
    return getattr(torch, dtype)
