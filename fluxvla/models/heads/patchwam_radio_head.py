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

from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from fluxvla.engines import HEADS
from ..third_party_models.patchwam.radio import (DEFAULT_RADIO_FEATURE_DIM,
                                                 RadioAuxFlowPolicy)
from .patchwam_head import _build_denoiser, _validate_denoiser_dims

__all__ = ['PatchWAMRadioAuxHead']


def _check_weight(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or not torch.isfinite(
            torch.tensor(float(value))) or float(value) < 0:
        raise ValueError(f'{name} must be finite and non-negative, got {value!r}')
    return float(value)


@HEADS.register_module()
class PatchWAMRadioAuxHead(RadioAuxFlowPolicy):
    """Original VAE flow plus the fused linear RADIO velocity auxiliary.

    ``loss = radio_weight*loss_radio + action_weight*loss_action`` (plus the
    language auxiliary folded in once by the VLA); the future VAE velocity MSE
    is never computed, so ``video_weight`` must be 0 and ``loss_video``/
    ``loss_vae`` are detached compatibility zeros. ``loss_radio`` is a weighted
    masked MSE between the training-only readout ``R = A(Y'') + B(H_selected)``
    and the fixed shared-noise target ``P @ epsilon_z - r0``, where
    ``adapter_layer_idex`` (zero-based, ``-1`` = last block) picks the block
    whose conditioned hidden feeds ``B``. All blocks still run in the same
    single pass and the shared ``Y''`` velocity head applies to every generated
    future/action token. ``radio_weight == 0`` skips the teacher, readout and
    target requirement, but the selected hidden and the learned velocity
    residual still execute.
    """

    def __init__(
        self,
        denoiser: Union[nn.Module, Mapping[str, Any]],
        *,
        action_dim: int,
        text_dim: int,
        radio_dim: int = DEFAULT_RADIO_FEATURE_DIM,
        radio_weight: float = 1.0,
        radio_noise_seed: int = 0,
        adapter_layer_idex: int = -1,
        proprio_dim: Optional[int] = None,
        token_dim: int = 128,
        action_scale: float = 1.0,
        shift: float = 5.0,
        video_weight: float = 0.0,
        action_weight: float = 1.0,
        isolate_actions: bool = False,
        condition_dropout: float = 0.0,
        flow_normalization: str = 'continuous',
        self_flow_variant: int = 0,
        pack_proprio_after_text: bool = False,
        action_mode: str = 'joint_only',
        joint_action_dim: Optional[int] = None,
        end_effector_action_dim: Optional[int] = None,
    ) -> None:
        if self_flow_variant:
            raise ValueError(
                'Self-Flow training is not wired up for the RADIO variants; '
                'set self_flow_variant=0.')
        video_weight = _check_weight('video_weight', video_weight)
        radio_weight = _check_weight('radio_weight', radio_weight)
        action_weight = _check_weight('action_weight', action_weight)
        if video_weight != 0:
            raise ValueError(
                'video_weight must be 0 for the RADIO auxiliary variant: the '
                'future VAE velocity MSE is not computed; use radio_weight and '
                'action_weight instead')
        if not isinstance(radio_dim, int) or radio_dim < 1:
            raise ValueError('radio_dim must be a positive integer')
        # The original 128-wide VAE future flow is untouched: the denoiser is
        # built exactly like the baseline PatchWAM denoiser.
        denoiser = _build_denoiser(denoiser, token_dim=token_dim,
                                   text_dim=text_dim)
        _validate_denoiser_dims(denoiser, token_dim=token_dim,
                                text_dim=text_dim)
        super().__init__(
            denoiser,
            action_dim=action_dim,
            text_dim=text_dim,
            radio_dim=radio_dim,
            radio_weight=radio_weight,
            radio_noise_seed=radio_noise_seed,
            adapter_layer_idex=adapter_layer_idex,
            proprio_dim=proprio_dim,
            token_dim=token_dim,
            action_scale=action_scale,
            shift=shift,
            video_weight=video_weight,
            action_weight=action_weight,
            isolate_actions=isolate_actions,
            condition_dropout=condition_dropout,
            flow_normalization=flow_normalization,
            pack_proprio_after_text=pack_proprio_after_text,
            action_mode=action_mode,
            joint_action_dim=joint_action_dim,
            end_effector_action_dim=end_effector_action_dim,
        )

    @property
    def condition_features(self) -> int:
        """Width of the encoded text context consumed by the denoiser."""
        return self.text_dim
