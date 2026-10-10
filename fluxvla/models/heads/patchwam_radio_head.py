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
    """Original VAE flow plus clean C-RADIO feature MSE.

    ``loss = video_weight*loss_video + radio_weight*loss_radio +
    action_weight*loss_action`` with the original language auxiliary folded in
    by the VLA exactly once. ``loss_radio`` is a masked clean-feature MSE from
    the transformer hidden future tokens of the block selected by
    ``adapter_layer_idex`` (zero-based block index, ``-1`` = last block) through
    ``radio_projection``. Every block still runs in the same single pass, so the
    original velocities are unchanged. ``radio_weight == 0`` is the untouched
    original objective: no capture, no target requirement and no teacher work.
    """

    def __init__(
        self,
        denoiser: Union[nn.Module, Mapping[str, Any]],
        *,
        action_dim: int,
        text_dim: int,
        radio_dim: int = DEFAULT_RADIO_FEATURE_DIM,
        radio_weight: float = 0.5,
        adapter_layer_idex: int = -1,
        proprio_dim: Optional[int] = None,
        token_dim: int = 128,
        action_scale: float = 1.0,
        shift: float = 5.0,
        video_weight: float = 0.5,
        action_weight: float = 1.0,
        isolate_actions: bool = False,
        condition_dropout: float = 0.0,
        flow_normalization: str = 'continuous',
        self_flow_variant: int = 0,
        pack_proprio_after_text: bool = False,
    ) -> None:
        if self_flow_variant:
            raise ValueError(
                'Self-Flow training is not wired up for the RADIO variants; '
                'set self_flow_variant=0.')
        video_weight = _check_weight('video_weight', video_weight)
        radio_weight = _check_weight('radio_weight', radio_weight)
        action_weight = _check_weight('action_weight', action_weight)
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
        )

    @property
    def condition_features(self) -> int:
        """Width of the encoded text context consumed by the denoiser."""
        return self.text_dim
