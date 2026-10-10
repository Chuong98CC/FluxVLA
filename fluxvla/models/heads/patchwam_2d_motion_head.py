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

import math
from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from fluxvla.engines import HEADS
from ..third_party_models.patchwam.motion2d import PatchMotion2DCodec
from ..third_party_models.patchwam.policy import PatchFlowPolicy
from .patchwam_head import _build_denoiser, _validate_denoiser_dims

__all__ = ['PatchWAM2DMotionHead']


def _check_weight(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite non-negative number, got '
                         f'{value!r}')
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f'{name} must be a finite non-negative number, got '
                         f'{value!r}')
    return float(value)


@HEADS.register_module()
class PatchWAM2DMotionHead(PatchFlowPolicy):
    """Joint visual-endpoint flow plus signed 2D patch-motion flow.

    The head keeps the vendored :class:`PatchFlowPolicy` machinery unchanged:
    the frozen observation tokens, the prefix-masked joint denoiser, the shifted
    flow schedule and the weighted masked velocity loss are reused verbatim.
    What changes is the "action" branch: one width-128 token per ``8x8`` patch
    region per future step, carrying 64 signed ``(delta_row, delta_column)``
    pixel displacements in the packed order produced by
    :class:`PatchMotion2DCodec`.

    Motion is supplied in pixels through ``motion_2d`` ``[B,T,h,w,2]`` with an
    optional ``motion_valid`` ``[B,T,h,w]`` boolean mask shared by the two
    coordinates. Normalization happens exactly once inside the codec, so
    invalid (padded or lost-track) coordinates are zeroed before packing and
    excluded from the loss.

    This variant is state-free: ``proprio_dim`` must stay ``None`` and robot
    state/action inputs are rejected instead of silently consumed.
    """

    def __init__(
        self,
        denoiser: Union[nn.Module, Mapping[str, Any]],
        *,
        image_height: int,
        image_width: int,
        text_dim: int,
        motion_horizon: int = 3,
        token_dim: int = 128,
        patch_size: int = 16,
        block_size: int = 8,
        video_weight: float = 0.5,
        motion_weight: float = 1.0,
        shift: float = 5.0,
        isolate_actions: bool = False,
        condition_dropout: float = 0.0,
        flow_normalization: str = 'continuous',
        self_flow_variant: int = 0,
        proprio_dim: Optional[int] = None,
    ) -> None:
        if proprio_dim is not None:
            raise ValueError(
                'PatchWAM2DMotionHead replaces robot state with 2D patch '
                'motion; proprio_dim must be None (this variant has no state '
                'projection).')
        if self_flow_variant:
            raise ValueError(
                'Self-Flow training is deferred from the native 2D '
                'patch-motion integration; set self_flow_variant=0.')
        video_weight = _check_weight('video_weight', video_weight)
        motion_weight = _check_weight('motion_weight', motion_weight)
        if isinstance(motion_horizon, bool) or not isinstance(motion_horizon, int) \
                or motion_horizon < 1:
            raise ValueError(
                f'motion_horizon must be a positive integer, got '
                f'{motion_horizon!r}')
        codec = PatchMotion2DCodec(image_height=image_height,
                                   image_width=image_width,
                                   patch_size=patch_size,
                                   block_size=block_size, token_dim=token_dim)
        denoiser = _build_denoiser(denoiser, token_dim=token_dim,
                                   text_dim=text_dim)
        _validate_denoiser_dims(denoiser, token_dim=token_dim,
                                text_dim=text_dim)
        super().__init__(
            denoiser,
            action_dim=2,
            text_dim=text_dim,
            proprio_dim=None,
            token_dim=token_dim,
            action_scale=1.0,
            shift=shift,
            video_weight=video_weight,
            action_weight=motion_weight,
            isolate_actions=isolate_actions,
            condition_dropout=condition_dropout,
            flow_normalization=flow_normalization,
            self_flow_variant=0,
        )
        # Replace the repeated-action codec with the real patch-motion codec.
        self.codec = codec
        self.motion_horizon = int(motion_horizon)
        self.motion_weight = float(motion_weight)

    @property
    def condition_features(self) -> int:
        """Width of the encoded text context consumed by the denoiser."""
        return self.text_dim

    # ------------------------------------------------------------------
    # Motion batch adaptation
    # ------------------------------------------------------------------
    def motion_inputs(self, batch: Mapping[str, Any]):
        """Validate ``motion_2d`` / ``motion_valid`` and sanitize placeholders.

        Returns ``(pixel_motion [B,T,h,w,2], valid [B,T,h,w])``. Valid
        coordinates must be finite; invalid placeholders may be nonfinite and
        are replaced with zero so they cannot reach the packed tokens.
        """
        if batch.get('proprio') is not None:
            raise ValueError(
                'PatchWAM2DMotionHead is state-free; remove `proprio` from the '
                'batch.')
        robot_action_fields = sorted(
            key for key in ('action', 'action_is_pad', 'action_dim_is_pad')
            if batch.get(key) is not None)
        if robot_action_fields:
            raise ValueError(
                'PatchWAM2DMotionHead consumes patch motion, not robot action '
                f'fields {robot_action_fields}; pass `motion_2d`/`motion_valid` '
                'instead.')
        if 'motion_2d' not in batch:
            raise ValueError(
                'PatchWAM2DMotionHead requires `motion_2d` [B,T,h,w,2] signed '
                'pixel displacements.')
        motion = batch['motion_2d']
        if not torch.is_tensor(motion):
            raise TypeError('motion_2d must be a torch tensor')
        motion = motion.to(torch.float32)
        rows, columns = self.codec.grid_shape
        if (motion.ndim != 5 or tuple(motion.shape[-3:-1]) != (rows, columns)
                or motion.shape[-1] != 2):
            raise ValueError(
                f'motion_2d must be [B,T,h,w,2] with (h,w)={self.codec.grid_shape}; '
                f'got {tuple(motion.shape)}')
        batch_size, horizon = motion.shape[:2]
        if horizon < 1:
            raise ValueError('motion_2d must contain at least one timestep')
        valid = batch.get('motion_valid')
        if valid is None:
            valid = torch.ones(batch_size, horizon, rows, columns,
                               dtype=torch.bool, device=motion.device)
        else:
            if not torch.is_tensor(valid) or valid.dtype != torch.bool:
                raise ValueError('motion_valid must be a boolean tensor')
            valid = valid.to(device=motion.device)
            if tuple(valid.shape) != (batch_size, horizon, rows, columns):
                raise ValueError(
                    f'motion_valid must be [B,T,h,w] matching motion_2d, got '
                    f'{tuple(valid.shape)}')
        finite = torch.isfinite(motion)
        if bool(((~finite) & valid[..., None]).any()):
            raise ValueError(
                'valid motion coordinates must be finite; found a nonfinite '
                'displacement at a valid (motion_valid=True) position')
        return torch.where(finite, motion, torch.zeros_like(motion)), valid

    def _motion_batch(self, batch: Mapping[str, Any]) -> dict:
        """Build the action-branch contract the vendored policy consumes."""
        motion, valid = self.motion_inputs(batch)
        batch_size, horizon = motion.shape[:2]
        rows, columns = self.codec.grid_shape
        coordinate_valid = valid[..., None].expand(batch_size, horizon, rows,
                                                   columns, 2)
        token_valid = self.codec.coordinate_validity(coordinate_valid)
        identifiers = self.codec.token_ids(horizon).to(motion.device)
        result = dict(batch)
        result['action'] = motion.reshape(batch_size,
                                         horizon * rows * columns, 2)
        result['action_dim_is_pad'] = ~coordinate_valid.reshape(
            batch_size, horizon * rows * columns, 2)
        result['action_is_pad'] = ~coordinate_valid.any(-1).reshape(
            batch_size, horizon * rows * columns)
        result['action_token_valid'] = token_valid
        result['action_ids'] = identifiers.unsqueeze(0).expand(
            batch_size, -1, -1)
        return result

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch: Mapping[str, Any],
                **kwargs: Any) -> dict[str, torch.Tensor]:
        """Joint future-endpoint and 2D patch-motion flow loss."""
        batch = self._motion_batch(batch)
        result = super().forward(batch, **kwargs)
        # `loss_action` is the vendored name for the motion branch; expose the
        # variant-specific name and keep the alias for runner compatibility.
        result['loss_motion'] = result['loss_action']
        return result

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_motion(
        self,
        batch: Mapping[str, Any],
        *,
        motion_horizon: Optional[int] = None,
        grid_shape: Optional[tuple[int, int]] = None,
        steps: int = 20,
        generator=None,
        initial_motion: Optional[torch.Tensor] = None,
        future_length: Optional[int] = None,
        guidance_scale: float = 1.0,
        motion_guidance_scale: Optional[float] = None,
    ) -> dict[str, torch.Tensor]:
        """Sample ``T*K`` motion tokens plus the joint future raster.

        Returns decoded signed pixel motion, the source-anchored cumulative
        traces and the sampled future latent tokens.
        """
        if batch.get('proprio') is not None:
            raise ValueError(
                'PatchWAM2DMotionHead is state-free; remove `proprio` from the '
                'batch.')
        if 'reference_tokens' not in batch:
            raise ValueError(
                'sample_motion requires encoded `reference_tokens`.')
        horizon = self.motion_horizon if motion_horizon is None else int(motion_horizon)
        if horizon < 1:
            raise ValueError(f'motion_horizon must be positive, got {horizon}.')
        if grid_shape is not None:
            grid = tuple(int(value) for value in grid_shape)
            if grid != self.codec.grid_shape:
                raise ValueError(
                    f'motion_grid_shape {grid} does not match the codec patch '
                    f'grid {self.codec.grid_shape}.')
        steps = int(steps)
        if steps < 1:
            raise ValueError(
                f'num_inference_steps must be positive, got {steps}.')
        reference = batch['reference_tokens']
        batch_size = reference.shape[0]
        identifiers = self.codec.token_ids(horizon).to(reference.device)
        batch = dict(batch)
        batch['action_ids'] = identifiers.unsqueeze(0).expand(
            batch_size, -1, -1)
        encoded_initial = None
        if initial_motion is not None:
            encoded_initial = self.codec.encode(initial_motion)
            expected = (batch_size, self.codec.token_count(horizon),
                        self.codec.token_dim)
            if tuple(encoded_initial.shape) != expected:
                raise ValueError(
                    f'initial_motion must encode to {expected}, got '
                    f'{tuple(encoded_initial.shape)}')
            encoded_initial = encoded_initial.to(reference)
        sampled = super().sample_actions(
            batch,
            horizon=self.codec.token_count(horizon),
            steps=steps,
            generator=generator,
            initial_action=encoded_initial,
            future_length=future_length,
            guidance_scale=guidance_scale,
            action_guidance_scale=motion_guidance_scale,
        )
        motion = sampled['action']
        return {
            'motion_2d': motion,
            'traces_2d': self.codec.traces_from_motion(motion),
            'future_tokens': sampled['future_tokens'],
        }
