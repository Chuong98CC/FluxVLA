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

from typing import Any, Dict, Optional, Tuple

import torch

from fluxvla.engines import VLAS, initialize_overwatch
from ..heads.patchwam_2d_motion_head import (  # noqa: F401  (registration)
    PatchWAM2DMotionHead)
from .patchwam_vla import PatchWAMVLA

overwatch = initialize_overwatch(__name__)

__all__ = ['PatchWAM2DMotionVLA']

#: Robot state projection tensors present in the robot PatchWAM lineages but
#: intentionally absent from this state-free variant.
STATE_PROJECTION_SOURCE_PREFIXES = (
    'state_projection.',
    'policy.state_projection.',
    'vla_head.state_projection.',
    'proprio_encoder.',
)


@VLAS.register_module()
class PatchWAM2DMotionVLA(PatchWAMVLA):
    """Native FluxVLA wrapper for the 2D patch-motion PatchWAM variant.

    Reuses :class:`PatchWAMBackbone` for the frozen observation/instruction
    encoding and the joint visual endpoint flow, and
    :class:`PatchWAM2DMotionHead` for the per-block signed displacement flow.
    Robot state, robot action chunks and action denormalization are not part of
    this model: ``motion_2d`` stays in signed pixel units and the codec owns its
    single internal normalization.

    Args:
        motion_horizon: Default number of consecutive future steps to sample
            (``T``); the example/default is three.
        num_inference_steps: Default Euler solver steps for sampling.
    """

    def __init__(self, *, motion_horizon: int = 3,
                 action_horizon: Optional[int] = None,
                 action_norm_type: str = 'none', **kwargs: Any) -> None:
        if action_horizon is not None:
            raise ValueError(
                'PatchWAM2DMotionVLA predicts 2D patch motion, not a robot '
                'action chunk; configure `motion_horizon` instead.')
        if (isinstance(motion_horizon, bool)
                or not isinstance(motion_horizon, int) or motion_horizon < 1):
            raise ValueError(
                f'motion_horizon must be a positive integer, got '
                f'{motion_horizon!r}')
        super().__init__(action_horizon=None,
                         action_norm_type=action_norm_type, **kwargs)
        self.motion_horizon = int(motion_horizon)

    def forward(self, images: Optional[torch.Tensor] = None,
                motion_2d: Optional[torch.Tensor] = None,
                motion_valid: Optional[torch.Tensor] = None,
                states: Optional[torch.Tensor] = None,
                actions: Optional[torch.Tensor] = None,
                action_masks: Optional[torch.Tensor] = None,
                action_dim_masks: Optional[torch.Tensor] = None,
                frame_masks: Optional[torch.Tensor] = None,
                img_masks: Optional[torch.Tensor] = None,
                task_description: Any = None,
                **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Run the joint future-endpoint and 2D patch-motion flow loss."""
        if states is not None:
            raise ValueError(
                'PatchWAM2DMotionVLA is state-free: robot `states` are not '
                'consumed.')
        if (actions is not None or action_masks is not None
                or action_dim_masks is not None):
            raise ValueError(
                'PatchWAM2DMotionVLA predicts a 2D patch-motion action chunk, '
                'not robot actions; pass `motion_2d`/`motion_valid`.')
        if motion_2d is None:
            motion_2d = kwargs.get('motion_2d')
        if motion_valid is None:
            motion_valid = kwargs.get('motion_valid')
        if motion_2d is None:
            raise ValueError(
                'PatchWAM2DMotionVLA.forward requires `motion_2d` [B,T,h,w,2] '
                'signed pixel displacements.')
        if self.vlm_backbone is None and (frame_masks is not None
                                          or img_masks is not None):
            raise ValueError(
                '`frame_masks`/`img_masks` require a `vlm_backbone`; cached '
                'callers must pass `reference_valid`/`future_valid` directly.')
        if frame_masks is not None:
            kwargs['frame_masks'] = frame_masks
        if img_masks is not None:
            kwargs['img_masks'] = img_masks
        batch = self._prepare_batch(images=images,
                                    task_description=task_description,
                                    target=True, kwargs=kwargs)
        batch['motion_2d'] = motion_2d
        if motion_valid is not None:
            batch['motion_valid'] = motion_valid
        self._require_tokens(batch, target=True)
        return self._run_head(batch, kwargs)

    def _resolve_motion_grid(
            self, images: Optional[torch.Tensor],
            motion_grid_shape: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        codec = self.vla_head.codec
        if motion_grid_shape is None:
            if images is None:
                raise ValueError(
                    '`motion_grid_shape=(h, w)` is required for cached '
                    'inference without raw images.')
            if images.ndim < 2:
                raise ValueError('images must expose height and width axes')
            motion_grid_shape = (images.shape[-2] // codec.patch_size,
                                 images.shape[-1] // codec.patch_size)
        grid = tuple(int(value) for value in motion_grid_shape)
        if grid != codec.grid_shape:
            raise ValueError(
                f'motion_grid_shape {grid} does not match the codec patch grid '
                f'{codec.grid_shape}.')
        return grid

    @torch.no_grad()
    def predict_motion(self, images: Optional[torch.Tensor] = None,
                       task_description: Any = None,
                       motion_horizon: Optional[int] = None,
                       num_inference_steps: Optional[int] = None,
                       seed: Optional[int] = None,
                       motion_grid_shape: Optional[Tuple[int, int]] = None,
                       initial_motion: Optional[torch.Tensor] = None,
                       states: Optional[torch.Tensor] = None,
                       actions: Optional[torch.Tensor] = None,
                       action_masks: Optional[torch.Tensor] = None,
                       action_dim_masks: Optional[torch.Tensor] = None,
                       **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Sample signed pixel motion, source-anchored traces and the raster.

        Returns ``motion_2d [B,T,h,w,2]`` (signed pixels), ``traces_2d
        [B,T+1,h,w,2]`` (absolute pixel positions) and ``future_tokens
        [B,S,128]`` (the jointly sampled future raster). Raw-image callers
        derive the grid from the image size; cached callers must pass
        ``motion_grid_shape=(h, w)``.
        """
        if states is not None:
            raise ValueError(
                'PatchWAM2DMotionVLA is state-free: robot `states` are not '
                'consumed.')
        if (actions is not None or action_masks is not None
                or action_dim_masks is not None):
            raise ValueError(
                'PatchWAM2DMotionVLA predicts a 2D patch-motion action chunk; '
                'robot action inputs are rejected at inference too.')
        horizon = (self.motion_horizon if motion_horizon is None
                   else int(motion_horizon))
        if horizon < 1:
            raise ValueError(f'motion_horizon must be positive, got {horizon}.')
        steps = (self.num_inference_steps if num_inference_steps is None
                 else int(num_inference_steps))
        if steps < 1:
            raise ValueError(
                f'num_inference_steps must be positive, got {steps}.')
        grid = self._resolve_motion_grid(images, motion_grid_shape)
        batch = self._prepare_batch(images=images,
                                    task_description=task_description,
                                    target=False, kwargs=kwargs)
        self._require_tokens(batch, target=False)
        generator = kwargs.get('generator')
        if generator is None and seed is not None:
            generator = torch.Generator(
                device=batch['reference_tokens'].device).manual_seed(
                    int(seed))
        return self.vla_head.sample_motion(
            batch,
            motion_horizon=horizon,
            grid_shape=grid,
            steps=steps,
            generator=generator,
            initial_motion=initial_motion,
            future_length=kwargs.get('future_length'),
            guidance_scale=kwargs.get('guidance_scale', 1),
            motion_guidance_scale=kwargs.get('motion_guidance_scale'),
        )

    def predict_action(self, *args: Any,
                       action_horizon: Optional[int] = None,
                       **kwargs: Any) -> torch.Tensor:
        """Compatibility entry point returning decoded pixel motion.

        This variant has no robot action decoder: the returned tensor is the
        ``[B,T,h,w,2]`` signed pixel motion, never a robot ``[B,H,A]`` chunk.
        """
        if action_horizon is not None:
            raise ValueError(
                'PatchWAM2DMotionVLA.predict_action returns 2D patch motion; '
                'configure `motion_horizon` instead of `action_horizon`.')
        return self.predict_motion(*args, **kwargs)['motion_2d']

    def _resolve_patchwam_state(self, state):
        """Drop recognized robot state projections; keep every other check.

        The 2D variant has no state projection, so a robot PatchWAM
        checkpoint's recognized ``state_projection``/``proprio_encoder``
        tensors are discarded deliberately and reported. Any other missing or
        unexpected tensor still fails a strict load, so a required DiT weight
        can never be omitted silently.
        """
        available, format_name, resolved, missing, unexpected = (
            super()._resolve_patchwam_state(state))
        dropped, kept = [], []
        for name in unexpected:
            target = dropped if str(name).startswith(
                STATE_PROJECTION_SOURCE_PREFIXES) else kept
            target.append(name)
        self._dropped_state_projection = dropped
        return available, format_name, resolved, missing, kept

    def load_patchwam_state_dict(self, state: Dict[str, torch.Tensor],
                                 *, strict: bool = True) -> Dict[str, Any]:
        report = super().load_patchwam_state_dict(state, strict=strict)
        report['dropped_state_projection'] = list(
            getattr(self, '_dropped_state_projection', []))
        return report
