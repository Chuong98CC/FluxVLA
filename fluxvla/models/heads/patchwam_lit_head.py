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
"""Latent Interface Training head for the native PatchWAM stack.

``PatchWAMLITFlowMatchingHead`` owns only the LIT-specific parameters: the
goal-state encoder (Stage 1), the 100-latent semantic/visual aggregator, the
pose reconstruction head (Stage 2), the layer-wise action expert and its
context projection. The shared FLUX.2 transformer and proprio projection stay
owned once by ``PatchWAMHead`` (``vla_head``); this head receives the streamed
prefix memories from the VLA and never registers a second transformer.

Stage 1 conditions the action expert on the goal-conditioned prefix. Stage 2
is the strict bottleneck: the prefix feeds the aggregator only, and the action
expert consumes just the per-depth projected latents and their validity mask.
"""

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from fluxvla.engines import HEADS
from fluxvla.engines.utils.builder import build_from_cfg
from ..third_party_models.fastwam.modules.action_dit import ActionDiT
from ..third_party_models.fastwam.modules.schedulers.scheduler_continuous import \
    WanContinuousFlowMatchScheduler
from ..third_party_models.patchwam.flow import masked_sample_mse
from ..third_party_models.patchwam.lit import (LIT_DROPOUT, LIT_FFN_RATIO,
                                               LIT_INNER_DIM, LIT_LATENT_DIM,
                                               LIT_NUM_GOAL_TOKENS,
                                               LIT_NUM_HEADS, LIT_NUM_LATENTS,
                                               LIT_NUM_LAYER_GROUPS,
                                               LIT_NUM_POSE_TOKENS,
                                               LIT_POSE_LOSS_WEIGHT,
                                               LIT_STAGES,
                                               LITLayerwiseActionExpert,
                                               GoalPoseDecoder,
                                               GoalPoseEncoder,
                                               SemanticVisualAggregator,
                                               compute_pose_reconstruction_loss,
                                               lit_coupling_layers)

__all__ = ['PatchWAMLITActionDiT', 'PatchWAMLITFlowMatchingHead']


@HEADS.register_module()
class PatchWAMLITActionDiT(ActionDiT):
    """Registered native FastWAM ``ActionDiT`` used as the LIT action expert.

    A registry-friendly subclass with no behavioral change: the LIT action
    expert reuses the native ActionDiT blocks, modulation, RoPE cache and
    gradient-checkpoint helper instead of loading the incompatible Wan/MoT
    video stack.
    """


def _build_action_expert(
    cfg: Union[Mapping[str, Any], nn.Module], *, action_dim: int,
    context_dim: int,
) -> nn.Module:
    if isinstance(cfg, nn.Module):
        return cfg
    if not isinstance(cfg, Mapping):
        raise TypeError(
            'PatchWAMLITFlowMatchingHead.action_expert must be an nn.Module or '
            f'a type-keyed config mapping, got {type(cfg)!r}')
    options = dict(cfg)
    options.setdefault('type', 'PatchWAMLITActionDiT')
    options.setdefault('action_dim', action_dim)
    options.setdefault('text_dim', context_dim)
    built = build_from_cfg(options, HEADS)
    if not isinstance(built, nn.Module):
        raise TypeError(
            f'action_expert config must build an nn.Module, got {type(built)!r}')
    return built


@HEADS.register_module()
class PatchWAMLITFlowMatchingHead(nn.Module):
    """Stage-1/Stage-2 LIT interface with a flow-matching action expert.

    Args:
        action_dim: Normalized action/state width (14 for RoboTwin).
        context_dim: Shared FLUX prefix hidden width (Klein base 4B: 3072).
        action_expert: Native ``ActionDiT`` config or module. ``num_layers``
            must be divisible by ``num_layer_groups`` so depth sharing is exact.
        stage: ``stage1`` (goal-conditioned action prior, image-free) or
            ``stage2`` (aggregator + jointly trained interface).
        pose_loss_weight: Goal reconstruction weight (default 0.3).
        fm_action_weight: Flow-matching action loss weight (default 1.0).
    """

    def __init__(
        self,
        *,
        action_dim: int,
        context_dim: int,
        action_expert: Union[Mapping[str, Any], nn.Module],
        stage: str = 'stage2',
        proprio_dim: Optional[int] = None,
        prefix_input_dim: Optional[int] = None,
        num_goal_tokens: int = LIT_NUM_GOAL_TOKENS,
        num_latents: int = LIT_NUM_LATENTS,
        num_pose_tokens: int = LIT_NUM_POSE_TOKENS,
        num_layer_groups: int = LIT_NUM_LAYER_GROUPS,
        latent_dim: int = LIT_LATENT_DIM,
        inner_dim: int = LIT_INNER_DIM,
        num_heads: int = LIT_NUM_HEADS,
        ffn_ratio: float = LIT_FFN_RATIO,
        dropout: float = LIT_DROPOUT,
        context_token_dropout: float = 0.0,
        context_blackout_prob: float = 0.0,
        pose_loss_weight: float = LIT_POSE_LOSS_WEIGHT,
        fm_action_weight: float = 1.0,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        prefix_timestep: float = 0.0,
        action_horizon: int = 16,
        coupling_layers: Optional[Sequence[int]] = None,
        prefix_depth: Optional[int] = None,
        latent_layout: str = 'free',
        action_sees_ref: bool = False,
    ) -> None:
        super().__init__()
        self.stage = str(stage)
        if self.stage not in LIT_STAGES:
            raise ValueError(f'stage must be one of {LIT_STAGES}, got {stage!r}')
        if int(action_dim) < 1:
            raise ValueError('action_dim must be positive')
        if int(context_dim) < 1:
            raise ValueError('context_dim must be positive')
        if str(latent_layout) != 'free':
            raise ValueError(
                "Only the strict 'free' latent layout is ported; ImageWAM's "
                f"relaxed layouts are not supported (got {latent_layout!r})")
        if bool(action_sees_ref):
            raise ValueError(
                'The strict LIT bottleneck never lets the action expert attend '
                'the raw reference image; `action_sees_ref` must stay False')
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        # Width of the tokens the shared encoder *consumes* (text encoder
        # readout). Goal tokens join that stream, so the goal encoder emits this
        # width; the aggregator and action expert consume ``context_dim``
        # (post-``txt_in``) hidden states.
        self.prefix_input_dim = int(
            context_dim if prefix_input_dim is None else prefix_input_dim)
        if self.prefix_input_dim < 1:
            raise ValueError('prefix_input_dim must be positive')
        self.pose_dim = int(
            proprio_dim if proprio_dim is not None else action_dim)
        self.num_goal_tokens = int(num_goal_tokens)
        self.num_latents = int(num_latents)
        self.num_pose_tokens = int(num_pose_tokens)
        self.num_layer_groups = int(num_layer_groups)
        self.latent_dim = int(latent_dim)
        self.pose_loss_weight = float(pose_loss_weight)
        self.fm_action_weight = float(fm_action_weight)
        for value, name in ((self.pose_loss_weight, 'pose_loss_weight'),
                            (self.fm_action_weight, 'fm_action_weight')):
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        self.prefix_timestep = float(prefix_timestep)
        if not torch.isfinite(torch.tensor(self.prefix_timestep)):
            raise ValueError('prefix_timestep must be finite')
        self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError('action_horizon must be positive')

        action_dit = _build_action_expert(action_expert,
                                          action_dim=self.action_dim,
                                          context_dim=self.context_dim)
        self.action_expert = LITLayerwiseActionExpert(
            action_dit, latent_dim=self.latent_dim,
            context_dim=self.context_dim)
        self.num_action_layers = self.action_expert.num_layers
        if self.num_action_layers % self.num_layer_groups != 0:
            raise ValueError(
                f'action expert num_layers ({self.num_action_layers}) must be '
                f'divisible by num_layer_groups ({self.num_layer_groups})')
        self.action_training_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(num_train_timesteps), shift=float(shift))
        self.action_inference_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=int(num_train_timesteps), shift=float(shift))

        self.coupling_layers: List[int] = []
        if coupling_layers is not None and len(tuple(coupling_layers)):
            layers = [int(layer) for layer in coupling_layers]
            if len(layers) != self.num_action_layers:
                raise ValueError(
                    'coupling_layers must have one entry per action-expert '
                    f'layer ({self.num_action_layers}), got {len(layers)}')
            if tuple(sorted(set(layers))) != tuple(layers):
                raise ValueError('coupling_layers must be strictly increasing')
            self.coupling_layers = layers
        elif prefix_depth is not None:
            self.coupling_layers = lit_coupling_layers(
                self.num_action_layers, int(prefix_depth))

        self.goal_pose_encoder: Optional[GoalPoseEncoder] = None
        self.semantic_visual_aggregator: Optional[SemanticVisualAggregator] = None
        self.semantic_visual_pose_norm: Optional[nn.LayerNorm] = None
        self.semantic_visual_pose_decoder: Optional[GoalPoseDecoder] = None
        if self.stage == 'stage1':
            self.goal_pose_encoder = GoalPoseEncoder(
                pose_dim=self.pose_dim, num_tokens=self.num_goal_tokens,
                hidden_size=self.prefix_input_dim, inner_dim=int(inner_dim))
        else:
            self.semantic_visual_aggregator = SemanticVisualAggregator(
                num_tokens=self.num_latents, latent_dim=self.latent_dim,
                context_dim=self.context_dim,
                num_layer_groups=self.num_layer_groups,
                num_pose_tokens=self.num_pose_tokens,
                num_heads=int(num_heads), ffn_ratio=float(ffn_ratio),
                dropout=float(dropout),
                context_token_dropout=float(context_token_dropout),
                context_blackout_prob=float(context_blackout_prob))
            self.semantic_visual_pose_norm = nn.LayerNorm(self.latent_dim)
            self.semantic_visual_pose_decoder = GoalPoseDecoder(
                num_tokens=self.num_pose_tokens, hidden_size=self.latent_dim,
                pose_dim=self.pose_dim, inner_dim=int(inner_dim))
        self.apply_trainable_policy()

    # ------------------------------------------------------------------
    # Stage policy
    # ------------------------------------------------------------------
    def _all_lit_modules(self) -> List[Optional[nn.Module]]:
        return [
            self.goal_pose_encoder,
            self.semantic_visual_aggregator,
            self.semantic_visual_pose_norm,
            self.semantic_visual_pose_decoder,
            self.action_expert,
        ]

    def stage1_trainable_modules(self) -> List[nn.Module]:
        return [module for module in (self.goal_pose_encoder, self.action_expert)
                if module is not None]

    def stage2_trainable_modules(self) -> List[nn.Module]:
        return [module for module in self._all_lit_modules()
                if module is not None
                and module is not self.goal_pose_encoder]

    def trainable_modules(self) -> List[nn.Module]:
        return (self.stage1_trainable_modules() if self.stage == 'stage1'
                else self.stage2_trainable_modules())

    def apply_trainable_policy(self) -> 'PatchWAMLITFlowMatchingHead':
        """Set ``requires_grad`` from the stage; the VLA owns shared modules."""
        trainable = {id(module) for module in self.trainable_modules()}
        for module in self._all_lit_modules():
            if module is None:
                continue
            module.requires_grad_(id(module) in trainable)
            module.train(id(module) in trainable)
        return self

    def train(self, mode: bool = True) -> 'PatchWAMLITFlowMatchingHead':
        super().train(mode)
        return self.apply_trainable_policy() if mode else self

    def stage_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for module in self.trainable_modules():
            params.extend(param for param in module.parameters()
                          if param.requires_grad)
        return params

    # ------------------------------------------------------------------
    # Stage 1: goal-conditioned, image-free action prior
    # ------------------------------------------------------------------
    def encode_goal(self, goal_state: Tensor,
                    goal_valid: Optional[Tensor] = None) -> Tensor:
        """Encode the normalized terminal goal into ``num_goal_tokens`` tokens."""
        if self.goal_pose_encoder is None:
            raise ValueError('encode_goal requires a stage1 head')
        if goal_state.ndim != 2 or goal_state.shape[1] != self.pose_dim:
            raise ValueError(
                f'goal_state must be [B, {self.pose_dim}], got '
                f'{tuple(goal_state.shape)}')
        if goal_valid is not None:
            if goal_valid.shape != (goal_state.shape[0],):
                raise ValueError('goal_valid must be [B]')
            goal_state = torch.where(
                goal_valid.to(device=goal_state.device,
                              dtype=torch.bool)[:, None],
                goal_state, torch.zeros_like(goal_state))
        return self.goal_pose_encoder(goal_state)

    def stage1_prefix(self, context: Tensor, context_valid: Optional[Tensor],
                      goal_state: Tensor, goal_valid: Optional[Tensor]):
        """``[language, state, goal]`` prefix tokens plus their validity mask."""
        if context.ndim != 3 or context.shape[-1] != self.prefix_input_dim:
            raise ValueError(
                f'context must be [B, L, {self.prefix_input_dim}], got '
                f'{tuple(context.shape)}')
        goal_tokens = self.encode_goal(goal_state, goal_valid)
        if goal_valid is None:
            goal_mask = torch.ones(goal_tokens.shape[0], goal_tokens.shape[1],
                                   dtype=torch.bool, device=context.device)
        else:
            goal_mask = goal_valid.to(device=context.device,
                                      dtype=torch.bool)[:, None].expand(
                                          -1, goal_tokens.shape[1])
        if context_valid is None:
            context_valid = torch.ones(context.shape[:2], dtype=torch.bool,
                                       device=context.device)
        return (torch.cat((context, goal_tokens.to(context)), dim=1),
                torch.cat((context_valid.to(device=context.device,
                                            dtype=torch.bool), goal_mask),
                          dim=1))

    def stage1_loss(self, prefix_hidden: Tensor,
                    prefix_mask: Optional[Tensor],
                    batch: Mapping[str, Any],
                    generator: Optional[torch.Generator] = None
                    ) -> Dict[str, Tensor]:
        if self.stage != 'stage1':
            raise ValueError('stage1_loss requires a stage1 head')
        noisy, _, target, timestep = self._flow_perturbation(
            batch, generator=generator)
        prediction = self.action_expert.forward_single_context(
            noisy, timestep, prefix_hidden, prefix_mask)
        loss_action = self._masked_action_loss(prediction, target, batch,
                                              timestep)
        loss = self.fm_action_weight * loss_action
        zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
        return {
            'loss': loss,
            'loss_action': (self.fm_action_weight * loss_action).detach(),
            'loss_goal': zero,
        }

    # ------------------------------------------------------------------
    # Stage 2: aggregator, goal reconstruction and layer-wise expert
    # ------------------------------------------------------------------
    def stage2_contexts(
        self,
        prefix_runner: Callable[[Callable[[int, Tensor, Tensor], None]], Any],
        *,
        batch_size: int,
        device: torch.device,
        semantic_valid: Optional[Tensor] = None,
        image_valid: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        training: Optional[bool] = None,
    ) -> tuple:
        """Stream the prefix memories through the aggregator.

        ``prefix_runner(capture)`` must run the shared FLUX prefix over the
        current observation only and call
        ``capture(position, semantic_hidden, visual_hidden)`` at each coupling
        layer. Only the 768-wide latent contexts are retained, never
        full-resolution per-layer features. Returns ``(contexts,
        final_latents)``; the action expert consumes the latents alone, so the
        prefix never needs to be handed over.
        """
        if self.stage != 'stage2':
            raise ValueError('stage2_contexts requires a stage2 head')
        aggregator = self.semantic_visual_aggregator
        if aggregator is None:
            raise ValueError('stage2 requires the semantic/visual aggregator')
        latents = aggregator.initial_queries(int(batch_size), device)
        keep_mask = aggregator.sample_context_keep_mask(
            int(batch_size), device, training=training, generator=generator)
        collected: List[Tensor] = []
        num_couplings = len(self.coupling_layers)
        if num_couplings < 1:
            raise ValueError(
                'Stage2 requires coupling_layers (or prefix_depth) so the '
                'action expert sees per-depth contexts')

        def capture(position: int, semantic_hidden: Tensor,
                    visual_hidden: Tensor) -> None:
            nonlocal latents
            layer_idx = int(position)
            if not 0 <= layer_idx < num_couplings:
                raise ValueError(
                    f'Unexpected coupling position {layer_idx}; expected '
                    f'{num_couplings} captures')
            latent_batch = latents.shape[0]
            semantic_mask = (
                semantic_valid if semantic_valid is not None else
                torch.ones(semantic_hidden.shape[:2], dtype=torch.bool,
                           device=semantic_hidden.device))
            image_mask = (
                image_valid if image_valid is not None else
                torch.ones(visual_hidden.shape[:2], dtype=torch.bool,
                           device=visual_hidden.device))
            if semantic_mask.shape[0] != latent_batch or \
                    image_mask.shape[0] != latent_batch:
                raise ValueError(
                    'Prefix memory batch size does not match the aggregator '
                    'latents')
            latents = aggregator.forward_layer(
                latents, semantic_hidden, visual_hidden,
                semantic_mask=semantic_mask, image_mask=image_mask,
                layer_idx=layer_idx, num_layers=num_couplings,
                context_keep_mask=keep_mask)
            collected.append(latents)

        result = prefix_runner(capture)
        if len(collected) != num_couplings:
            raise ValueError(
                f'prefix runner produced {len(collected)} captures but the '
                f'head expects {num_couplings}')
        del result
        return collected, latents

    def stage2_loss(self, contexts: Sequence[Tensor], final_latents: Tensor,
                    batch: Mapping[str, Any],
                    generator: Optional[torch.Generator] = None
                    ) -> Dict[str, Tensor]:
        if self.stage != 'stage2':
            raise ValueError('stage2_loss requires a stage2 head')
        if (self.semantic_visual_pose_decoder is None
                or self.semantic_visual_pose_norm is None):
            raise ValueError('stage2_loss requires the pose reconstruction head')
        noisy, _, target, timestep = self._flow_perturbation(
            batch, generator=generator)
        prediction = self.action_expert.forward_layerwise(
            noisy, timestep, list(contexts))
        loss_action = self._masked_action_loss(prediction, target, batch,
                                              timestep)
        goal_state = batch.get('goal_state')
        if goal_state is None:
            raise ValueError(
                'Stage2 requires `goal_state` (the normalized terminal state '
                'at t + horizon)')
        goal_valid = batch.get('goal_valid')
        pose_hidden = self.semantic_visual_pose_norm(
            final_latents[:, :self.num_pose_tokens])
        pred_pose = self.semantic_visual_pose_decoder(pose_hidden)
        if goal_state.ndim != 2 or goal_state.shape[1] != self.pose_dim:
            raise ValueError(
                f'goal_state must be [B, {self.pose_dim}], got '
                f'{tuple(goal_state.shape)}')
        loss_goal = compute_pose_reconstruction_loss(
            pred_pose.to(goal_state), goal_state,
            is_pad=None if goal_valid is None else ~goal_valid.to(
                device=goal_state.device, dtype=torch.bool))
        loss = (self.fm_action_weight * loss_action
                + self.pose_loss_weight * loss_goal)
        return {
            'loss': loss,
            'loss_action': (self.fm_action_weight * loss_action).detach(),
            'loss_goal': (self.pose_loss_weight * loss_goal).detach(),
        }

    # ------------------------------------------------------------------
    # Flow helpers
    # ------------------------------------------------------------------
    def _flow_perturbation(self, batch: Mapping[str, Any],
                           generator: Optional[torch.Generator] = None
                           ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action = batch.get('action')
        if action is None:
            raise ValueError('The LIT flow objective requires `action`')
        if action.ndim != 3 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f'action must be [B, H, {self.action_dim}], got '
                f'{tuple(action.shape)}')
        clean = self.clean_actions(batch)
        noise = torch.randn(clean.shape, device=clean.device,
                            dtype=clean.dtype, generator=generator)
        timestep = self.action_training_scheduler.sample_training_t(
            batch_size=clean.shape[0], device=clean.device, dtype=clean.dtype)
        noisy = self.action_training_scheduler.add_noise(clean, noise, timestep)
        target = self.action_training_scheduler.training_target(
            clean, noise, timestep)
        return noisy, clean, target, timestep

    def clean_actions(self, batch: Mapping[str, Any]) -> Tensor:
        """Zero padded coordinates so they cannot leak into the flow target."""
        action = batch['action']
        return torch.where(self.action_validity(batch, action), action,
                           torch.zeros_like(action))

    @staticmethod
    def action_validity(batch: Mapping[str, Any], action: Tensor) -> Tensor:
        """Native polarity ``True`` = valid coordinate, shaped ``[B, H, A]``."""
        device = action.device
        step_pad = batch.get('action_is_pad')
        dim_pad = batch.get('action_dim_is_pad')
        if step_pad is None:
            valid = torch.ones_like(action, dtype=torch.bool)
        else:
            step_pad = step_pad.to(device=device, dtype=torch.bool)
            if step_pad.shape != action.shape[:2]:
                raise ValueError('action_is_pad must be [B, H]')
            valid = (~step_pad)[..., None].expand_as(action)
        if dim_pad is not None:
            dim_pad = dim_pad.to(device=device, dtype=torch.bool)
            if dim_pad.ndim == 2:
                dim_pad = dim_pad[:, None, :]
            if dim_pad.shape[-1] != action.shape[-1]:
                raise ValueError(
                    'action_dim_is_pad must be [B, A] or [B, H, A]')
            try:
                valid = valid & ~torch.broadcast_to(dim_pad, action.shape)
            except RuntimeError as exc:
                raise ValueError(
                    'action_dim_is_pad cannot broadcast to [B, H, A]') from exc
        return valid

    def _masked_action_loss(self, prediction: Tensor, target: Tensor,
                            batch: Mapping[str, Any],
                            timestep: Tensor) -> Tensor:
        valid = self.action_validity(batch, batch['action'])
        per_sample = masked_sample_mse(prediction, target, valid)
        weight = self.action_training_scheduler.training_weight(timestep).to(
            device=per_sample.device, dtype=per_sample.dtype)
        return (per_sample * weight).mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        prefix_hidden: Optional[Tensor] = None,
        prefix_mask: Optional[Tensor] = None,
        *,
        contexts: Optional[Sequence[Tensor]] = None,
        horizon: Optional[int] = None,
        steps: int = 10,
        generator: Optional[torch.Generator] = None,
        initial_action: Optional[Tensor] = None,
    ) -> Tensor:
        """Denoise normalized ``[B, H, action_dim]`` actions.

        Stage 1 conditions on the goal-conditioned prefix; Stage 2 conditions
        on the aggregator's latent contexts only.
        """
        steps = int(steps)
        if steps <= 0:
            raise ValueError('steps must be positive')
        horizon = self.action_horizon if horizon is None else int(horizon)
        if horizon < 1:
            raise ValueError('horizon must be positive')
        if self.stage == 'stage1':
            if prefix_hidden is None:
                raise ValueError('Stage-1 sampling requires the goal prefix')
            reference = prefix_hidden
        else:
            if contexts is None or len(contexts) != self.num_action_layers:
                raise ValueError(
                    'Stage-2 sampling requires one latent context per expert '
                    f'layer ({self.num_action_layers})')
            reference = contexts[0]
        batch = reference.shape[0]
        action = initial_action
        if action is None:
            action = torch.randn(batch, horizon, self.action_dim,
                                 device=reference.device,
                                 dtype=reference.dtype, generator=generator)
        if action.shape != (batch, horizon, self.action_dim):
            raise ValueError(
                f'initial_action must be [B, {horizon}, {self.action_dim}]')
        timesteps, deltas = \
            self.action_inference_scheduler.build_inference_schedule(
                steps, reference.device, reference.dtype)
        for index in range(steps):
            timestep = timesteps[index].expand(batch)
            if self.stage == 'stage1':
                velocity = self.action_expert.forward_single_context(
                    action, timestep, prefix_hidden, prefix_mask)
            else:
                velocity = self.action_expert.forward_layerwise(
                    action, timestep, list(contexts))
            action = self.action_inference_scheduler.step(
                velocity, deltas[index], action)
        return action

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def checkpoint_contract(self) -> Dict[str, Any]:
        return {
            'stage': self.stage,
            'action_dim': self.action_dim,
            'context_dim': self.context_dim,
            'prefix_input_dim': self.prefix_input_dim,
            'num_action_layers': self.num_action_layers,
            'num_layer_groups': self.num_layer_groups,
            'coupling_layers': list(self.coupling_layers),
        }
