# Copyright 2026 PatchWAM contributors.
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
"""Latent Interface Training (LIT) modules for the native PatchWAM stack.

Source: the ``goal_pose_prior`` modules of ImageWAM
(``imagewam/models/backbones/goal_pose_prior.py``), itself adapted from the
MolmoAct2 V3 goal-pose prior, Apache-2.0. The strict LIT variant is copied
here: a goal-pose encoder (Stage 1), a recurrent 100-latent aggregator with
depth-grouped parameters (Stage 2), a pose reconstruction head, and the
layer-wise action-expert wrapper that consumes the per-depth latents.

The relaxed ImageWAM ablations (Plan-B ``action_sees_ref``, grid latents,
cold-start gating) are intentionally not ported: the action expert never
attends the raw reference image, so the LIT bottleneck stays strict. Knobs
that would silently no-op instead raise.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from ..fastwam.modules.helpers.gradient import gradient_checkpoint_forward

__all__ = [
    'LIT_NUM_GOAL_TOKENS',
    'LIT_NUM_LATENTS',
    'LIT_NUM_LAYER_GROUPS',
    'LIT_NUM_POSE_TOKENS',
    'LIT_POSE_LOSS_WEIGHT',
    'LIT_STAGES',
    'LIT_STAGE1_ONLY_PREFIXES',
    'LIT_STAGE2_ONLY_PREFIXES',
    'GoalPoseDecoder',
    'GoalPoseEncoder',
    'LITLayerwiseActionExpert',
    'SemanticVisualAggregator',
    'compute_pose_reconstruction_loss',
    'layer_group_index',
    'lit_coupling_layers',
    'validate_lit_checkpoint_keys',
]

LIT_NUM_GOAL_TOKENS = 8
LIT_NUM_LATENTS = 100
LIT_NUM_POSE_TOKENS = 8
LIT_NUM_LAYER_GROUPS = 5
LIT_LATENT_DIM = 768
LIT_INNER_DIM = 512
LIT_FFN_RATIO = 4.0
LIT_NUM_HEADS = 8
LIT_DROPOUT = 0.0
LIT_POSE_LOSS_WEIGHT = 0.3

# Model-state prefixes that exist in exactly one stage. The bridge from Stage 1
# to Stage 2 tolerates the Stage-2-only modules as newly initialized weights and
# the Stage-1-only goal encoder as a dropped input. Everything else must match.
LIT_STAGE1_ONLY_PREFIXES = ('lit_head.goal_pose_encoder.',)
LIT_STAGE2_ONLY_PREFIXES = (
    'lit_head.semantic_visual_aggregator.',
    'lit_head.semantic_visual_pose_norm.',
    'lit_head.semantic_visual_pose_decoder.',
)
LIT_STAGES = ('stage1', 'stage2')


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rrms = torch.rsqrt(torch.mean(x_float**2, dim=-1, keepdim=True) + self.eps)
        return (x_float * rrms).to(dtype=dtype) * self.scale


class GoalPoseEncoder(nn.Module):
    """Maps a normalized goal state vector to K FLUX hidden tokens."""

    def __init__(
        self,
        pose_dim: int,
        num_tokens: int = LIT_NUM_GOAL_TOKENS,
        hidden_size: int = 3072,
        inner_dim: int = LIT_INNER_DIM,
    ) -> None:
        super().__init__()
        if int(pose_dim) < 1:
            raise ValueError('pose_dim must be positive')
        if int(num_tokens) < 1:
            raise ValueError('num_tokens must be positive')
        self.pose_dim = int(pose_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.pose_dim, int(inner_dim)),
            nn.GELU(),
            nn.Linear(int(inner_dim), int(inner_dim)),
            nn.GELU(),
            nn.Linear(int(inner_dim), self.num_tokens * self.hidden_size),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.ndim != 2:
            raise ValueError(f'`pose` must be [B, pose_dim], got {tuple(pose.shape)}')
        if pose.shape[1] != self.pose_dim:
            raise ValueError(
                f'`pose` last dim must be {self.pose_dim}, got {pose.shape[1]}')
        return self.net(pose).view(pose.shape[0], self.num_tokens, self.hidden_size)


class GoalPoseDecoder(nn.Module):
    """Reconstructs a normalized goal state from the pose latents."""

    def __init__(
        self,
        num_tokens: int = LIT_NUM_POSE_TOKENS,
        hidden_size: int = LIT_LATENT_DIM,
        pose_dim: int = 14,
        inner_dim: int = LIT_INNER_DIM,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_size = int(hidden_size)
        self.pose_dim = int(pose_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_tokens * self.hidden_size, int(inner_dim)),
            nn.GELU(),
            nn.Linear(int(inner_dim), int(inner_dim)),
            nn.GELU(),
            nn.Linear(int(inner_dim), self.pose_dim),
        )

    def forward(self, goal_hidden: torch.Tensor) -> torch.Tensor:
        if goal_hidden.ndim != 3:
            raise ValueError(
                f'`goal_hidden` must be [B, K, D], got {tuple(goal_hidden.shape)}')
        if (goal_hidden.shape[1] != self.num_tokens
                or goal_hidden.shape[2] != self.hidden_size):
            raise ValueError(
                f'`goal_hidden` must be [B, {self.num_tokens}, '
                f'{self.hidden_size}], got {tuple(goal_hidden.shape)}')
        return self.net(goal_hidden.reshape(goal_hidden.shape[0], -1))


class _SelfAttentionBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int, ffn_ratio: float,
                 dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, dropout=dropout,
            batch_first=True)
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(tokens)
        update, _ = self.self_attn(normalized, normalized, normalized,
                                   need_weights=False)
        tokens = tokens + self.dropout(update)
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))


class _CrossAttentionBlock(nn.Module):
    """Latent queries attending a masked memory (semantic or visual)."""

    def __init__(self, latent_dim: int, context_dim: int, num_heads: int,
                 ffn_ratio: float, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, dropout=dropout,
            kdim=context_dim, vdim=context_dim, batch_first=True)
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(
                f'`context` must be [B, S, D], got {tuple(context.shape)}')
        if valid_mask.ndim != 2 or valid_mask.shape != context.shape[:2]:
            raise ValueError(
                f'`valid_mask` must be [B, S]={tuple(context.shape[:2])}, '
                f'got {tuple(valid_mask.shape)}')
        valid_mask = valid_mask.to(device=context.device, dtype=torch.bool)
        # Invalid slots are zeroed so masked NaNs can never reach SDPA. A
        # non-finite value inside a *valid* slot is a real data error.
        finite = torch.isfinite(context).all(dim=-1, keepdim=True)
        if bool((finite.logical_not() & valid_mask[..., None]).any()):
            raise ValueError('`context` contains non-finite values')
        context = torch.where(valid_mask[..., None], context,
                              torch.zeros_like(context))
        key_padding_mask = ~valid_mask
        # MultiheadAttention forbids fully-masked keys; keep one dummy slot.
        fully_masked = key_padding_mask.all(dim=1)
        if bool(fully_masked.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_masked, 0] = False
        normalized = self.context_norm(context)
        update, _ = self.cross_attn(
            self.query_norm(queries), normalized, normalized,
            key_padding_mask=key_padding_mask, need_weights=False)
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class SemanticVisualAggregatorGroup(nn.Module):
    """One depth-group: latent self-attention, then semantic, then visual."""

    def __init__(self, latent_dim: int, context_dim: int, num_heads: int,
                 ffn_ratio: float, dropout: float) -> None:
        super().__init__()
        self.self_block = _SelfAttentionBlock(latent_dim, num_heads, ffn_ratio,
                                              dropout)
        self.semantic_block = _CrossAttentionBlock(latent_dim, context_dim,
                                                   num_heads, ffn_ratio, dropout)
        self.visual_block = _CrossAttentionBlock(latent_dim, context_dim,
                                                 num_heads, ffn_ratio, dropout)

    def forward(self, queries: torch.Tensor, semantic_hidden: torch.Tensor,
                visual_hidden: torch.Tensor, *, semantic_mask: torch.Tensor,
                image_mask: torch.Tensor) -> torch.Tensor:
        queries = self.self_block(queries)
        queries = self.semantic_block(queries, semantic_hidden, semantic_mask)
        return self.visual_block(queries, visual_hidden, image_mask)


def layer_group_index(layer_idx: int, num_layers: int,
                      num_layer_groups: int) -> int:
    """Depth sharing: expert layer -> aggregator parameter group."""
    if int(num_layers) <= 0:
        raise ValueError(f'`num_layers` must be positive, got {num_layers}')
    if int(num_layer_groups) <= 0:
        raise ValueError(
            f'`num_layer_groups` must be positive, got {num_layer_groups}')
    if int(num_layers) % int(num_layer_groups) != 0:
        raise ValueError(
            f'`num_layers` ({num_layers}) must be divisible by '
            f'`num_layer_groups` ({num_layer_groups})')
    if not 0 <= int(layer_idx) < int(num_layers):
        raise ValueError(
            f'`layer_idx` must be in [0, {num_layers}), got {layer_idx}')
    return int(layer_idx) // (int(num_layers) // int(num_layer_groups))


class SemanticVisualAggregator(nn.Module):
    """Recurrent 100-token aggregator with depth-grouped parameters."""

    def __init__(
        self,
        num_tokens: int = LIT_NUM_LATENTS,
        latent_dim: int = LIT_LATENT_DIM,
        context_dim: int = 3072,
        num_layer_groups: int = LIT_NUM_LAYER_GROUPS,
        num_pose_tokens: int = LIT_NUM_POSE_TOKENS,
        num_heads: int = LIT_NUM_HEADS,
        ffn_ratio: float = LIT_FFN_RATIO,
        dropout: float = LIT_DROPOUT,
        context_token_dropout: float = 0.0,
        context_blackout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.num_pose_tokens = int(num_pose_tokens)
        if not 1 <= self.num_pose_tokens <= self.num_tokens:
            raise ValueError(
                f'`num_pose_tokens` ({num_pose_tokens}) must be in '
                f'[1, num_tokens={self.num_tokens}]')
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.num_layer_groups = int(num_layer_groups)
        if self.num_layer_groups < 1:
            raise ValueError('`num_layer_groups` must be >= 1')
        self.context_token_dropout = float(context_token_dropout)
        if not 0.0 <= self.context_token_dropout < 1.0:
            raise ValueError(
                f'`context_token_dropout` must be in [0, 1), got '
                f'{context_token_dropout}')
        self.context_blackout_prob = float(context_blackout_prob)
        if not 0.0 <= self.context_blackout_prob < 1.0:
            raise ValueError(
                f'`context_blackout_prob` must be in [0, 1), got '
                f'{context_blackout_prob}')
        queries = torch.empty(self.num_tokens, self.latent_dim)
        nn.init.trunc_normal_(queries, std=0.02)
        self.queries = nn.Parameter(queries)
        self.groups = nn.ModuleList(
            [
                SemanticVisualAggregatorGroup(
                    latent_dim=self.latent_dim,
                    context_dim=self.context_dim,
                    num_heads=int(num_heads),
                    ffn_ratio=float(ffn_ratio),
                    dropout=float(dropout),
                )
                for _ in range(self.num_layer_groups)
            ]
        )

    def initial_queries(self, batch_size: int,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(int(batch_size), -1, -1)
        return queries.contiguous() if device is None else queries.to(
            device).contiguous()

    def sample_context_keep_mask(
        self,
        batch_size: int,
        device: torch.device,
        *,
        training: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[torch.Tensor]:
        """[B, num_tokens] bool, True = keep. Pose tokens are never dropped."""
        if training is None:
            training = self.training
        if not training:
            return None
        if self.context_token_dropout <= 0.0 and self.context_blackout_prob <= 0.0:
            return None
        batch_size = int(batch_size)
        keep = torch.ones(batch_size, self.num_tokens, dtype=torch.bool,
                          device=device)
        n_context = self.num_tokens - self.num_pose_tokens
        if n_context <= 0:
            return keep
        if self.context_token_dropout > 0.0:
            drop = torch.rand(batch_size, n_context, device=device,
                              generator=generator) < self.context_token_dropout
            keep[:, self.num_pose_tokens:] = ~drop
        if self.context_blackout_prob > 0.0:
            # Fixed count so every batch carries both regimes on every rank.
            count = int(round(self.context_blackout_prob * batch_size))
            if count <= 0 and self.context_blackout_prob > 0.0:
                count = 1 if batch_size > 1 else 0
            count = min(count, max(batch_size - 1, 0))
            if count > 0:
                order = torch.randperm(batch_size, device=device,
                                       generator=generator)[:count]
                keep[order, self.num_pose_tokens:] = False
        return keep

    def forward_layer(
        self,
        queries: torch.Tensor,
        semantic_hidden: torch.Tensor,
        visual_hidden: torch.Tensor,
        *,
        semantic_mask: torch.Tensor,
        image_mask: torch.Tensor,
        layer_idx: int,
        num_layers: int,
        context_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_keep_mask is not None:
            if context_keep_mask.shape != (queries.shape[0], self.num_tokens):
                raise ValueError(
                    '`context_keep_mask` must be [B, num_tokens]='
                    f'{(queries.shape[0], self.num_tokens)}, got '
                    f'{tuple(context_keep_mask.shape)}')
            pose = context_keep_mask[:, :self.num_pose_tokens]
            if not bool(pose.all()):
                raise ValueError('pose latents must never be dropped')
            keep = context_keep_mask.to(device=queries.device, dtype=torch.bool)
            queries = torch.where(keep[..., None], queries,
                                  torch.zeros_like(queries))
        group = self.groups[layer_group_index(layer_idx, num_layers,
                                              self.num_layer_groups)]
        return group(queries, semantic_hidden, visual_hidden,
                     semantic_mask=semantic_mask, image_mask=image_mask)


def compute_pose_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_pad: Optional[torch.Tensor] = None,
    dim_is_pad: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked MSE on the normalized goal state with NaN-safe gradients.

    Invalid coordinates are replaced by zeros *before* the subtraction, so a
    NaN in a masked prediction or target can neither reach the loss nor poison
    ``pred.grad``. Non-finite values inside a *valid* coordinate are a real data
    or model error and raise instead of being silently masked.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f'pose pred/target shape mismatch: {tuple(pred.shape)} vs '
            f'{tuple(target.shape)}')
    pred_values = pred.float()
    target_values = target.float()
    valid = torch.ones_like(pred_values, dtype=torch.bool)
    if dim_is_pad is not None:
        dimension_valid = dim_is_pad.to(device=pred.device, dtype=torch.bool)
        if dimension_valid.ndim == 1:
            dimension_valid = dimension_valid.unsqueeze(0).expand_as(valid)
        if dimension_valid.shape != valid.shape:
            raise ValueError(
                '`dim_is_pad` must be [A] or [B, A] matching the goal shape')
        valid = valid & ~dimension_valid
    if is_pad is not None:
        sample_valid = is_pad.reshape(-1).to(device=pred.device,
                                             dtype=torch.bool)
        if sample_valid.shape[0] != pred.shape[0]:
            raise ValueError('`is_pad` must be [B]')
        valid = valid & ~sample_valid[:, None]
    finite = torch.isfinite(pred_values) & torch.isfinite(target_values)
    if bool((valid & ~finite).any()):
        raise ValueError(
            'Non-finite goal values inside valid coordinates; a padded '
            'coordinate may be NaN, a supervised one may not')
    safe_pred = torch.where(valid, pred_values, torch.zeros_like(pred_values))
    safe_target = torch.where(valid, target_values,
                              torch.zeros_like(target_values))
    per_dim = (safe_pred - safe_target).square()
    per_sample = per_dim.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    supervise = valid.any(dim=1)
    return (per_sample * supervise).sum() / supervise.sum().clamp_min(1)


class LITLayerwiseActionExpert(nn.Module):
    """Layer-wise context driver around the native FastWAM ``ActionDiT``.

    ``ActionDiT.forward`` applies one shared context to every block. LIT needs
    an independently updated latent context per expert layer, so this wrapper
    reuses the *native* DiT pieces (``pre_dit`` / blocks / ``post_dit`` /
    ``text_embedding``) and drives the block loop itself. The native blocks,
    modulation, RoPE cache, and gradient-checkpoint helper are untouched.
    """

    def __init__(self, action_dit: nn.Module, *, latent_dim: int,
                 context_dim: int) -> None:
        super().__init__()
        self.action_dit = action_dit
        dit_context = getattr(action_dit, 'text_dim', None)
        if dit_context is not None and int(dit_context) != int(context_dim):
            raise ValueError(
                f'ActionDiT.text_dim ({dit_context}) must equal the LIT '
                f'context width ({context_dim})')
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.num_layers = len(action_dit.blocks)
        self.context_projection = nn.Linear(self.latent_dim, self.context_dim)

    def _run_blocks(self, action_tokens: torch.Tensor, timestep: torch.Tensor,
                    contexts: Sequence[torch.Tensor],
                    context_masks: Optional[Sequence[Optional[torch.Tensor]]]
                    ) -> torch.Tensor:
        if len(contexts) != self.num_layers:
            raise ValueError(
                f'Expected one context per expert layer ({self.num_layers}), '
                f'got {len(contexts)}')
        if context_masks is not None and len(context_masks) != len(contexts):
            raise ValueError('One context mask per context is required')
        reference = contexts[0]
        # Match the shared prefix dtype/device (the native PatchWAM stack casts
        # action tokens to the encoded observation dtype the same way).
        action_tokens = action_tokens.to(dtype=reference.dtype,
                                         device=reference.device)
        timestep = timestep.to(dtype=reference.dtype, device=reference.device)
        pre_state = self.action_dit.pre_dit(
            action_tokens=action_tokens, timestep=timestep,
            context=contexts[0],
            context_mask=(None if context_masks is None
                          else context_masks[0]))
        states = pre_state['tokens']
        t_mod = pre_state['t_mod']
        freqs = pre_state['freqs']
        sequence_length = states.shape[1]
        checkpoint = (self.action_dit.use_gradient_checkpointing
                      and self.training)
        for index, block in enumerate(self.action_dit.blocks):
            embedded = self.action_dit.text_embedding(contexts[index])
            mask = None
            if context_masks is not None and context_masks[index] is not None:
                mask = context_masks[index].to(device=states.device,
                                               dtype=torch.bool)
                if mask.ndim != 2 or mask.shape[0] != states.shape[0]:
                    raise ValueError('Context masks must be [B, L]')
                # A fully masked context would make softmax(all -inf) NaN; keep
                # one dummy slot exactly like the aggregator does.
                fully_masked = ~mask.any(dim=1)
                if bool(fully_masked.any()):
                    mask = mask.clone()
                    mask[fully_masked, 0] = True
                mask = mask[:, None, :].expand(-1, sequence_length, -1)
            states = gradient_checkpoint_forward(
                block, checkpoint, states, embedded, t_mod, freqs,
                context_mask=mask)
        return self.action_dit.post_dit(states, pre_state)

    def forward_single_context(self, action_tokens: torch.Tensor,
                               timestep: torch.Tensor,
                               context: torch.Tensor,
                               context_mask: Optional[torch.Tensor] = None,
                               ) -> torch.Tensor:
        """Stage 1: one goal-conditioned prefix for every expert layer."""
        return self._run_blocks(action_tokens, timestep,
                                [context] * self.num_layers,
                                [context_mask] * self.num_layers)

    def forward_layerwise(self, action_tokens: torch.Tensor,
                          timestep: torch.Tensor,
                          latents: Sequence[torch.Tensor],
                          latent_masks: Optional[Sequence[torch.Tensor]] = None,
                          ) -> torch.Tensor:
        """Stage 2: the action expert consumes the projected latents only.

        The strict LIT bottleneck deliberately has no text/image prefix on this
        path. The shared backbone memory is consumed *exclusively* by the
        aggregator, so the action expert cannot learn an image-to-action bypass
        through the semantic stream. Each expert layer receives the projected
        latents of its own depth; ``latent_masks`` are ``[B, num_tokens]``
        booleans where ``True`` marks an attendable latent.
        """
        if len(latents) != self.num_layers:
            raise ValueError(
                f'Expected one latent context per expert layer '
                f'({self.num_layers}), got {len(latents)}')
        if latent_masks is not None and len(latent_masks) != len(latents):
            raise ValueError(
                'One latent mask per latent context is required')
        contexts: list[torch.Tensor] = []
        masks: list[Optional[torch.Tensor]] = []
        for index in range(self.num_layers):
            latent = latents[index].to(self.context_projection.weight)
            mask = None if latent_masks is None else latent_masks[index]
            if mask is not None:
                mask = mask.to(device=latent.device, dtype=torch.bool)
                if mask.ndim != 2 or mask.shape != latent.shape[:2]:
                    raise ValueError(
                        'Latent masks must be [B, num_tokens] matching the '
                        'latent context')
                # Zero masked latents *before* the projection: a NaN inside a
                # masked slot would otherwise survive the zero attention weight
                # (0 * NaN) and poison the whole context.
                latent = torch.where(mask[..., None], latent,
                                     torch.zeros_like(latent))
            contexts.append(self.context_projection(latent))
            masks.append(mask)
        return self._run_blocks(action_tokens, timestep, contexts, masks)


def validate_lit_checkpoint_keys(
    *,
    stage: str,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    bridge_from_stage1: bool = False,
) -> None:
    """Fail-closed Stage1/Stage2 key check with exact category reporting."""
    stage = str(stage)
    if stage not in LIT_STAGES:
        raise ValueError(f'Unsupported LIT stage {stage!r}')
    missing = list(missing_keys)
    unexpected = list(unexpected_keys)
    if stage == 'stage1':
        if missing or unexpected:
            raise RuntimeError(
                f'Stage1 LIT checkpoint key mismatch: missing={missing[:20]}, '
                f'unexpected={unexpected[:20]}')
        return
    allowed_missing = LIT_STAGE2_ONLY_PREFIXES if bridge_from_stage1 else ()
    allowed_unexpected = LIT_STAGE1_ONLY_PREFIXES if bridge_from_stage1 else ()

    def _allowed(key: str, prefixes: Sequence[str]) -> bool:
        return any(key.startswith(prefix) for prefix in prefixes)

    bad_missing = [key for key in missing if not _allowed(key, allowed_missing)]
    bad_unexpected = [
        key for key in unexpected if not _allowed(key, allowed_unexpected)
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f'Stage2 LIT checkpoint key mismatch (bridge={bridge_from_stage1}): '
            f'missing={bad_missing[:20]}, unexpected={bad_unexpected[:20]}')


def lit_coupling_layers(num_couplings: int, depth: int) -> list[int]:
    """Evenly spread coupling layers across the FLUX prefix depth (1-based)."""
    if int(num_couplings) < 1:
        raise ValueError('num_couplings must be positive')
    if int(depth) < 1:
        raise ValueError('depth must be positive')
    if int(num_couplings) > int(depth):
        raise ValueError(
            f'num_couplings ({num_couplings}) cannot exceed the prefix depth '
            f'({depth})')
    if num_couplings == 1:
        return [int(depth)]
    step = int(depth) / int(num_couplings)
    layers = [max(1, int(round((index + 1) * step)))
              for index in range(int(num_couplings))]
    if any(later <= earlier for earlier, later in zip(layers, layers[1:])):
        raise ValueError('Coupling layer selection is not strictly increasing')
    return layers
