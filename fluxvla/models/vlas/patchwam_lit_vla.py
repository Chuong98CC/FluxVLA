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
"""PatchWAM + Latent Interface Training (LIT) VLA.

``PatchWAMLITVLA`` keeps the released PatchWAM ``vla_head`` and adds an
independently configurable ``lit_head``. The shared FLUX.2 transformer and the
proprio projection are owned exactly once, by ``vla_head`` (``PatchWAMHead``);
the LIT path only borrows the prefix feature API of that denoiser, so no second
transformer is registered and no module alias crosses the state-dict or FSDP
boundary.

Stage 1 trains an image-free, goal-conditioned LIT action prior. Stage 2 trains
the original PatchWAM branch jointly with the LIT interface, adds the 0.3
weighted terminal-goal reconstruction, and keeps the native video/action loss.
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from fluxvla.engines import HEADS, VLAS, initialize_overwatch
from fluxvla.engines.utils.builder import build_from_cfg
from ..heads.patchwam_head import PatchWAMHead  # noqa: F401  (registration)
from ..heads.patchwam_lit_head import \
    PatchWAMLITFlowMatchingHead  # noqa: F401  (registration)
from ..third_party_models.patchwam.geometry import (joint_visibility,
                                                    sequence_coordinates)
from ..third_party_models.patchwam.lit import (LIT_STAGES,
                                               lit_coupling_layers,
                                               validate_lit_checkpoint_keys)
from .patchwam_vla import PatchWAMVLA, released_action_patch_state

overwatch = initialize_overwatch(__name__)

__all__ = ['PatchWAMLITVLA']

_TRAINING_MODES = ('patchwam', 'lit', 'joint')
_ACTIVE_HEADS = ('patchwam', 'lit')
_LIT_BATCH_KEYS = ('goal_state', 'goal_valid')
# Stage 1 is image-free: any of these caller inputs is dropped, never kept.
_STAGE1_DROPPED_KEYS = (
    'images', 'video', 'camera_video', 'history_video', 'history_valid',
    'vl_current', 'vl_history', 'vl_history_valid', 'frame_masks', 'img_masks',
    'decoded_frames', 'world_frames',
)


@VLAS.register_module()
class PatchWAMLITVLA(PatchWAMVLA):
    """PatchWAM VLA with a jointly configurable LIT + flow-matching head.

    Args:
        lit_head: ``PatchWAMLITFlowMatchingHead`` config (``type``-keyed).
        lit_stage: ``stage1`` or ``stage2``.
        training_mode: ``patchwam``, ``lit`` or ``joint``. Defaults to ``lit``
            for Stage 1 and ``joint`` for Stage 2, matching the user-selected
            joint Stage-2 training.
        active_head: Inference head used by :meth:`predict_action` when no
            explicit ``head`` kwarg is supplied.
        patchwam_loss_weight / lit_loss_weight: Branch weights of the joint
            Stage-2 objective (both default to 1).
    """

    def __init__(
        self,
        *,
        lit_head: Optional[Dict] = None,
        lit_stage: str = 'stage2',
        training_mode: Optional[str] = None,
        active_head: Optional[str] = None,
        patchwam_loss_weight: float = 1.0,
        lit_loss_weight: float = 1.0,
        lit_stage1_checkpoint: Optional[str] = None,
        lit_stage1_required: bool = True,
        **kwargs: Any,
    ) -> None:
        stage = str(lit_stage)
        if stage not in LIT_STAGES:
            raise ValueError(f'lit_stage must be one of {LIT_STAGES}, got '
                             f'{lit_stage!r}')
        mode = training_mode or ('lit' if stage == 'stage1' else 'joint')
        if mode not in _TRAINING_MODES:
            raise ValueError(
                f'training_mode must be one of {_TRAINING_MODES}, got '
                f'{training_mode!r}')
        if stage == 'stage1' and mode != 'lit':
            raise ValueError(
                'Stage 1 is the image-free goal-conditioned prior and only '
                "supports training_mode='lit'; "
                f'got {mode!r}.')
        if stage == 'stage1' and lit_stage1_checkpoint:
            raise ValueError(
                '`lit_stage1_checkpoint` is the Stage-2 bridge input; a '
                'Stage-1 model must not configure it.')
        head_name = active_head or 'lit'
        if head_name not in _ACTIVE_HEADS:
            raise ValueError(
                f'active_head must be one of {_ACTIVE_HEADS}, got '
                f'{active_head!r}')
        if lit_head is None:
            raise ValueError('PatchWAMLITVLA requires a `lit_head` config.')
        for value, name in ((patchwam_loss_weight, 'patchwam_loss_weight'),
                            (lit_loss_weight, 'lit_loss_weight')):
            if not torch.isfinite(torch.tensor(float(value))) or float(value) < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        super().__init__(**kwargs)
        self.lit_stage = stage
        self.training_mode = mode
        self.active_head = head_name
        self.patchwam_loss_weight = float(patchwam_loss_weight)
        self.lit_loss_weight = float(lit_loss_weight)
        self.lit_stage1_checkpoint = (None if lit_stage1_checkpoint is None
                                      else str(lit_stage1_checkpoint))
        self.lit_stage1_required = bool(lit_stage1_required)
        self.lit_head = build_from_cfg(copy.deepcopy(lit_head), HEADS)
        if not isinstance(self.lit_head, PatchWAMLITFlowMatchingHead):
            raise TypeError(
                'lit_head must build a PatchWAMLITFlowMatchingHead, got '
                f'{type(self.lit_head)!r}')
        if self.lit_head.stage != self.lit_stage:
            raise ValueError(
                f'lit_head.stage ({self.lit_head.stage}) must match lit_stage '
                f'({self.lit_stage})')
        if self.lit_head.prefix_input_dim != self.vla_head.text_dim:
            raise ValueError(
                f'lit_head.prefix_input_dim ({self.lit_head.prefix_input_dim}) '
                f'must match the shared text width ({self.vla_head.text_dim})')
        if self.lit_head.action_dim != self.vla_head.codec.action_dim:
            raise ValueError(
                f'lit_head.action_dim ({self.lit_head.action_dim}) must match '
                f'vla_head.action_dim ({self.vla_head.codec.action_dim})')
        hidden = getattr(self.vla_head.denoiser, 'representation_dim', None)
        transformer = getattr(self.vla_head.denoiser, 'transformer', None)
        if transformer is not None:
            hidden = getattr(transformer, 'hidden_size', hidden)
        if hidden is not None and int(hidden) != self.lit_head.context_dim:
            raise ValueError(
                f'shared denoiser hidden width ({hidden}) must match '
                f'lit_head.context_dim ({self.lit_head.context_dim})')
        if not self.lit_head.coupling_layers:
            depth = getattr(self.vla_head.denoiser, 'representation_layers', None)
            if depth is None:
                raise ValueError(
                    'lit_head needs explicit `coupling_layers` when the shared '
                    'denoiser does not expose `representation_layers`')
            self.lit_head.coupling_layers = lit_coupling_layers(
                self.lit_head.num_action_layers, int(depth))
        self.all_module_keys = ['vlm_backbone', 'vla_head', 'lit_head']
        self.apply_lit_stage_policy()

    # ------------------------------------------------------------------
    # Freeze / train policy
    # ------------------------------------------------------------------
    def apply_lit_stage_policy(self) -> 'PatchWAMLITVLA':
        """Apply the Stage-1/Stage-2 trainability policy.

        AE/Qwen stay frozen exactly as in PatchWAM. Stage 1 additionally
        freezes the shared FLUX transformer, the proprio projection and the
        original branch; Stage 2 trains them jointly with the LIT interface.
        In Stage-2 ``patchwam`` mode the whole LIT interface is frozen, so no
        unused LIT parameter is ever handed to the optimizer.
        """
        backbone = self.vlm_backbone
        if backbone is not None:
            backbone.apply_trainable_policy()
            # Keep the native module-mode policy (AE always eval, VL encoders
            # following the requested mode) after the frozen-parameter pass.
            backbone.train(self.training)
        if self.lit_stage == 'stage1':
            if backbone is not None:
                backbone.requires_grad_(False)
            self.vla_head.requires_grad_(False)
            self.lit_head.apply_trainable_policy()
            return self
        # Stage 2: the shared FLUX prefix and proprio projection follow the
        # active objective; the LIT interface is only trained when the LIT
        # branch is active.
        self.vla_head.requires_grad_(True)
        if self.training_mode == 'patchwam':
            self.lit_head.requires_grad_(False)
            self.lit_head.eval()
        else:
            if self.training_mode == 'lit':
                # The LIT path reads the shared prefix only. The original
                # head's output modules (FLUX `final_layer`, tiny
                # `output_norm`/`token_output`) are never executed, so they must
                # not sit in the optimizer while the prefix stays trainable.
                for module in self.original_output_modules():
                    module.requires_grad_(False)
            self.lit_head.apply_trainable_policy()
        return self

    _ORIGINAL_OUTPUT_ATTRS = ('final_layer', 'output_norm', 'token_output')

    def original_output_modules(self) -> List[torch.nn.Module]:
        """Shared-head modules that only build the original output stream."""
        modules: List[torch.nn.Module] = []
        for owner in (self.vla_head.denoiser,
                      getattr(self.vla_head.denoiser, 'transformer', None)):
            if owner is None:
                continue
            for name in self._ORIGINAL_OUTPUT_ATTRS:
                module = getattr(owner, name, None)
                if isinstance(module, torch.nn.Module) and module not in modules:
                    modules.append(module)
        return modules

    def lit_branch_active(self) -> bool:
        """Whether the LIT head contributes to the objective."""
        return self.training_mode in ('lit', 'joint')

    def freeze_backbones(self) -> None:
        super().freeze_backbones()
        self.apply_lit_stage_policy()

    def train(self, mode: bool = True) -> 'PatchWAMLITVLA':
        super().train(mode)
        if mode:
            self.apply_lit_stage_policy()
        return self

    # ------------------------------------------------------------------
    # LIT prefix pass
    # ------------------------------------------------------------------
    def _prefix_denoiser(self) -> torch.nn.Module:
        denoiser = self.vla_head.denoiser
        if not hasattr(denoiser, 'forward_prefix'):
            raise TypeError(
                'The LIT prefix API requires the shared denoiser to implement '
                f'`forward_prefix`; {type(denoiser).__name__} does not')
        return denoiser

    def _run_lit_prefix(
        self,
        *,
        context: torch.Tensor,
        context_valid: Optional[torch.Tensor],
        reference: Optional[torch.Tensor] = None,
        reference_valid: Optional[torch.Tensor] = None,
        reference_ids: Optional[torch.Tensor] = None,
        capture_layers: Sequence[int] = (),
        capture: Optional[Callable[[int, torch.Tensor, torch.Tensor],
                                   None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the shared FLUX prefix over the current observation only."""
        denoiser = self._prefix_denoiser()
        batch = context.shape[0]
        device = context.device
        token_dim = self.vla_head.codec.token_dim
        if reference is None:
            reference = context.new_zeros(batch, 0, token_dim)
        else:
            reference = reference.to(device=device, dtype=context.dtype)
        if reference_valid is None:
            reference_valid = torch.ones(
                batch, reference.shape[1], dtype=torch.bool, device=device)
        else:
            reference_valid = reference_valid.to(device=device, dtype=torch.bool)
        if reference_ids is None:
            reference_ids = sequence_coordinates(batch, reference.shape[1],
                                                 group=10, device=device)
        else:
            reference_ids = reference_ids.to(device=device)
        if context_valid is None:
            context_valid = torch.ones(batch, context.shape[1],
                                       dtype=torch.bool, device=device)
        else:
            context_valid = context_valid.to(device=device, dtype=torch.bool)
        # Native PatchWAM contract: invalid keys/values are zeroed before the
        # transformer so a padded camera or text slot cannot leak through the
        # residual stream (masking attention alone does not stop that).
        context = torch.where(context_valid[..., None].to(context.device),
                              context, torch.zeros_like(context))
        reference = torch.where(reference_valid[..., None].to(reference.device),
                                reference, torch.zeros_like(reference))
        context_ids = sequence_coordinates(batch, context.shape[1],
                                           group=0, device=device)
        visibility = joint_visibility(
            context.shape[1], reference.shape[1], 0, 0,
            text_valid=context_valid, reference_valid=reference_valid,
            device=device)
        sigma = torch.full((batch,), self.lit_head.prefix_timestep,
                           device=device, dtype=context.dtype)
        layers = tuple(int(layer) for layer in capture_layers)
        translated: Optional[Callable[[int, torch.Tensor, torch.Tensor],
                                      None]] = capture
        if capture is not None and layers:
            # The prefix pass reports absolute 1-based layer indices; the LIT
            # head consumes the 0-based position inside the coupling list.
            order = {layer: position for position, layer in enumerate(layers)}

            def translated(index: int, semantic: torch.Tensor,
                           visual: torch.Tensor) -> None:
                if index not in order:
                    raise ValueError(
                        f'Prefix pass captured an unexpected layer {index}')
                capture(order[index], semantic, visual)
        text_states, _ = denoiser.forward_prefix(
            context, reference, sigma, context_ids=context_ids,
            reference_ids=reference_ids, visibility=visibility,
            capture_layers=layers, capture=translated)
        return text_states, context_valid

    def _shared_context(self, batch: Dict[str, Any]
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Text (+ packed proprio) context and validity from the shared head."""
        return self.vla_head._context(batch)

    def _prepare_batch(self, images, task_description, target, kwargs):
        if self.training_mode == 'lit':
            # The LIT branch never consumes ground-truth future targets, so the
            # backbone stays on the current-observation path (this is what
            # makes a current-frame-only Stage-2 lit run work through the real
            # encoder).
            target = False
        return super()._prepare_batch(images, task_description, target, kwargs)

    @staticmethod
    def _batch_size(states: Optional[torch.Tensor],
                    actions: Optional[torch.Tensor]) -> int:
        for candidate in (actions, states):
            if candidate is not None:
                return int(candidate.shape[0])
        raise ValueError('Stage 1 requires `states` (or `actions`) for batch size')

    def _stage1_inputs(self, batch_size: int,
                       kwargs: Dict[str, Any]) -> None:
        """Force the Stage-1 batch to be image-free.

        Any caller-provided imagery, history or frame/camera mask is dropped
        rather than retained, so a Stage-1 step can never encode a camera.
        """
        for key in _STAGE1_DROPPED_KEYS:
            kwargs.pop(key, None)
        device = None
        for candidate in (kwargs.get('states'), kwargs.get('text_tokens')):
            if isinstance(candidate, torch.Tensor):
                device = candidate.device
                break
        token_dim = self.vla_head.codec.token_dim
        kwargs['reference_tokens'] = torch.zeros(
            batch_size, 0, token_dim, device=device, dtype=torch.float32)
        kwargs['reference_ids'] = sequence_coordinates(
            batch_size, 0, group=10, device=device)
        kwargs['reference_valid'] = torch.ones(
            batch_size, 0, dtype=torch.bool, device=device)

    def _require_tokens(self, batch: Dict[str, Any], target: bool) -> None:
        """LIT-only training never consumes ground-truth future tokens."""
        if self.training_mode != 'lit':
            super()._require_tokens(batch, target=target)
            return
        required = ['text_tokens']
        if self.lit_stage == 'stage2':
            required.append('reference_tokens')
        missing = [key for key in required if key not in batch]
        if missing and self.vlm_backbone is None:
            raise ValueError(
                'PatchWAMLITVLA without a vlm_backbone requires pre-encoded '
                f'tensors; missing {missing}.')
        if missing:
            raise ValueError(f'Encoded batch is missing {missing}.')

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        action_dim_masks: Optional[torch.Tensor] = None,
        frame_masks: Optional[torch.Tensor] = None,
        img_masks: Optional[torch.Tensor] = None,
        task_description: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        generator = kwargs.get('generator')
        stage1 = self.lit_stage == 'stage1'
        if stage1:
            # Stage 1 is image-free end to end: the autoencoder never sees the
            # world cameras, so no image gradient or feature can enter the path.
            images = None
            self._stage1_inputs(self._batch_size(states, actions), kwargs)
        batch = self._build_forward_batch(
            images=images,
            states=states,
            actions=actions,
            action_masks=action_masks,
            action_dim_masks=action_dim_masks,
            frame_masks=frame_masks,
            img_masks=img_masks,
            task_description=task_description,
            kwargs=kwargs,
        )
        for key in _LIT_BATCH_KEYS:
            if key in kwargs and kwargs[key] is not None:
                batch[key] = kwargs[key]
        original = None
        if self.training_mode in ('patchwam', 'joint'):
            original = self._run_head(batch, kwargs)
        if self.training_mode == 'patchwam':
            return original
        lit = self._lit_loss(batch, generator=generator)
        if original is None:
            return lit
        total = (self.patchwam_loss_weight * original['loss']
                 + self.lit_loss_weight * lit['loss'])
        merged: Dict[str, torch.Tensor] = dict(original)
        merged.update({f'lit_{key}': value for key, value in lit.items()})
        merged['loss_patchwam'] = original['loss'].detach()
        merged['loss_lit'] = lit['loss'].detach()
        merged['loss'] = total
        return merged

    def _lit_loss(self, batch: Dict[str, Any],
                  generator: Optional[torch.Generator] = None
                  ) -> Dict[str, torch.Tensor]:
        context, context_valid = self._shared_context(batch)
        if self.lit_stage == 'stage1':
            goal_state = batch.get('goal_state')
            if goal_state is None:
                raise ValueError(
                    'Stage 1 requires the oracle `goal_state` input '
                    '(normalized terminal state at t + horizon)')
            prefix_tokens, prefix_mask = self.lit_head.stage1_prefix(
                context, context_valid, goal_state, batch.get('goal_valid'))
            prefix_hidden, prefix_mask = self._run_lit_prefix(
                context=prefix_tokens, context_valid=prefix_mask)
            return self.lit_head.stage1_loss(prefix_hidden, prefix_mask, batch,
                                             generator=generator)
        reference = batch.get('reference_tokens')
        if reference is None or reference.shape[1] == 0:
            raise ValueError(
                'Stage 2 LIT requires `reference_tokens` from the current '
                'observation prefix')
        reference = reference.to(device=context.device, dtype=context.dtype)
        reference_valid = batch.get('reference_valid')
        reference_ids = batch.get('reference_ids')

        def runner(capture):
            return self._run_lit_prefix(
                context=context, context_valid=context_valid,
                reference=reference, reference_valid=reference_valid,
                reference_ids=reference_ids,
                capture_layers=self.lit_head.coupling_layers, capture=capture)

        contexts, final_latents = self.lit_head.stage2_contexts(
            runner, batch_size=context.shape[0], device=context.device,
            semantic_valid=context_valid, image_valid=reference_valid,
            generator=generator, training=self.training)
        return self.lit_head.stage2_loss(
            contexts, final_latents, batch, generator=generator)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action(
        self,
        images: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        task_description: Any = None,
        action_horizon: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        head: Optional[str] = None,
        oracle_goal: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        head = head or self.active_head
        if head not in _ACTIVE_HEADS:
            raise ValueError(f'head must be one of {_ACTIVE_HEADS}, got {head!r}')
        if head == 'patchwam':
            return super().predict_action(
                images=images, states=states, task_description=task_description,
                action_horizon=action_horizon,
                num_inference_steps=num_inference_steps, seed=seed, **kwargs)
        horizon = (action_horizon if action_horizon is not None
                   else self.action_horizon)
        if horizon is None:
            raise ValueError('`action_horizon` must be provided or configured')
        steps = (self.num_inference_steps if num_inference_steps is None
                 else int(num_inference_steps))
        stage1 = self.lit_stage == 'stage1'
        goal_state = kwargs.pop('goal_state', None)
        goal_valid = kwargs.pop('goal_valid', None)
        if not stage1:
            if goal_state is not None or oracle_goal:
                raise ValueError(
                    'Stage-2 LIT inference is current-observation only and '
                    'rejects oracle goal/future inputs')
            if kwargs.get('future_tokens') is not None:
                raise ValueError(
                    'Stage-2 LIT inference rejects future-token inputs')
        if stage1 and goal_state is None:
            raise ValueError(
                'Stage-1 LIT inference is an explicit oracle-goal diagnostic; '
                'pass `goal_state=` (and `oracle_goal=True`)')
        if stage1 and not oracle_goal:
            raise ValueError(
                'Stage-1 LIT inference needs `oracle_goal=True` to acknowledge '
                'the oracle terminal-goal input')
        if stage1:
            images = None
            self._stage1_inputs(self._batch_size(states, None), kwargs)
        batch = self._build_forward_batch(
            images=images, states=states, actions=None, action_masks=None,
            action_dim_masks=None, frame_masks=None, img_masks=None,
            task_description=task_description, kwargs=kwargs)
        if goal_state is not None:
            batch['goal_state'] = goal_state
        if goal_valid is not None:
            batch['goal_valid'] = goal_valid
        context, context_valid = self._shared_context(batch)
        reference_tokens = batch.get('reference_tokens', None)
        generator = kwargs.get('generator')
        if generator is None and seed is not None:
            device = context.device
            generator = torch.Generator(device=device).manual_seed(int(seed))
        if stage1:
            prefix_tokens, prefix_mask = self.lit_head.stage1_prefix(
                context, context_valid, batch['goal_state'],
                batch.get('goal_valid'))
            prefix_hidden, prefix_mask = self._run_lit_prefix(
                context=prefix_tokens, context_valid=prefix_mask)
            return self.lit_head.sample_actions(
                prefix_hidden, prefix_mask, horizon=int(horizon),
                steps=int(steps), generator=generator,
                initial_action=kwargs.get('initial_action'))
        if reference_tokens is None or reference_tokens.shape[1] == 0:
            raise ValueError(
                'Stage-2 LIT inference requires the current observation prefix')

        def runner(capture):
            return self._run_lit_prefix(
                context=context, context_valid=context_valid,
                reference=reference_tokens,
                reference_valid=batch.get('reference_valid'),
                reference_ids=batch.get('reference_ids'),
                capture_layers=self.lit_head.coupling_layers, capture=capture)

        contexts, _ = self.lit_head.stage2_contexts(
            runner, batch_size=context.shape[0], device=context.device,
            semantic_valid=context_valid,
            image_valid=batch.get('reference_valid'),
            generator=generator, training=False)
        return self.lit_head.sample_actions(
            contexts=contexts, horizon=int(horizon), steps=int(steps),
            generator=generator, initial_action=kwargs.get('initial_action'))

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_original_payload(state: Dict[str, Any]) -> str:
        """native / compact / bare, matching the native PatchWAM convention."""
        keys = tuple(state)
        if any(key.startswith(('vla_head.', 'vlm_backbone.')) for key in keys):
            return 'native'
        if any(key.startswith('policy.') for key in keys):
            return 'compact'
        return 'bare'

    def required_original_keys(self) -> set:
        """Original/shared tensors a load must provide.

        Independent of ``requires_grad``: a Stage-1 model freezes the whole
        original branch, and a frozen parameter still has to be present in a
        native dual-head payload. The frozen AE/Qwen base assets are loaded from
        local files rather than checkpoints, so they are optional in every
        format; trainable Qwen LoRA adapters are required when configured.
        """
        encoder = getattr(self.vlm_backbone, 'text_encoder', None)
        adapters = bool(getattr(encoder, 'trainable_adapters', False))
        required = set()
        for key in self.state_dict():
            if key.startswith('lit_head.'):
                continue
            if key.startswith('vlm_backbone.autoencoder.'):
                continue
            if key.startswith('vlm_backbone.text_encoder.'):
                if adapters and ('residual_in' in key or 'residual_out' in key):
                    required.add(key)
                continue
            required.add(key)
        return required

    def plan_original_load(self, state: Dict[str, Any],
                           *, strict: bool = True) -> Dict[str, Any]:
        """Resolve and validate the original/shared half without mutating it."""
        released = released_action_patch_state(state)
        if released is not None:
            available, format_name = released, 'action_patch'
        else:
            available = dict(state)
            format_name = self._classify_original_payload(available)
        current = self.state_dict()
        model_keys = set(current)
        resolved: Dict[str, str] = {}
        unexpected: List[str] = []
        for name in available:
            if format_name in ('native', 'action_patch'):
                target = name
            elif format_name == 'compact':
                if name.startswith('policy.'):
                    target = 'vla_head.' + name[len('policy.'):]
                elif name.startswith('text_encoder.'):
                    target = ('vlm_backbone.text_encoder.'
                              + name[len('text_encoder.'):])
                else:
                    target = None
            else:  # bare
                target = 'vla_head.' + name
            if (target is None or target not in model_keys
                    or target.startswith('lit_head.')):
                unexpected.append(name)
                continue
            resolved[target] = name
        missing = sorted(self.required_original_keys() - set(resolved))
        if strict and (missing or unexpected):
            raise ValueError(
                f'Strict PatchWAM checkpoint load failed (format='
                f'{format_name}): missing={missing[:8]}'
                f'{"..." if len(missing) > 8 else ""}, '
                f'unexpected={sorted(unexpected)[:8]}'
                f'{"..." if len(unexpected) > 8 else ""}')
        plan = []
        values: Dict[str, torch.Tensor] = {}
        for target, source in resolved.items():
            value = available[source]
            if not torch.is_tensor(value):
                raise TypeError(f'Checkpoint entry {source} is not a tensor')
            target_tensor = current[target]
            if value.shape != target_tensor.shape:
                raise ValueError(
                    f'Weight shape differs for {target}: '
                    f'{tuple(value.shape)} vs {tuple(target_tensor.shape)}')
            if (target_tensor.is_floating_point()
                    and not torch.isfinite(
                        value.to(target_tensor.dtype)).all()):
                raise ValueError(f'Weight values are invalid for {target}')
            plan.append((target, target_tensor, value))
            values[target] = value
        return {
            'format': format_name,
            'plan': plan,
            'values': values,
            'missing': missing,
            'unexpected': sorted(unexpected),
            'required_tensors': len(self.required_original_keys()),
            'resolved_tensors': len(resolved),
        }

    def plan_lit_load(self, state: Dict[str, Any],
                      *, bridge_from_stage1: bool = False) -> Dict[str, Any]:
        """Validate the LIT half without mutating it."""
        prefix = 'lit_head.'
        lit_state = {key: value for key, value in state.items()
                     if key.startswith(prefix)}
        if not lit_state:
            raise ValueError('The payload contains no `lit_head.` tensors')
        current = self.lit_head.state_dict()
        model_keys = {prefix + key for key in current}
        missing = sorted(model_keys - set(lit_state))
        unexpected = sorted(set(lit_state) - model_keys)
        validate_lit_checkpoint_keys(
            stage=self.lit_stage, missing_keys=missing,
            unexpected_keys=unexpected,
            bridge_from_stage1=bool(bridge_from_stage1))
        plan = []
        for key, value in lit_state.items():
            if key not in model_keys:
                continue
            if not torch.is_tensor(value):
                raise TypeError(f'LIT checkpoint entry {key} is not a tensor')
            target_tensor = current[key[len(prefix):]]
            if value.shape != target_tensor.shape:
                raise ValueError(
                    f'LIT weight shape differs for {key}: '
                    f'{tuple(value.shape)} vs {tuple(target_tensor.shape)}')
            if (target_tensor.is_floating_point()
                    and not torch.isfinite(
                        value.to(target_tensor.dtype)).all()):
                raise ValueError(f'LIT weight values are invalid for {key}')
            plan.append((key, target_tensor, value))
        return {
            'stage': self.lit_stage,
            'bridge_from_stage1': bool(bridge_from_stage1),
            'plan': plan,
            'loaded_tensors': len(plan),
            'missing': missing,
            'unexpected': unexpected,
        }

    @staticmethod
    def _apply_plan(plan) -> None:
        with torch.no_grad():
            for _, target_tensor, value in plan:
                target_tensor.copy_(value.to(target_tensor.dtype))

    def plan_state_dict_payload(
        self,
        state: Dict[str, Any],
        *,
        bridge_from_stage1: bool = False,
        strict: bool = True,
        require_original: bool = True,
    ) -> Dict[str, Any]:
        """Validate both halves of a payload before any tensor is written.

        A native dual-head restore requires the complete original/shared half:
        carrying LIT tensors while omitting the FLUX transformer or proprio
        projection would silently leave a fine-tuned head attached to base or
        random shared weights. The head-only API for a LIT payload is
        :meth:`load_lit_state_dict`.
        """
        lit_payload = {key: value for key, value in state.items()
                       if key.startswith('lit_head.')}
        original_payload = {key: value for key, value in state.items()
                            if not key.startswith('lit_head.')}
        if not lit_payload and not original_payload:
            raise ValueError('The payload contains no loadable tensors')
        if require_original and not original_payload:
            raise ValueError(
                'A full native restore requires the complete original/shared '
                'half; the payload only carries `lit_head.*` tensors. Use '
                'load_lit_state_dict for a head-only LIT load.')
        plans = []
        report: Dict[str, Any] = {'lit_head': None, 'patchwam': None}
        original_values: Dict[str, torch.Tensor] = {}
        if original_payload:
            original_plan = self.plan_original_load(original_payload,
                                                    strict=strict)
            original_values = original_plan.get('values', {})
            original_plan.pop('values', None)
            plans.append(original_plan.pop('plan'))
            report['patchwam'] = original_plan
        if lit_payload:
            lit_plan = self.plan_lit_load(
                lit_payload, bridge_from_stage1=bridge_from_stage1)
            plans.append(lit_plan.pop('plan'))
            report['lit_head'] = lit_plan
        return {'plans': plans, 'report': report,
                'original_values': original_values}

    def load_patchwam_state_dict(self, state: Dict[str, torch.Tensor],
                                 *, strict: bool = True) -> Dict[str, Any]:
        """Initialize original/shared PatchWAM weights; LIT stays random."""
        lit_keys = sorted(key for key in state if key.startswith('lit_head.'))
        if lit_keys:
            raise ValueError(
                'LIT weights must be loaded through load_lit_state_dict; '
                f'found {lit_keys[:8]}')
        plan = self.plan_original_load(state, strict=strict)
        report = {key: value for key, value in plan.items() if key != 'plan'}
        report['lit_head_initialized'] = 'random'
        report['lit_head_tensors'] = len(
            [key for key in self.state_dict() if key.startswith('lit_head.')])
        self._apply_plan(plan['plan'])
        return report

    def load_lit_state_dict(
        self,
        state: Dict[str, torch.Tensor],
        *,
        bridge_from_stage1: bool = False,
    ) -> Dict[str, Any]:
        """Strictly load LIT tensors, validating every key before any write."""
        plan = self.plan_lit_load(state,
                                  bridge_from_stage1=bridge_from_stage1)
        report = {key: value for key, value in plan.items() if key != 'plan'}
        self._apply_plan(plan['plan'])
        return report

    @staticmethod
    def _shared_tensors_match(planned: torch.Tensor,
                              staged: torch.Tensor) -> bool:
        """Compare shared tensors after normal bfloat16 storage rounding.

        Stage 1 stores its frozen original half in the model dtype (bfloat16 in
        the real recipe), while the released checkpoint may be float32. Rounding
        both sides to bfloat16 accepts that storage rounding but still rejects a
        real value change (for example an edited proprio projection).

        Non-floating tensors (integer/bool buffers) keep exact equality and are
        never quantized through a float cast.
        """
        if planned.shape != staged.shape:
            return False
        # Checkpoints are read on CPU, while local AE/Qwen assets may already
        # live on CUDA. Compare on CPU without allocating float copies on GPU.
        left = planned.detach().cpu()
        right = staged.detach().cpu()
        if left.is_floating_point() and right.is_floating_point():
            return torch.equal(left.float().to(torch.bfloat16),
                               right.float().to(torch.bfloat16))
        if left.dtype == right.dtype:
            return torch.equal(left, right)
        if left.is_complex() or right.is_complex():
            return torch.equal(left, right)
        return torch.equal(left.to(torch.int64), right.to(torch.int64))

    def plan_lit_bridge_load(
        self,
        state: Dict[str, Any],
        *,
        primary_original_values: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Validate a Stage-1 payload against the planned primary original half.

        The bridge must be a real Stage-1 checkpoint (goal encoder present) and
        its mandatory original/shared tensors must agree with the released
        weights the Stage-2 model is about to load, so the LIT prior cannot be
        conditioned on a different backbone. Only ``lit_head.*`` tensors are
        ever copied.

        Native DDP exports every ``state_dict`` tensor, including the frozen
        AE/Qwen base assets that a released PatchWAM checkpoint omits, so those
        optional asset tensors are checked against this model's locally loaded
        asset tensors (or against the primary when the primary carries them)
        instead of being demanded from the released primary.
        """
        if not any(key.startswith('lit_head.goal_pose_encoder.')
                   for key in state):
            raise ValueError(
                'The Stage-1 bridge payload has no '
                '`lit_head.goal_pose_encoder.*` tensors; it is not a Stage-1 '
                'LIT checkpoint (bridge flags alone do not make one).')
        lit_plan = self.plan_lit_load(state, bridge_from_stage1=True)
        original_payload = {key: value for key, value in state.items()
                            if not key.startswith('lit_head.')}
        if not original_payload:
            raise ValueError(
                'The Stage-1 bridge payload carries no original/shared '
                'tensors; native Stage-1 checkpoints are full model payloads.')
        staged_original = self.plan_original_load(original_payload,
                                                  strict=True)
        required = self.required_original_keys()
        current = self.state_dict()
        mismatched = []
        validated_optional = 0
        for key, _, staged_value in staged_original['plan']:
            reference = primary_original_values.get(key)
            if reference is None:
                if key in required:
                    # The planned primary is complete for mandatory keys, so a
                    # missing reference here is an internal inconsistency.
                    mismatched.append(key)
                    continue
                # Optional frozen base asset: compare against the tensors this
                # model will actually use instead of requiring them in the
                # released primary.
                reference = current.get(key)
                if reference is None:
                    raise ValueError(
                        f'The Stage-1 bridge carries {key} but this model has '
                        'no such tensor to compare it against.')
                validated_optional += 1
            if not self._shared_tensors_match(reference, staged_value):
                mismatched.append(key)
        if mismatched:
            raise ValueError(
                'The Stage-1 bridge original/shared weights disagree with the '
                f'primary checkpoint for {len(mismatched)} tensor(s): '
                f'{sorted(mismatched)[:8]}'
                f'{"..." if len(mismatched) > 8 else ""}. Stage 1 must start '
                'from the same released PatchWAM checkpoint as Stage 2.')
        lit_plan['validated_shared_tensors'] = len(
            [key for key, _, _ in staged_original['plan'] if key in required])
        lit_plan['validated_optional_assets'] = validated_optional
        return lit_plan

    def load_state_dict_payload(
        self,
        state: Dict[str, torch.Tensor],
        *,
        bridge_from_stage1: bool = False,
        strict: bool = True,
        require_original: bool = True,
    ) -> Dict[str, Any]:
        """Validate both halves, then apply. Refuses before mutating anything."""
        plan = self.plan_state_dict_payload(
            state, bridge_from_stage1=bridge_from_stage1, strict=strict,
            require_original=require_original)
        for entries in plan['plans']:
            self._apply_plan(entries)
        report = plan['report']
        if report['patchwam'] is not None and report['lit_head'] is None:
            report['patchwam']['lit_head_initialized'] = 'random'
        return report

    def from_pretrained(self) -> None:
        """Native initialization: released PatchWAM plus the Stage-1 bridge.

        The native DDP/FSDP runners call this method. A Stage-2 recipe must use
        the trained Stage-1 LIT weights instead of silently training a random
        action prior: when the primary checkpoint carries no LIT tensors and no
        Stage-1 checkpoint is configured, ``lit_stage1_required`` raises (the
        random-init ablation is an explicit opt-out). Evaluation loads a
        Stage-2 checkpoint that already contains both halves, so it never needs
        the Stage-1 file.

        Both payloads are read and validated *before* any tensor is written, and
        a Stage-1 bridge must agree with the primary original/shared weights, so
        a rejected load leaves the model untouched.
        """
        primary: Optional[Dict[str, Any]] = None
        primary_plan: Optional[Dict[str, Any]] = None
        if self.pretrained_name_or_path is not None:
            primary = self._read_checkpoint_state(self.pretrained_name_or_path)
            self._validate_pretrained_skip_prefixes()
            primary_plan = self.plan_state_dict_payload(primary)
        primary_has_lit = bool(
            primary is not None
            and any(key.startswith('lit_head.') for key in primary))
        bridge_plan: Optional[Dict[str, Any]] = None
        if self.lit_stage1_checkpoint:
            if primary_has_lit:
                raise ValueError(
                    'Both the primary checkpoint and `lit_stage1_checkpoint` '
                    'carry LIT weights; bridging would silently overwrite the '
                    'trained LIT interface. Unset `lit_stage1_checkpoint` '
                    '(PATCHWAM_LIT_STAGE1_CHECKPOINT) for a resume/eval load.')
            if primary_plan is None:
                raise ValueError(
                    'A Stage-1 bridge requires the primary released PatchWAM '
                    'checkpoint (`pretrained_name_or_path`) so the shared '
                    'original weights can be verified before the bridge is '
                    'copied.')
            payload = self._read_checkpoint_state(self.lit_stage1_checkpoint)
            bridge_plan = self.plan_lit_bridge_load(
                payload, primary_original_values=primary_plan[
                    'original_values'])
        if (bridge_plan is None and self.lit_stage == 'stage2'
                and not primary_has_lit and self.lit_stage1_required):
            raise ValueError(
                'Stage 2 requires the trained Stage-1 LIT checkpoint to '
                'initialize the LIT interface. Set `lit_stage1_checkpoint` '
                '(or PATCHWAM_LIT_STAGE1_CHECKPOINT), or opt into the '
                'random-initialization ablation explicitly with '
                '`lit_stage1_required=False`.')
        # Everything validated: now mutate.
        if primary_plan is not None:
            for entries in primary_plan['plans']:
                self._apply_plan(entries)
            report = primary_plan['report']
            if report['patchwam'] is not None and report['lit_head'] is None:
                report['patchwam']['lit_head_initialized'] = 'random'
            overwatch.info(f'PatchWAMLIT checkpoint loaded: {report}')
        if bridge_plan is not None:
            self._apply_plan(bridge_plan['plan'])
            bridge_report = {key: value for key, value in bridge_plan.items()
                             if key != 'plan'}
            overwatch.info(
                f'Stage-1 LIT bridge loaded from '
                f'{self.lit_stage1_checkpoint}: {bridge_report}')

    # ------------------------------------------------------------------
    # BaseVLA integration
    # ------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Wrap the outer VLA unit only.

        The LIT prefix and conditioning helpers call borrowed
        ``vla_head``/``lit_head`` methods outside any per-head ``forward``, and
        the official FLUX blocks execute through adapter helpers instead of
        ``block.forward``. Wrapping those heads or blocks would bypass FSDP
        all-gather/unshard hooks, so this initial integration wraps the outer
        VLA only (the native runner uses ``use_orig_params=True``). The native
        default stays DDP; multi-GPU FSDP is not validated here.
        """
        from fluxvla.engines.utils.fsdp_wrapping import build_module_wrap_policy
        return build_module_wrap_policy({PatchWAMLITVLA})

    def _validate_backbone_contract(self) -> None:
        super()._validate_backbone_contract()
        if not hasattr(self, 'lit_head') or self.vlm_backbone is None:
            return
        text_dim = getattr(self.vlm_backbone, 'text_dim', None)
        if text_dim is not None and text_dim != self.lit_head.prefix_input_dim:
            raise ValueError(
                f'backbone text_dim={text_dim} does not match '
                f'lit_head.prefix_input_dim={self.lit_head.prefix_input_dim}')
