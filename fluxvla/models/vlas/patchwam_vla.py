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

import os
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from fluxvla.engines import VLAS, initialize_overwatch
from ..heads.patchwam_head import PatchWAMHead  # noqa: F401  (registration)
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)

__all__ = ['DEFAULT_PATCHWAM_NAME_MAPPING', 'DEFAULT_PATCHWAM_PROMPT_TEMPLATE',
           'PatchWAMVLA', 'RELEASED_ACTION_PATCH_TRANSFORMER_PREFIX',
           'resolve_action_padding', 'released_action_patch_state']

#: Model key prefix owned by the PatchWAM head (denoiser, codec, state map).
HEAD_PREFIX = 'vla_head.'

#: Default instruction template the source PatchWAM data pipeline applies
#: before the instruction encoder (``src/patchwam/data/episodes.py`` and
#: ``evaluation/policy.py``). The native VLA applies the same fallback once,
#: centrally, so a raw ``task_description`` reaches the encoder wrapped exactly
#: as the source checkpoint expects.
DEFAULT_PATCHWAM_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following "
    'instruction: {task}')

#: Text shared by every templated instruction; detects a caller that already
#: formatted its instructions so the template is never applied twice.
_DEFAULT_PATCHWAM_PROMPT_PREFIX = DEFAULT_PATCHWAM_PROMPT_TEMPLATE.split(
    '{task}', 1)[0]

# Default FluxVLA -> PatchWAM key-prefix mapping (model key -> source checkpoint
# key, as BaseVLA expects) for initializing the native split model from a source
# ``FluxAssetPolicy`` checkpoint. Applied left to right; the prefixes are
# disjoint so order does not change the result.
DEFAULT_PATCHWAM_NAME_MAPPING = {
    'vlm_backbone.autoencoder.': 'autoencoder.',
    'vlm_backbone.text_encoder.': 'text_encoder.',
    'vla_head.': 'policy.',
}

# Released ``QZWang/patchwam-robotwin-c2r-step140000`` (PatchWAM Action-as-Patch)
# layout. The payload stores the fine-tuned FLUX.2 transformer under
# ``mot.mixtures.video.transformer.*`` and the proprio projection under
# ``proprio_encoder.*``. ``mixtures.video.{double,single}_blocks`` are module
# aliases of the same transformer blocks and carry no extra parameters.
RELEASED_ACTION_PATCH_TRANSFORMER_PREFIX = 'mot.mixtures.video.transformer.'
RELEASED_ACTION_PATCH_PROPRIO_PREFIX = 'proprio_encoder.'

# Pre-encoded tensors passed straight through to the vendored policy.
_TOKEN_KEYS = (
    'reference_tokens',
    'future_tokens',
    'radio_future_tokens',
    'text_tokens',
    'text_valid',
    'reference_valid',
    'future_valid',
    'reference_ids',
    'future_ids',
    'proprio',
    'current_raster_layout',
    'future_raster_layout',
    'current_raster_length',
)

# Arguments the backbone ``encode(batch, target)`` (or a cached batch) consume.
_ENCODE_KEYS = _TOKEN_KEYS + (
    'images',
    'video',
    'camera_video',
    'history_video',
    'history_valid',
    'vl_current',
    'vl_history',
    'vl_history_valid',
    'prompt',
    'subtask',
    'text_cache_format',
    'frame_masks',
    'img_masks',
)

# Forward knobs that belong to the head, not the encoder.
_HEAD_KNOB_KEYS = (
    'generator',
    'sigma',
    'future_noise',
    'action_noise',
    'condition_drop_mask',
    'second_sigma',
    'timestep_mask',
    'pseudo_label_mask',
    'structured_branches',
    'token_sigma',
)

# Training-only metadata consumed by the optional action/FK objectives. These
# bypass the backbone encoder and are passed to the policy unchanged.
_POLICY_METADATA_KEYS = (
    'active_arm',
    'fk_action_mean',
    'fk_action_std',
    'fk_action_epsilon',
)

_DEFERRED_PREFIXES = ('self_flow', 'ema')


def _format_task_prompts(task_description: Any) -> Any:
    """Return encoder prompts for one or more raw task instructions.

    A single instruction string becomes a one-element prompt list instead of
    being iterated per character; a list or tuple is formatted element-wise.
    Instructions that already start with the default source prefix pass through
    unchanged, so preparing an already formatted batch cannot wrap the template
    twice. Unexpected input types keep the previous pass-through behavior.
    """
    if isinstance(task_description, str):
        instructions = [task_description]
    elif isinstance(task_description, (list, tuple)):
        instructions = list(task_description)
    else:
        return task_description
    return [
        instruction if not isinstance(instruction, str) or
        instruction.startswith(_DEFAULT_PATCHWAM_PROMPT_PREFIX) else
        DEFAULT_PATCHWAM_PROMPT_TEMPLATE.format(task=instruction)
        for instruction in instructions
    ]


def resolve_action_padding(
    actions: torch.Tensor,
    action_masks: Optional[torch.Tensor] = None,
    action_dim_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert native ``True``-means-valid masks to source padding masks.

    The vendored policy consumes ``action_is_pad`` ``[B, H]`` and
    ``action_dim_is_pad`` ``[B, A]`` or ``[B, H, A]`` where ``True`` marks an
    invalid coordinate. This adapter preserves time and dimension validity
    separately:

    * ``action_masks`` ``[B, H]`` sets step padding only.
    * ``action_masks`` ``[B, H, A]`` sets exact per-coordinate padding, plus a
      step is time-invalid when every one of its dimensions is padded.
    * ``action_dim_masks`` ``[B, A]`` or ``[B, H, A]`` marks dimension padding
      and is OR-combined with any coordinate padding already present.

    Fully padded steps/dimensions stay zeroed inside the policy, so all-invalid
    examples produce finite losses without supervising padded coordinates.
    """
    if not torch.is_tensor(actions) or actions.ndim != 3:
        shape = tuple(actions.shape) if torch.is_tensor(actions) else None
        raise ValueError(f'`actions` must be [B, H, A], got shape {shape}')
    batch, horizon, action_dim = actions.shape
    device = actions.device
    step_pad = torch.zeros(batch, horizon, dtype=torch.bool, device=device)
    dim_pad: Optional[torch.Tensor] = None

    if action_masks is not None:
        mask = _as_bool_mask(action_masks, device)
        if mask.shape == (batch, horizon):
            step_pad = ~mask
        elif mask.shape == (batch, horizon, action_dim):
            step_pad = ~mask.any(dim=-1)
            dim_pad = ~mask
        else:
            raise ValueError(
                '`action_masks` must be [B, H] or [B, H, A], got '
                f'{tuple(mask.shape)}')

    if action_dim_masks is not None:
        dims = _as_bool_mask(action_dim_masks, device)
        if dims.shape == (batch, action_dim):
            extra = ~dims
        elif dims.shape == (batch, horizon, action_dim):
            extra = ~dims
        else:
            raise ValueError(
                '`action_dim_masks` must be [B, A] or [B, H, A], got '
                f'{tuple(dims.shape)}')
        if dim_pad is None:
            # No coordinate mask: keep the separable [B, A] form.
            dim_pad = extra
        else:
            # A coordinate mask makes ``dim_pad`` ``[B, H, A]``. Broadcast a
            # ``[B, A]`` dimension mask over the horizon axis explicitly; the
            # right-aligned broadcast would otherwise cross-wire batch and
            # horizon when ``B == H``.
            if extra.ndim == 2:
                extra = extra[:, None, :]
            dim_pad = dim_pad | extra

    if dim_pad is None:
        dim_pad = torch.zeros(batch, action_dim, dtype=torch.bool,
                              device=device)
    return step_pad, dim_pad


def _as_bool_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(mask):
        raise TypeError('Action masks must be torch tensors')
    mask = mask.to(device=device)
    return mask if mask.dtype == torch.bool else mask.bool()


def _apply_name_mapping(name: str, mapping: Dict[str, str]) -> str:
    for source, target in mapping.items():
        name = name.replace(source, target)
    return name


def _classify_checkpoint(state: Dict[str, torch.Tensor]) -> str:
    """Classify a checkpoint as native / compact / bare PatchWAM weights."""
    if not state:
        raise ValueError('Checkpoint is empty')
    keys = tuple(state)
    if any(key.startswith(('vla_head.', 'vlm_backbone.')) for key in keys):
        return 'native'
    if any(key.startswith('policy.') for key in keys):
        return 'compact'
    return 'bare'


#: Adapter module path the removed private LoRA implementation attached to the
#: denoiser. Payloads that carry it are not loadable into a PEFT-adapted model.
LEGACY_PATCHWAM_LORA_MARKER = 'dit_lora.'


def _legacy_lora_keys(state: Dict[str, torch.Tensor]) -> list:
    return sorted(key for key in state
                  if LEGACY_PATCHWAM_LORA_MARKER in str(key))


def _peft_base_layer_target(target: str, model_keys) -> Optional[str]:
    """Map ``<module>.<param>`` onto its PEFT ``base_layer`` parameter key.

    ``inject_adapter_in_model`` moves the wrapped layer under
    ``<module>.base_layer``, so a checkpoint written before adaptation names
    the same tensor without that segment. Returns ``None`` when this model has
    no such key, leaving the caller's unexpected-key handling in charge.
    """
    module, separator, parameter = target.rpartition('.')
    if not separator:
        return None
    candidate = f'{module}.base_layer.{parameter}'
    return candidate if candidate in model_keys else None


def _iter_released_entries(state: Dict[str, Any]):
    """Split a payload into ``(mot_entries, proprio_entries)`` mappings.

    Accepts both the nested form (``{'mot': {...}, 'proprio_encoder': {...}}``)
    and the flattened ``mot./proprio_encoder.`` prefixed form. Non-tensor
    metadata (``step``/``torch_dtype``/...) is ignored. Returns ``None`` when
    the payload carries no Action-as-Patch marker at all; a payload that does
    carry one but also holds unexpected tensor entries is rejected instead of
    silently dropping them.
    """
    if not isinstance(state, dict):
        return None
    nested_mot = state.get('mot')
    if isinstance(nested_mot, dict):
        mot = dict(nested_mot)
        raw_proprio = state.get('proprio_encoder')
        if not isinstance(raw_proprio, dict):
            raise ValueError(
                'A released Action-as-Patch payload with a `mot` section must '
                'also carry a `proprio_encoder` section')
        for key, value in state.items():
            if key in ('mot', 'proprio_encoder'):
                continue
            if torch.is_tensor(value):
                raise ValueError(
                    'Unexpected tensor entry in the released Action-as-Patch '
                    f'payload: {key}')
        if not mot:
            raise ValueError('The released Action-as-Patch `mot` section is empty')
        return mot, dict(raw_proprio)
    if not any(str(key).startswith('mot.') for key in state):
        return None
    mot: Dict[str, Any] = {}
    proprio: Dict[str, Any] = {}
    for key, value in state.items():
        name = str(key)
        if name.startswith(RELEASED_ACTION_PATCH_TRANSFORMER_PREFIX):
            mot[name[len('mot.'):]] = value
        elif name.startswith('mot.mixtures.video.'):
            mot[name[len('mot.'):]] = value
        elif name.startswith(RELEASED_ACTION_PATCH_PROPRIO_PREFIX):
            proprio[name[len('proprio_encoder.'):]] = value
        elif torch.is_tensor(value):
            raise ValueError(
                'Unexpected tensor entry in the released Action-as-Patch '
                f'payload: {name}')
    if not mot or not proprio:
        raise ValueError(
            'The released Action-as-Patch payload is missing its transformer '
            'or proprio_encoder entries')
    return mot, proprio


def released_action_patch_state(state: Dict[str, Any]):
    """Map a released PatchWAM Action-as-Patch payload onto native model keys.

    The published RoboTwin C2R checkpoint stores the fine-tuned FLUX.2
    transformer under ``mot.mixtures.video.transformer.*`` and the 14-D proprio
    projection under ``proprio_encoder.*``. ``mixtures.video.{double,single}_blocks``
    are module aliases of those same blocks (shared storage and identical
    values) and add no parameters; they are checked and dropped.

    Returns ``{'vla_head.denoiser.transformer.*' / 'vla_head.state_projection.*':
    tensor}`` when the payload matches, otherwise ``None``.
    """
    entries = _iter_released_entries(state)
    if entries is None:
        return None
    mot, proprio = entries
    transformer: Dict[str, Any] = {}
    aliases: Dict[str, Any] = {}
    for key, value in mot.items():
        if key.startswith('mixtures.video.transformer.'):
            transformer[key[len('mixtures.video.transformer.'):]] = value
        elif key.startswith(('mixtures.video.double_blocks.',
                             'mixtures.video.single_blocks.')):
            aliases[key[len('mixtures.video.'):]] = value
        else:
            raise ValueError(
                'Unsupported key in the released Action-as-Patch mixture: '
                f'{key}')
    if not transformer:
        return None
    resolved: Dict[str, Any] = {}
    for key, tensor in transformer.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f'Released transformer entry {key} is not a tensor')
        resolved['vla_head.denoiser.transformer.' + key] = tensor
    for key, tensor in aliases.items():
        reference = transformer.get(key)
        if reference is None:
            raise ValueError(
                'Released Action-as-Patch block alias has no transformer '
                f'counterpart: {key}')
        if not torch.is_tensor(tensor):
            raise TypeError(f'Released block alias {key} is not a tensor')
        if tensor.shape != reference.shape:
            raise ValueError(
                f'Released block alias {key} shape {tuple(tensor.shape)} '
                f'differs from the transformer copy {tuple(reference.shape)}')
        if tensor.data_ptr() != reference.data_ptr():
            if not torch.equal(tensor, reference):
                raise ValueError(
                    'Released Action-as-Patch block alias differs from its '
                    f'transformer copy: {key}')
    for key, tensor in proprio.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f'Released proprio entry {key} is not a tensor')
        resolved['vla_head.state_projection.' + key] = tensor
    return resolved


@VLAS.register_module()
class PatchWAMVLA(BaseVLA):
    """Native FluxVLA wrapper around the vendored PatchWAM joint flow model.

    ``vlm_backbone`` (``PatchWAMBackbone``) owns the frozen autoencoder and the
    instruction encoder and produces ``reference_tokens`` / ``future_tokens`` /
    ``text_tokens`` / validity masks. ``vla_head`` (``PatchWAMHead``) owns the
    denoiser, proprio projection, codec, and flow objective. The two hold
    disjoint parameters and are the only ``all_module_keys``.

    Training follows the shared FluxVLA batch contract: ``images`` is
    ``[B, V, 3, T, H, W]`` (or separate cameras), ``states`` is normalized
    proprioception, ``actions`` is the normalized ``[B, H, A]`` target window,
    and ``action_masks`` / ``action_dim_masks`` / ``frame_masks`` / ``img_masks``
    use ``True`` for valid entries. When no backbone is configured the caller
    must pass the pre-encoded tensors directly (used by the CPU tests).

    Args:
        action_horizon: Default normalized action horizon for prediction.
        action_norm_type: Normalization applied by the data transform; the
            runner/decoder denormalizes once.
        num_inference_steps: Default number of Euler solver steps (ten).
    """

    #: Pre-encoded keys accepted without a backbone. Subclasses may extend
    #: these class attributes without adding a second batch-preparation path.
    TOKEN_KEYS = _TOKEN_KEYS
    ENCODE_KEYS = _ENCODE_KEYS

    def __init__(
        self,
        vlm_backbone: Optional[Dict] = None,
        vla_head: Optional[Dict] = None,
        action_horizon: Optional[int] = None,
        action_norm_type: str = 'min_max',
        num_inference_steps: int = 10,
        freeze_vlm_backbone: bool = True,
        pretrained_name_or_path: Optional[str] = None,
        visual_stage: Optional[str] = None,
        stage1_checkpoint: Optional[str] = None,
        name_mapping: Optional[Dict] = None,
        strict_mapping: bool = False,
        pretrained_skip_prefixes: Optional[List[str]] = None,
        vision_backbone: Optional[Dict] = None,
        llm_backbone: Optional[Dict] = None,
        projector: Optional[Dict] = None,
        enable_mixed_precision_training: bool = True,
        freeze_vision_backbone: bool = True,
        freeze_llm_backbone: bool = True,
        freeze_projector: bool = False,
        vision_backbone_fp32: bool = False,
        unfreeze_last_layer: bool = False,
        ignore_index: int = -100,
        norm_stats: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        deferred = sorted(
            key for key in kwargs if key.startswith(_DEFERRED_PREFIXES))
        if deferred:
            raise ValueError(
                'EMA/Self-Flow training options are deferred from the native '
                f'PatchWAM integration and cannot be configured: {deferred}')
        if kwargs:
            raise TypeError(
                f'Unexpected PatchWAMVLA arguments: {sorted(kwargs)}')
        super().__init__(
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            vlm_backbone=vlm_backbone,
            projector=projector,
            vla_head=vla_head,
            enable_mixed_precision_training=enable_mixed_precision_training,
            freeze_vision_backbone=freeze_vision_backbone,
            freeze_llm_backbone=freeze_llm_backbone,
            freeze_vlm_backbone=freeze_vlm_backbone,
            freeze_projector=freeze_projector,
            vision_backbone_fp32=vision_backbone_fp32,
            unfreeze_last_layer=unfreeze_last_layer,
            ignore_index=ignore_index,
            norm_stats=norm_stats,
            pretrained_name_or_path=pretrained_name_or_path,
            name_mapping=name_mapping,
            strict_mapping=strict_mapping,
            pretrained_skip_prefixes=pretrained_skip_prefixes,
        )
        if self.vla_head is None:
            raise ValueError('PatchWAMVLA requires a `vla_head` config.')
        if num_inference_steps <= 0:
            raise ValueError('num_inference_steps must be positive, got '
                             f'{num_inference_steps}.')
        if self.name_mapping is None:
            # Match source FluxAssetPolicy keys by default.
            self.name_mapping = dict(DEFAULT_PATCHWAM_NAME_MAPPING)
        self.action_horizon = (None if action_horizon is None else
                               int(action_horizon))
        self.num_inference_steps = int(num_inference_steps)
        self.action_norm_type = str(action_norm_type)
        self.all_module_keys = ['vlm_backbone', 'vla_head']
        # Instance mirror of the encode keys so subclasses can extend the
        # accepted request (the RADIO auxiliary variant adds `need_radio`).
        self._encode_keys = tuple(self.ENCODE_KEYS)
        head_stage = getattr(self.vla_head, 'stage', None)
        # True when the head runs the unstaged DiT PEFT adapter policy; the
        # mode is a head-owned policy that this model applies to the frontend.
        self.peft_finetune = bool(getattr(self.vla_head, 'peft_finetune',
                                          False))
        self._finetune_initialization: Optional[Dict[str, Any]] = None
        if self.peft_finetune and visual_stage is not None:
            raise ValueError(
                'An unstaged PatchWAM adapter finetune and a `visual_stage` are '
                'mutually exclusive: the staged recipes open the new visual '
                'projections and the adapters, the unstaged adapter policy keeps '
                'the released policy trainable beside the adapters')
        if self.peft_finetune and not freeze_vlm_backbone:
            raise ValueError(
                'An unstaged PatchWAM adapter finetune freezes the vision '
                'backbone and the instruction encoder; `freeze_vlm_backbone` '
                'must stay True')
        if visual_stage is not None:
            if head_stage is None:
                raise ValueError(
                    'visual_stage requires a matching `vla_head.stage`; the '
                    'recipe must configure both or neither')
            if head_stage != visual_stage:
                raise ValueError(
                    f'vla_head stage {head_stage!r} conflicts with '
                    f'visual_stage {visual_stage!r}')
        self.visual_stage = visual_stage
        self.stage1_checkpoint = (None if stage1_checkpoint in (None, '') else
                                  str(stage1_checkpoint))
        if self.visual_stage == 'stage2' and self.stage1_checkpoint is None:
            raise ValueError(
                'visual_stage=stage2 requires `stage1_checkpoint`; a stage 2 '
                'run must inherit a stage 1 checkpoint')
        self._validate_backbone_contract()

    def _validate_backbone_contract(self) -> None:
        """Ensure the backbone encoder widths match the head."""
        if self.vlm_backbone is None:
            return
        token_dim = getattr(self.vlm_backbone, 'token_dim', None)
        text_dim = getattr(self.vlm_backbone, 'text_dim', None)
        if token_dim is not None and token_dim != self.vla_head.codec.token_dim:
            raise ValueError(
                f'backbone token_dim={token_dim} does not match head '
                f'token_dim={self.vla_head.codec.token_dim}')
        if text_dim is not None and text_dim != self.vla_head.text_dim:
            raise ValueError(
                f'backbone text_dim={text_dim} does not match head '
                f'text_dim={self.vla_head.text_dim}')
        visual_dim = getattr(self.vlm_backbone, 'visual_dim', None)
        head_visual = getattr(self.vla_head, 'visual_dim', None)
        if (visual_dim is not None and head_visual is not None
                and int(visual_dim) != int(head_visual)):
            raise ValueError(
                f'backbone visual_dim={visual_dim} does not match head '
                f'visual_dim={head_visual}')

    # ------------------------------------------------------------------
    # Batch adaptation
    # ------------------------------------------------------------------
    def _prepare_batch(
        self,
        images: Optional[torch.Tensor],
        task_description: Any,
        target: bool,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the source-batch contract from native inputs."""
        batch: Dict[str, Any] = {}
        if self.vlm_backbone is not None:
            request = {
                key: kwargs[key]
                for key in self._encode_keys
                if key in kwargs and kwargs[key] is not None
            }
            if images is not None:
                request['images'] = images
            if task_description is not None:
                request['task_description'] = task_description
                if 'prompt' not in request:
                    # Apply the source instruction template centrally; a caller
                    # that passes an explicit `prompt` owns that prepared text.
                    request['prompt'] = _format_task_prompts(task_description)
            encoded = self.vlm_backbone.encode(request, target=target)
            if not isinstance(encoded, dict):
                raise TypeError('vlm_backbone.encode must return a dict, got '
                                f'{type(encoded)!r}')
            batch.update(encoded)
        else:
            for key in self.TOKEN_KEYS:
                value = kwargs.get(key)
                if value is not None:
                    batch[key] = value
        return batch

    def _require_tokens(self, batch: Dict[str, Any], target: bool) -> None:
        required = ['reference_tokens', 'text_tokens']
        if target:
            required.append('future_tokens')
        missing = [key for key in required if key not in batch]
        if missing and self.vlm_backbone is None:
            raise ValueError(
                'PatchWAMVLA without a vlm_backbone requires pre-encoded '
                f'tensors; missing {missing}.')
        if missing:
            raise ValueError(f'Encoded batch is missing {missing}.')

    def _build_forward_batch(
        self,
        images: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        action_masks: Optional[torch.Tensor],
        action_dim_masks: Optional[torch.Tensor],
        frame_masks: Optional[torch.Tensor],
        img_masks: Optional[torch.Tensor],
        task_description: Any,
        kwargs: Dict[str, Any],
        active_arm: Optional[torch.Tensor] = None,
        fk_action_mean: Optional[torch.Tensor] = None,
        fk_action_std: Optional[torch.Tensor] = None,
        fk_action_epsilon: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Build the shared source-batch contract from native inputs."""
        if frame_masks is not None:
            kwargs['frame_masks'] = frame_masks
        if img_masks is not None:
            kwargs['img_masks'] = img_masks
        batch = self._prepare_batch(
            images=images,
            task_description=task_description,
            target=actions is not None,
            kwargs=kwargs,
        )
        if states is not None:
            batch['proprio'] = states
        if actions is not None:
            step_pad, dim_pad = resolve_action_padding(actions, action_masks,
                                                       action_dim_masks)
            batch['action'] = actions
            batch['action_is_pad'] = step_pad
            batch['action_dim_is_pad'] = dim_pad
        for key, value in (
                ('active_arm', active_arm),
                ('fk_action_mean', fk_action_mean),
                ('fk_action_std', fk_action_std),
                ('fk_action_epsilon', fk_action_epsilon)):
            if value is not None:
                batch[key] = value
        for key in _POLICY_METADATA_KEYS:
            value = kwargs.get(key)
            if value is not None:
                batch[key] = value
        self._require_tokens(batch, target=actions is not None)
        return batch

    def _run_head(self, batch: Dict[str, Any],
                  kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run the head and fold the optional language auxiliary loss once."""
        head_kwargs = {
            key: kwargs[key] for key in _HEAD_KNOB_KEYS if key in kwargs
        }
        result = self.vla_head(batch, **head_kwargs)
        auxiliary = batch.get('language_auxiliary_loss')
        if auxiliary is not None and 'loss' in result:
            weight = _auxiliary_weight(self.vlm_backbone)
            result['loss_subtask'] = auxiliary
            result['loss'] = result['loss'] + weight * auxiliary
        return result

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
        active_arm: Optional[torch.Tensor] = None,
        fk_action_mean: Optional[torch.Tensor] = None,
        fk_action_std: Optional[torch.Tensor] = None,
        fk_action_epsilon: Optional[torch.Tensor] = None,
        task_description: Any = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Run the PatchWAM flow-matching loss.

        ``frame_masks`` (``[B, T]``, current first and future endpoint last) and
        ``img_masks`` (``[B, V]`` camera validity) map to reference/future raster
        validity inside the backbone; pre-encoded callers may pass
        ``reference_valid`` / ``future_valid`` directly and those are preserved.
        """
        batch = self._build_forward_batch(
            images=images,
            states=states,
            actions=actions,
            action_masks=action_masks,
            action_dim_masks=action_dim_masks,
            frame_masks=frame_masks,
            img_masks=img_masks,
            active_arm=active_arm,
            fk_action_mean=fk_action_mean,
            fk_action_std=fk_action_std,
            fk_action_epsilon=fk_action_epsilon,
            task_description=task_description,
            kwargs=kwargs,
        )
        return self._run_head(batch, kwargs)

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
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate normalized ``[B, H, A]`` actions.

        The existing runner or configured action decoder applies denormalization
        once, so this returns normalized actions. A generated future raster is
        always ``sampling_future_length`` tokens long; an appended historical
        prefix (``reference_tokens``) never changes it.
        """
        horizon = (action_horizon if action_horizon is not None else
                   self.action_horizon)
        if horizon is None:
            raise ValueError('`action_horizon` must be provided or configured '
                             'on the model via `action_horizon=`.')
        horizon = int(horizon)
        steps = (self.num_inference_steps if num_inference_steps is None else
                 int(num_inference_steps))
        if steps <= 0:
            raise ValueError('`num_inference_steps` must be positive, got '
                             f'{steps}.')
        batch = self._prepare_batch(
            images=images,
            task_description=task_description,
            target=False,
            kwargs=kwargs,
        )
        if states is not None:
            batch['proprio'] = states
        self._require_tokens(batch, target=False)

        future_length = kwargs.get('future_length')
        if future_length is None:
            future_length = batch.get('sampling_future_length')
        if future_length is None and 'future_ids' in batch:
            future_length = batch['future_ids'].shape[1]
        future_length = None if future_length is None else int(future_length)

        reference = batch['reference_tokens']
        generator = kwargs.get('generator')
        if generator is None and seed is not None:
            # Use the full device (including the CUDA index) so multi-GPU
            # sampling stays reproducible.
            generator = torch.Generator(device=reference.device).manual_seed(
                int(seed))
        result = self.vla_head.sample_actions(
            batch,
            horizon=horizon,
            steps=steps,
            generator=generator,
            future_length=future_length,
            initial_future=kwargs.get('initial_future'),
            initial_action=kwargs.get('initial_action'),
            guidance_scale=kwargs.get('guidance_scale', 1),
            action_guidance_scale=kwargs.get('action_guidance_scale'),
        )
        return result['action']

    # ------------------------------------------------------------------
    # Checkpoint initialization mapping
    # ------------------------------------------------------------------
    def load_patchwam_state_dict(self, state: Dict[str, torch.Tensor],
                                 *, strict: bool = True,
                                 allow_missing: Iterable[str] = ()
                                 ) -> Dict[str, torch.Tensor]:
        """Load a PatchWAM or native checkpoint into this split model.

        Three on-disk layouts are recognized, matching the source save helpers:

        * ``bare``: head-only, unprefixed keys (source ``policy_weight_state``
          with no LoRA adapters).
        * ``compact``: ``policy.*`` plus only the trainable ``text_encoder.*``
          LoRA adapter keys (source ``load_policy_weights`` with adapters).
        * ``native``: the FluxVLA ``vla_head.*`` / ``vlm_backbone.*`` state.

        Frozen autoencoder / Qwen base assets are never in a compact checkpoint
        and are loaded separately from config. Every trainable tensor must be
        present with a matching shape and no unexpected keys, all validated
        before any parameter is mutated.
        """
        # Only forward the staged allow-list when it is actually used, so
        # subclasses that override the resolver keep their existing signature.
        if allow_missing:
            available, format_name, resolved, missing, unexpected = (
                self._resolve_patchwam_state(state,
                                             allow_missing=allow_missing))
        else:
            available, format_name, resolved, missing, unexpected = (
                self._resolve_patchwam_state(state))
        if strict and (missing or unexpected):
            raise ValueError(
                f'Strict PatchWAM checkpoint load failed (format='
                f'{format_name}): missing={missing[:8]}'
                f'{"..." if len(missing) > 8 else ""}, '
                f'unexpected={sorted(unexpected)[:8]}'
                f'{"..." if len(unexpected) > 8 else ""}')
        self._apply_patchwam_state(available, resolved)
        return {'format': format_name, 'loaded_tensors': len(resolved)}

    def _apply_patchwam_state(self, available, resolved) -> None:
        """Validate every resolved tensor, then copy it into the model."""
        current = self.state_dict()
        for target, source in resolved.items():
            if available[source].shape != current[target].shape:
                raise ValueError(
                    f'Weight shape differs for {target}: '
                    f'{tuple(available[source].shape)} vs '
                    f'{tuple(current[target].shape)}')
            if (current[target].is_floating_point()
                    and not torch.isfinite(
                        available[source].to(current[target].dtype)).all()):
                raise ValueError(f'Weight values are invalid for {target}')
        with torch.no_grad():
            for target, source in resolved.items():
                current[target].copy_(available[source].to(current[target].dtype))

    def _resolve_patchwam_state(self, state: Dict[str, torch.Tensor],
                                allow_missing: Iterable[str] = ()):
        """Resolve checkpoint keys onto model keys without mutating weights."""
        legacy = _legacy_lora_keys(state)
        if legacy:
            raise ValueError(
                'This checkpoint carries the removed private LoRA adapters '
                f'({legacy[:3]}{"..." if len(legacy) > 3 else ""}); PatchWAM '
                'DiT adaptation now goes through PEFT (`vla_head.peft`), whose '
                'adapter tensors have different names. Retrain from the base '
                'policy with `vla_head.peft` or convert the payload to the '
                'PEFT `base_layer` naming before loading it')
        released = released_action_patch_state(state)
        if released is not None:
            available = released
            format_name = 'action_patch'
        else:
            available = {name: tensor for name, tensor in state.items()}
            format_name = _classify_checkpoint(available)
        model_keys = set(self.state_dict())
        trainable = {
            name for name, param in self.named_parameters()
            if param.requires_grad
        }
        resolved: Dict[str, str] = {}
        unexpected = []
        for name in available:
            if format_name in ('native', 'action_patch'):
                target = name
            elif format_name == 'compact':
                if name.startswith('policy.'):
                    target = 'vla_head.' + name[len('policy.'):]
                elif name.startswith('text_encoder.'):
                    target = 'vlm_backbone.text_encoder.' + name[
                        len('text_encoder.'):]
                else:
                    target = None
            else:  # bare
                target = 'vla_head.' + name
            if target is not None and target not in model_keys:
                # A PEFT-adapted denoiser keeps the released tensors under the
                # wrapped layer's ``base_layer`` child.
                target = _peft_base_layer_target(target, model_keys) or target
            if target is None or target not in model_keys:
                unexpected.append(name)
                continue
            resolved[target] = name
        allowed = set(allow_missing)
        unknown_allowed = sorted(allowed - model_keys)
        if unknown_allowed:
            raise ValueError(
                'Allowed-missing keys are not model keys: '
                f'{unknown_allowed[:8]}')
        buffers = {name for name, _ in self.named_buffers()}
        inherited_allowed = sorted(allowed - trainable - buffers)
        if inherited_allowed:
            # Only a stage's own new modules may be absent; an inherited tensor
            # must always be present so initialization stays strict.
            raise ValueError(
                'Only new trainable modules may be missing from a staged '
                f'checkpoint: {inherited_allowed[:8]}')
        if allowed:
            # A staged or unstaged-adapter load inherits the whole head,
            # including the tensors the policy freezes: the pretrained denoiser
            # projections, the state projection, and the DiT blocks must all be
            # present even though they are not trainable, and stage 2 must also
            # carry stage 1's visual projections. Frozen encoder assets stay in
            # the config, so only `vla_head.*` keys are required here.
            required = {name for name in model_keys
                        if name.startswith(HEAD_PREFIX)}
        else:
            # Legacy/unstaged behavior: every trainable tensor must be present.
            required = trainable
        missing = sorted(required - set(resolved) - allowed)
        return available, format_name, resolved, missing, unexpected

    def _read_checkpoint_state(self, path: str):
        """Read a native/compact/bare checkpoint payload from local files."""
        if path.endswith('.safetensors') or os.path.isdir(path):
            if path.endswith('.safetensors'):
                return load_file(path, device='cpu')
            state = {}
            for file in os.listdir(path):
                if file.endswith('.safetensors'):
                    state.update(load_file(os.path.join(path, file),
                                           device='cpu'))
            return state
        if path.endswith(('.pt', '.pth')):
            # weights_only=True rejects arbitrary pickle payloads and mmap keeps
            # multi-GB payloads out of host RAM. Legacy non-zip saves fall back
            # to a plain safe load.
            try:
                checkpoint = torch.load(
                    path, map_location='cpu', weights_only=True, mmap=True)
            except RuntimeError:
                checkpoint = torch.load(
                    path, map_location='cpu', weights_only=True)
            state = (checkpoint['model']
                     if isinstance(checkpoint, dict) and 'model' in checkpoint
                     else checkpoint)
            if isinstance(checkpoint, dict) and 'step' in checkpoint:
                overwatch.info(
                    f'PatchWAM checkpoint step metadata: {checkpoint["step"]}')
            return state
        raise ValueError(f'Unsupported checkpoint format: {path}')

    def from_pretrained(self) -> None:
        """Load the recipe's initialization source, format-aware.

        A staged recipe loads its own source: stage 1 the released/native
        pretrained PatchWAM policy, stage 2 the stage 1 checkpoint, both through
        the staged allow-list. An unstaged adapter finetune loads its base
        pretrained payload through the adapter allow-list: every inherited head
        tensor (including the frozen DiT blocks) must be present, and only the
        new PEFT adapter tensors may be absent. Other unstaged recipes keep the
        original native/full/compact behavior.
        """
        if self.visual_stage is not None:
            self._stage_initialization = self.initialize_visual_stage()
            return
        path = self.pretrained_name_or_path
        if path is None:
            return
        state = self._read_checkpoint_state(path)
        self._validate_pretrained_skip_prefixes()
        if self.peft_finetune:
            allowed = self.finetune_new_model_keys()
            report = self.load_patchwam_state_dict(state, strict=True,
                                                   allow_missing=allowed)
            report.update({'new_keys': sorted(allowed), 'source': str(path)})
            self._finetune_initialization = report
            overwatch.info(f'PatchWAM adapter finetune initialized: {report}')
            return
        report = self.load_patchwam_state_dict(state, strict=True)
        overwatch.info(f'PatchWAM checkpoint loaded: {report}')

    def freeze_backbones(self) -> None:
        """Freeze the backbone and apply the configured stage contract.

        The DDP runner freezes before ``from_pretrained`` and the FSDP runner
        after it, so this must be order independent: the stage policy sets every
        ``requires_grad`` flag explicitly, and the weight copy that follows or
        precedes it never re-opens a frozen tensor. The unstaged adapter finetune
        mode re-applies its own policy here for the same reason.
        """
        super().freeze_backbones()
        if self.peft_finetune:
            self.apply_peft_finetune_policy()
            return
        if self.visual_stage is not None:
            self.apply_visual_stage_policy()

    # ------------------------------------------------------------------
    # Unstaged DiT PEFT adapter finetuning
    # ------------------------------------------------------------------
    def finetune_new_model_keys(self) -> set:
        """Model keys a base pretrained payload may leave missing.

        Empty unless the head runs the unstaged adapter policy, where exactly
        the attached PEFT adapter tensors are new on top of the base policy.
        """
        resolver = getattr(self.vla_head, 'finetune_new_keys', None)
        if not callable(resolver):
            return set()
        return {f'{HEAD_PREFIX}{name}' for name in resolver()}

    def apply_peft_finetune_policy(self) -> dict:
        """Freeze the frontend and open the head's unstaged adapter policy.

        The vision backbone and the instruction encoder are frozen outright,
        including the VL language LoRA residuals ``PatchWAMBackbone`` would
        otherwise reopen, so the native optimizer (which selects on
        ``requires_grad``) sees only the head's trainable parameters. Runs after
        ``super().freeze_backbones()`` because that call applies the backbone's
        own trainable policy.
        """
        if not self.peft_finetune:
            return {}
        if self.vlm_backbone is not None:
            self.vlm_backbone.requires_grad_(False)
            if self.vlm_backbone.training:
                self.vlm_backbone.eval()
        return self.vla_head.apply_peft_finetune_policy()

    def trainable_parameter_report(self) -> Dict[str, Any]:
        """Fully qualified trainable names for the active freeze policy.

        Reports the model-level names the optimizer would select, the frozen
        DiT block patterns, and the backbone's trainable parameters (empty in
        the unstaged adapter finetune mode, where both encoders stay frozen).
        """
        names = sorted(name for name, param in self.named_parameters()
                       if param.requires_grad)
        block_patterns = getattr(self.vla_head, 'dit_block_patterns', None)
        head_report = getattr(self.vla_head, 'trainable_parameter_report', None)
        if self.peft_finetune:
            mode = 'peft_finetune'
        else:
            mode = self.visual_stage or 'none'
        return {
            'mode': mode,
            'trainable_parameter_names': names,
            'num_trainable_parameters': len(names),
            'dit_block_patterns': [
                f'{HEAD_PREFIX}{pattern}'
                for pattern in (block_patterns() if callable(block_patterns)
                                else [])
            ],
            'backbone_trainable_parameters': [
                name for name in names if name.startswith('vlm_backbone.')
            ],
            'head': head_report() if callable(head_report) else None,
        }

    # ------------------------------------------------------------------
    # Staged visual-backbone adaptation
    # ------------------------------------------------------------------
    def staged_new_model_keys(self, stage: Optional[str] = None) -> set:
        """Model keys this stage introduces on top of the previous checkpoint."""
        stage = self.visual_stage if stage is None else stage
        resolver = getattr(self.vla_head, 'staged_new_keys', None)
        if stage is None or not callable(resolver):
            return set()
        return {f'vla_head.{name}' for name in resolver(stage)}

    def apply_visual_stage_policy(self) -> dict:
        """Apply the configured stage's freezing to the whole native model.

        Stage 1 opens only the new visual projections; stage 2 opens the
        projections and the DiT PEFT adapters. The frozen visual backbone, text
        encoder, state projection, and pretrained denoiser projections stay
        frozen, so the native optimizer (which selects on ``requires_grad``)
        sees exactly the stage's parameter groups.
        """
        if self.visual_stage is None:
            return {}
        head = getattr(self.vla_head, 'apply_stage_policy', None)
        if not callable(head):
            raise ValueError(
                'The configured PatchWAM head does not support staged '
                'adaptation')
        resolved = self.vla_head.apply_stage_policy(self.visual_stage)
        if self.vlm_backbone is not None:
            text_encoder = getattr(self.vlm_backbone, 'text_encoder', None)
            if getattr(text_encoder, 'trainable_adapters', False):
                raise ValueError(
                    'A staged visual recipe trains only the visual projections '
                    'and the DiT PEFT adapters; VL text adapters must stay '
                    'frozen (vl_trainable_adapters=False)')
            # Freeze the frontend outright: the backbone's own trainable policy
            # would reopen VL language adapters, which is outside the stage
            # contract.
            self.vlm_backbone.requires_grad_(False)
            if self.vlm_backbone.training:
                self.vlm_backbone.eval()
        return resolved

    def initialize_visual_stage(self) -> Dict[str, Any]:
        """Initialize the active visual stage from its own source checkpoint.

        Stage 1 inherits the released/native pretrained PatchWAM policy;
        stage 2 inherits a stage 1 checkpoint. Every inherited tensor must be
        present with an identical shape and finite value, and only the stage's
        own new keys may be absent. Validation completes before any parameter
        is written, and no optimizer state is read, so stages never reuse each
        other's optimizer.
        """
        stage = self.visual_stage
        if stage is None:
            raise ValueError('This model has no configured visual_stage')
        source = (self.stage1_checkpoint if stage == 'stage2' else
                  self.pretrained_name_or_path)
        if not source:
            raise ValueError(
                f'visual_stage={stage} requires '
                f'{"stage1_checkpoint" if stage == "stage2" else "pretrained_name_or_path"}')
        state = self._read_checkpoint_state(source)
        allowed = self.staged_new_model_keys(stage)
        if self.vlm_backbone is not None and getattr(self.vlm_backbone,
                                                     'visual_encoder', None) is None:
            allowed = {key for key in allowed if '.visual_in' not in key
                       and '.visual_out' not in key}
        report = self.load_patchwam_state_dict(state, strict=True,
                                               allow_missing=allowed)
        report.update({'stage': stage, 'source': str(source),
                       'new_keys': sorted(allowed)})
        overwatch.info(f'PatchWAM visual stage initialized: {report}')
        return report

    def checkpoint_mapping_report(
        self,
        checkpoint_keys,
        name_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Resolve model keys against a source/native checkpoint key set.

        Returns ``resolved`` (model key -> checkpoint key), ``missing`` model
        keys, and ``unexpected`` checkpoint keys. With ``strict_mapping`` set on
        the model, any missing or unexpected key raises instead of silently
        leaving random trainable weights.
        """
        mapping = dict(name_mapping if name_mapping is not None else
                       (self.name_mapping or DEFAULT_PATCHWAM_NAME_MAPPING))
        available = set(checkpoint_keys)
        resolved: Dict[str, str] = {}
        missing: List[str] = []
        for name in self.state_dict():
            candidate = _apply_name_mapping(name, mapping)
            if candidate in available:
                resolved[name] = candidate
            else:
                missing.append(name)
        used = set(resolved.values())
        unexpected = sorted(available - used)
        if self.strict_mapping and (missing or unexpected):
            raise ValueError(
                'Strict PatchWAM checkpoint mapping failed: '
                f'missing={missing[:8]}{"..." if len(missing) > 8 else ""}, '
                f'unexpected={unexpected[:8]}{"..." if len(unexpected) > 8 else ""}'
            )
        return {'resolved': resolved, 'missing': missing,
                'unexpected': unexpected}

    # ------------------------------------------------------------------
    # BaseVLA integration
    # ------------------------------------------------------------------
    def get_fsdp_wrapping_policy(self) -> Callable:
        # Wrap the one module whose ``forward`` is actually called across the
        # denoiser boundary. ``OfficialFluxDenoiser`` invokes the inner
        # ``double_blocks``/``single_blocks`` via ``_paired_block`` /
        # ``_unified_block`` instead of their ``forward``, so auto-wrapping
        # those blocks would bypass their FSDP all-gather hooks. Encoders stay
        # frozen in the root unit; ownership is disjoint from the head.
        from fluxvla.engines.utils.fsdp_wrapping import build_module_wrap_policy
        return build_module_wrap_policy({PatchWAMHead})

    @property
    def config(self):
        from transformers import PretrainedConfig
        cfg = PretrainedConfig()
        cfg.is_encoder_decoder = False
        return cfg


def _auxiliary_weight(backbone: Optional[torch.nn.Module]) -> float:
    text_encoder = getattr(backbone, 'text_encoder', None)
    weight = getattr(text_encoder, 'auxiliary_weight', None)
    if weight is None:
        return 1.0
    weight = float(weight)
    if not torch.isfinite(torch.tensor(weight)) or weight < 0:
        raise ValueError(
            'VL auxiliary loss weight must be finite and non-negative')
    return weight
