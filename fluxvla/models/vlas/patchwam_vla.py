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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from fluxvla.engines import VLAS, initialize_overwatch
from ..heads.patchwam_head import PatchWAMHead  # noqa: F401  (registration)
from .base_vla import BaseVLA

overwatch = initialize_overwatch(__name__)

__all__ = ['DEFAULT_PATCHWAM_NAME_MAPPING', 'PatchWAMVLA',
           'RELEASED_ACTION_PATCH_TRANSFORMER_PREFIX',
           'resolve_action_padding', 'released_action_patch_state']

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
    'text_tokens',
    'text_valid',
    'reference_valid',
    'future_valid',
    'reference_ids',
    'future_ids',
    'proprio',
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

_DEFERRED_PREFIXES = ('self_flow', 'ema')


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

    def __init__(
        self,
        vlm_backbone: Optional[Dict] = None,
        vla_head: Optional[Dict] = None,
        action_horizon: Optional[int] = None,
        action_norm_type: str = 'min_max',
        num_inference_steps: int = 10,
        freeze_vlm_backbone: bool = True,
        pretrained_name_or_path: Optional[str] = None,
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
                for key in _ENCODE_KEYS
                if key in kwargs and kwargs[key] is not None
            }
            if images is not None:
                request['images'] = images
            if task_description is not None:
                request['task_description'] = task_description
                request.setdefault('prompt', task_description)
            encoded = self.vlm_backbone.encode(request, target=target)
            if not isinstance(encoded, dict):
                raise TypeError('vlm_backbone.encode must return a dict, got '
                                f'{type(encoded)!r}')
            batch.update(encoded)
        else:
            for key in _TOKEN_KEYS:
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
        """Run the PatchWAM flow-matching loss.

        ``frame_masks`` (``[B, T]``, current first and future endpoint last) and
        ``img_masks`` (``[B, V]`` camera validity) map to reference/future raster
        validity inside the backbone; pre-encoded callers may pass
        ``reference_valid`` / ``future_valid`` directly and those are preserved.
        """
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
        self._require_tokens(batch, target=actions is not None)
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
                                 *, strict: bool = True) -> Dict[str, int]:
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
            if target is None or target not in model_keys:
                unexpected.append(name)
                continue
            resolved[target] = name
        missing = sorted(trainable - set(resolved))
        if strict and (missing or unexpected):
            raise ValueError(
                f'Strict PatchWAM checkpoint load failed (format='
                f'{format_name}): missing={missing[:8]}'
                f'{"..." if len(missing) > 8 else ""}, '
                f'unexpected={sorted(unexpected)[:8]}'
                f'{"..." if len(unexpected) > 8 else ""}')
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
        return {'format': format_name, 'loaded_tensors': len(resolved)}

    def from_pretrained(self) -> None:
        """Load native/full or compact PatchWAM checkpoints format-aware."""
        path = self.pretrained_name_or_path
        if path is None:
            return
        if path.endswith('.safetensors') or os.path.isdir(path):
            if path.endswith('.safetensors'):
                state = load_file(path, device='cpu')
            else:
                state = {}
                for file in os.listdir(path):
                    if file.endswith('.safetensors'):
                        state.update(load_file(os.path.join(path, file),
                                               device='cpu'))
        elif path.endswith(('.pt', '.pth')):
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
        else:
            raise ValueError(f'Unsupported checkpoint format: {path}')
        self._validate_pretrained_skip_prefixes()
        report = self.load_patchwam_state_dict(state, strict=True)
        overwatch.info(f'PatchWAM checkpoint loaded: {report}')

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
