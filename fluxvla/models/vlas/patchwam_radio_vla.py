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

from typing import Any, Dict

import torch

from fluxvla.engines import VLAS, initialize_overwatch
from ..heads.patchwam_radio_head import (  # noqa: F401  (registration)
    PatchWAMRadioAuxHead)
from .patchwam_vla import _ENCODE_KEYS, PatchWAMVLA

overwatch = initialize_overwatch(__name__)

__all__ = ['PatchWAMRadioAuxVLA']


class _RadioExtrasMixin:
    """Variant-only resume validation and an explicit legacy initializer.

    Native training resume (``load_state_dict``, including the runners'
    ``strict=False`` calls, and ``from_pretrained``) must restore every
    trainable tensor, including the newly introduced RADIO layers. The only
    relaxation is the explicitly named
    :meth:`initialize_from_legacy_state_dict`, which leaves exactly the
    declared new layers at their fresh initialization.
    """

    RADIO_PROJECTION_KEYS: tuple = ()
    #: Non-trainable tensors that native resume must also restore.
    RADIO_BUFFER_KEYS: tuple = ()

    def load_state_dict(self, state_dict, strict: bool = True,
                        assign: bool = False):
        """Reject incomplete/corrupt native resume before mutating weights."""
        self._assert_trainable_resume_keys(state_dict)
        self._assert_resume_buffers(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _assert_resume_buffers(self, state) -> None:
        """Require the fixed RADIO buffers even under ``strict=False``."""
        keys = getattr(self, 'RADIO_BUFFER_KEYS', ())
        if not keys or not hasattr(state, 'keys'):
            return
        current = self.state_dict()
        for name in keys:
            if name not in state:
                raise ValueError(
                    f'Strict PatchWAM resume rejected a checkpoint that is '
                    f'missing the trained RADIO buffer {name!r}')
            value = state[name]
            if not torch.is_tensor(value):
                raise TypeError(f'Checkpoint entry {name} is not a tensor')
            if tuple(value.shape) != tuple(current[name].shape):
                raise ValueError(
                    f'Buffer shape differs for {name}: {tuple(value.shape)} vs '
                    f'{tuple(current[name].shape)}')
            if not torch.isfinite(value.float()).all():
                raise ValueError(f'Buffer values are invalid for {name}')

    def _assert_trainable_resume_keys(self, state) -> None:
        if not hasattr(state, 'keys'):
            raise TypeError(
                'PatchWAM resume requires a state-dict mapping, got '
                f'{type(state)!r}')
        current = self.state_dict()
        trainable = [name for name, parameter in self.named_parameters()
                     if parameter.requires_grad]
        missing = [name for name in trainable if name not in state]
        if missing:
            raise ValueError(
                'Strict PatchWAM resume rejected a checkpoint that is missing '
                f'trainable tensors: {missing[:8]}'
                f'{"..." if len(missing) > 8 else ""}')
        for name in trainable:
            value = state[name]
            if not torch.is_tensor(value):
                raise TypeError(f'Checkpoint entry {name} is not a tensor')
            if tuple(value.shape) != tuple(current[name].shape):
                raise ValueError(
                    f'Weight shape differs for {name}: {tuple(value.shape)} vs '
                    f'{tuple(current[name].shape)}')
            if (current[name].is_floating_point()
                    and not torch.isfinite(
                        value.to(current[name].dtype)).all()):
                raise ValueError(f'Weight values are invalid for {name}')

    def initialize_from_legacy_state_dict(self, state,
                                          *, strict: bool = True) -> Dict[str, Any]:
        """Explicitly initialize from a pre-RADIO PatchWAM checkpoint.

        Exactly :attr:`RADIO_PROJECTION_KEYS` may be absent and stay at their
        fresh initialization; every other trainable tensor must be present with
        the right shape and finite values. This operation is deliberately
        separate from native resume so it can never hide a missing trained
        projection.
        """
        available, format_name, resolved, missing, unexpected = (
            self._resolve_patchwam_state(state))
        fresh = sorted(set(missing) & set(self.RADIO_PROJECTION_KEYS))
        # The fixed projection buffer is newly introduced as well; it stays at
        # its constructed (seeded) value when it is absent from the legacy
        # checkpoint.
        for name in getattr(self, 'RADIO_BUFFER_KEYS', ()):
            if name not in state:
                fresh.append(name)
        fresh = sorted(fresh)
        unresolved = sorted(set(missing) - set(self.RADIO_PROJECTION_KEYS))
        if strict and (unresolved or unexpected):
            raise ValueError(
                'Legacy PatchWAM initialization failed (format='
                f'{format_name}): missing={unresolved[:8]}'
                f'{"..." if len(unresolved) > 8 else ""}, '
                f'unexpected={sorted(unexpected)[:8]}'
                f'{"..." if len(unexpected) > 8 else ""}')
        self._apply_patchwam_state(available, resolved)
        return {'format': format_name, 'loaded_tensors': len(resolved),
                'left_fresh': fresh}

    def radio_projection_report(self) -> Dict[str, Any]:
        """Report whether the new RADIO layers are present and trainable."""
        parameters = dict(self.named_parameters())
        return {
            'declared': list(self.RADIO_PROJECTION_KEYS),
            'missing': [key for key in self.RADIO_PROJECTION_KEYS
                        if key not in parameters],
            'trainable': [key for key in self.RADIO_PROJECTION_KEYS
                          if key in parameters and parameters[key].requires_grad],
        }

    def checkpoint_mapping_report(self, checkpoint_keys, name_mapping=None):
        report = super().checkpoint_mapping_report(checkpoint_keys,
                                                   name_mapping)
        report['radio'] = self.radio_projection_report()
        return report


@VLAS.register_module()
class PatchWAMRadioAuxVLA(_RadioExtrasMixin, PatchWAMVLA):
    """Original VAE flow plus the fused linear RADIO velocity auxiliary.

    The training model carries the frozen teacher (outside the module tree),
    the three biased linear adapters ``radio_a``/``radio_b``/``radio_c`` and the
    fixed shared-noise buffer ``radio_noise_projection``.
    :meth:`export_inference_state_dict` folds the final-layer adapters into one
    biased native PatchWAM output head. Earlier-layer selections deploy through
    :meth:`export_unfused_inference_state_dict`, retaining the runtime adapters.
    Both deployment paths preserve the trained velocity residual.
    """

    #: Trainable adapter tensors introduced by the auxiliary variant.
    RADIO_PROJECTION_KEYS = (
        'vla_head.radio_a.weight', 'vla_head.radio_a.bias',
        'vla_head.radio_b.weight', 'vla_head.radio_b.bias',
        'vla_head.radio_c.weight', 'vla_head.radio_c.bias',
    )
    #: Fixed training-only projection buffer restored by native resume.
    RADIO_BUFFER_KEYS = ('vla_head.radio_noise_projection',)
    #: Tensors removed when folding the final-layer inference head.
    RADIO_TRAINING_KEYS = RADIO_PROJECTION_KEYS + RADIO_BUFFER_KEYS

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The auxiliary backbone accepts an explicit `need_radio` request flag
        # so a zero RADIO weight never loads or runs the teacher.
        self._encode_keys = tuple(_ENCODE_KEYS) + ('need_radio',)

    def _radio_targets_required(self) -> bool:
        return float(getattr(self.vla_head, 'radio_weight', 0.0)) > 0

    def _prepare_batch(self, images, task_description, target, kwargs):
        request = dict(kwargs)
        request['need_radio'] = self._radio_targets_required()
        return super()._prepare_batch(images, task_description, target,
                                      request)

    def _validate_backbone_contract(self) -> None:
        super()._validate_backbone_contract()
        if self.vlm_backbone is None:
            return
        backbone_dim = getattr(self.vlm_backbone, 'radio_dim', None)
        head_dim = getattr(self.vla_head, 'radio_dim', None)
        if backbone_dim is None or head_dim is None:
            raise ValueError(
                'PatchWAMRadioAuxVLA requires an auxiliary RADIO backbone and '
                'head exposing `radio_dim`.')
        if int(backbone_dim) != int(head_dim):
            raise ValueError(
                f'backbone radio_dim={backbone_dim} does not match head '
                f'radio_dim={head_dim}')

    def export_inference_state_dict(
            self, *, clone: bool = True) -> Dict[str, torch.Tensor]:
        """Public deployable default: exact bias-aware fused export.

        Folds ``A/B/C`` into the original final output projection so the trained
        residual is never lost:

        ``W = F + C A F + C B``, ``b = f + C(A f + a + b) + c`` in float32.
        The result loads strictly into the bias-enabled native PatchWAM stack
        (``denoiser.output_bias=True``) and needs no teacher, adapters or ``P``.
        An earlier-layer selection cannot be folded into one projection and
        raises clearly; deploy :meth:`export_unfused_inference_state_dict`
        instead.
        """
        return self.fused_output_state_dict(clone=clone)

    def export_unfused_inference_state_dict(
            self, *, clone: bool = True) -> Dict[str, torch.Tensor]:
        """Teacher-free runtime deployment state that keeps ``A/B/C`` and ``P``.

        Loads strictly back into a :class:`PatchWAMRadioAuxVLA` configured
        without a teacher (``radio_model_path=None``); no teacher weights exist
        in the state at any point. This is the documented deployment path for an
        earlier-layer selection, where no single-affine fold exists.
        """
        state = self.state_dict()
        return {name: (tensor.detach().clone() if clone else tensor)
                for name, tensor in state.items()}

    def export_original_core_state_dict(
            self, *, clone: bool = True) -> Dict[str, torch.Tensor]:
        """NON-DEPLOYABLE original PatchWAM core weights (initialization only).

        Drops the three adapters and ``P`` so the result matches the original
        PatchWAM key set. It intentionally does **not** carry the trained
        residual: use it only to initialize/compare against a plain model, never
        to deploy a trained variant.
        """
        state = self.state_dict()
        exported = {name: (tensor.detach().clone() if clone else tensor)
                    for name, tensor in state.items()
                    if name not in self.RADIO_TRAINING_KEYS}
        return exported

    def fused_output_state_dict(self, *, clone: bool = True):
        """Single-affine folded export for a final-layer adapter selection.

        Folds ``A/B/C`` into the original final output projection
        (``W = F + C A F + C B``, ``b = f + C(A f + a + b) + c``) and returns a
        state dict loadable by a bias-enabled native PatchWAM model. Early-layer
        selections cannot be fused into one projection and raise clearly; use
        the unfused export plus the teacher-free runtime adapters instead.
        """
        head = self.vla_head
        if not head.is_final_selection():
            raise ValueError(
                'The fused single-head export requires a final-layer '
                f'adapter selection (adapter_layer_idex={head.adapter_layer_idex} '
                'is an earlier block); deploy the unfused export with the '
                'teacher-free runtime adapters instead')
        state = self.export_original_core_state_dict(clone=clone)
        weight, bias = head.fused_output_affine()
        name = getattr(head.denoiser, 'final_output_name', None)
        if not name:
            raise ValueError(
                'The fused export requires a denoiser exposing final_output_name')
        prefix = f'vla_head.denoiser.{name}'
        dtype = state[f'{prefix}.weight'].dtype
        state[f'{prefix}.weight'] = weight.to(dtype)
        state[f'{prefix}.bias'] = bias.to(dtype)
        return state

    @staticmethod
    def _mentions_teacher(key: str) -> bool:
        return any(part in ('_radio_teacher', 'radio_teacher', 'teacher')
                   for part in key.split('.'))

    def exported_inference_keys_report(self) -> Dict[str, Any]:
        exported = (self.export_inference_state_dict(clone=False)
                    if self.vla_head.is_final_selection() else
                    self.export_unfused_inference_state_dict(clone=False))
        training_state = self.state_dict()
        return {
            'exported_tensors': len(exported),
            'removed': [key for key in self.RADIO_TRAINING_KEYS
                        if key in training_state and key not in exported],
            'teacher_keys': sorted(key for key in training_state
                                   if self._mentions_teacher(key)),
            'teacher_in_state_dict': any(self._mentions_teacher(key)
                                         for key in training_state),
        }
