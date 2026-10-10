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

    def load_state_dict(self, state_dict, strict: bool = True,
                        assign: bool = False):
        """Reject incomplete/corrupt native resume before mutating weights."""
        self._assert_trainable_resume_keys(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

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
    """Original VAE flow plus training-only RADIO supervision.

    The training model carries the frozen teacher (outside the module tree) and
    one ``vla_head.radio_projection`` adapter.
    :meth:`export_inference_state_dict` returns the deployable artifact: the
    original native PatchWAM weights with only that training adapter removed,
    which an ordinary :class:`PatchWAMVLA` loads strictly and reproduces with
    identical seeded actions.
    """

    #: Trainable adapter weights introduced by the auxiliary variant.
    RADIO_PROJECTION_KEYS = (
        'vla_head.radio_projection.weight',
        'vla_head.radio_projection.bias',
    )

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
        """Return native PatchWAM weights without the training-only adapter.

        Only :attr:`RADIO_PROJECTION_KEYS` are dropped, so the result has the
        exact key set and values of the original PatchWAM model; the frozen
        teacher was never part of the module tree.
        """
        state = self.state_dict()
        exported = {name: (tensor.detach().clone() if clone else tensor)
                    for name, tensor in state.items()
                    if name not in self.RADIO_PROJECTION_KEYS}
        return exported

    @staticmethod
    def _mentions_teacher(key: str) -> bool:
        return any(part in ('_radio_teacher', 'radio_teacher', 'teacher')
                   for part in key.split('.'))

    def exported_inference_keys_report(self) -> Dict[str, Any]:
        exported = self.export_inference_state_dict(clone=False)
        training_state = self.state_dict()
        return {
            'exported_tensors': len(exported),
            'removed': [key for key in self.RADIO_PROJECTION_KEYS
                        if key in training_state],
            'teacher_keys': sorted(key for key in training_state
                                   if self._mentions_teacher(key)),
            'teacher_in_state_dict': any(self._mentions_teacher(key)
                                         for key in training_state),
        }
