"""Registered teacher-feature head for the native PatchWAM VLA."""

from typing import Any, Mapping, Optional

import torch

from fluxvla.engines import HEADS
from ..third_party_models.patchwam.teacher_forcing import (
    TeacherForcingConfig,
    TeacherForcingPolicyMixin,
)
from .patchwam_head import PatchWAMHead


@HEADS.register_module()
class TeacherForcingPatchWAMHead(TeacherForcingPolicyMixin, PatchWAMHead):
    """PatchWAM head with current-feature fusion and future-feature forcing."""

    SUPPORTS_OPTIONAL_OBJECTIVES = False

    def __init__(
        self,
        *args: Any,
        teacher_forcing: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if teacher_forcing is None:
            raise ValueError(
                'TeacherForcingPatchWAMHead requires `teacher_forcing`')
        if kwargs.get('self_flow_variant', 0):
            raise ValueError(
                'Teacher feature forcing cannot be combined with Self-Flow')
        if kwargs.get('stage') is not None:
            raise ValueError(
                'Teacher feature forcing supports full head finetuning; '
                'staged visual adaptation is unsupported')
        if kwargs.get('peft') is not None or kwargs.get('lora') is not None:
            raise ValueError(
                'Teacher feature forcing supports full head finetuning; '
                'PEFT/LoRA adaptation is unsupported in this integration')
        visual_dim = kwargs.get('visual_dim')
        token_dim = kwargs.get('token_dim', 128)
        if visual_dim is not None and int(visual_dim) != int(token_dim):
            raise ValueError(
                'Teacher feature forcing requires the ordinary VAE path; '
                'visual_dim must equal token_dim')
        config = TeacherForcingConfig.from_mapping(teacher_forcing)
        super().__init__(*args, **kwargs)
        self._init_teacher_policy(config)

    @property
    def teacher_enabled(self) -> bool:
        return True

    def _teacher_relative_state(
        self, state_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        relative = {}
        for name, value in state_dict.items():
            name = str(name)
            for prefix in ('vla_head.', 'policy.'):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            relative[name] = value
        return relative

    def _teacher_state_preflight(
        self, state_dict: Mapping[str, Any], *,
        allow_missing_teacher: bool = False,
    ) -> bool:
        """Require a complete, identity-matching teacher restore before copy."""
        if not isinstance(state_dict, Mapping) or not state_dict:
            if allow_missing_teacher:
                return False
            raise ValueError('Teacher checkpoint state is empty')
        relative = self._teacher_relative_state(state_dict)
        required = set(self.teacher_new_state_keys())
        present = required & set(relative)
        if not present:
            if allow_missing_teacher:
                return False
            raise ValueError(
                'Teacher checkpoint restore is missing all teacher tensors; '
                'use explicit base_init for an inherited-only checkpoint')
        missing = sorted(required - set(relative))
        if missing:
            raise ValueError(
                'Teacher checkpoint restore is missing required tensors: '
                f'{missing[:8]}{"..." if len(missing) > 8 else ""}')
        digest = relative['teacher_contract_digest']
        if not torch.is_tensor(digest):
            raise ValueError(
                'Teacher checkpoint digest must be a uint8[32] tensor')
        expected = self.teacher_contract_digest.detach().cpu()
        actual = digest.detach().cpu().to(torch.uint8)
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            raise ValueError(
                'Teacher feature contract mismatch: encoder representation, '
                'switch, predictor, or normalization changed; retrain from a '
                'base_init checkpoint')
        return True

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Direct head loads are restores, never implicit base initialization."""
        self._teacher_state_preflight(state_dict, allow_missing_teacher=False)
        return super().load_state_dict(
            state_dict, strict=strict, **kwargs)


__all__ = ['TeacherForcingPatchWAMHead']
