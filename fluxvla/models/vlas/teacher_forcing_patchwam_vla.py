"""Teacher-feature PatchWAM VLA with frozen encoder resource lifecycle."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from fluxvla.engines import VLAS
from ..backbones.visions.teacher_features import TeacherFeatureProvider
from ..third_party_models.patchwam.teacher_forcing import RasterLayout
from ..third_party_models.patchwam.encoding import (
    resolve_endpoint_video,
)
from .patchwam_vla import HEAD_PREFIX, PatchWAMVLA, _ENCODE_KEYS, _TOKEN_KEYS

__all__ = ['TeacherForcingPatchWAMVLA']

_TEACHER_TOKEN_KEYS = (
    'teacher_current_tokens',
    'teacher_future_tokens',
    'teacher_current_valid',
    'teacher_future_valid',
    'teacher_current_fingerprints',
    'teacher_future_fingerprints',
)


@VLAS.register_module()
class TeacherForcingPatchWAMVLA(PatchWAMVLA):
    """Native PatchWAM with frozen current/future teacher descriptors."""

    TOKEN_KEYS = _TOKEN_KEYS + _TEACHER_TOKEN_KEYS
    ENCODE_KEYS = _ENCODE_KEYS + _TEACHER_TOKEN_KEYS

    def __init__(
        self,
        *args: Any,
        teacher_extractors: Optional[Mapping[str, Any]] = None,
        teacher_provenance: Optional[Mapping[str, Mapping[str, Any]]] = None,
        teacher_checkpoint_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        head = self.vla_head
        if not getattr(head, 'teacher_enabled', False):
            raise ValueError(
                'TeacherForcingPatchWAMVLA requires a '
                'TeacherForcingPatchWAMHead')
        if self.visual_stage is not None or self.peft_finetune:
            raise ValueError(
                'Teacher feature forcing supports full head finetuning; '
                'staged/PEFT PatchWAM modes are unsupported')
        if (getattr(head, 'fused_hidden_layer', None) is not None
                or getattr(head, 'velocity_transform', None) is not None):
            raise ValueError(
                'Teacher feature forcing cannot use the fused RADIO velocity '
                'transform')
        if teacher_checkpoint_mode not in (None, 'base_init', 'resume'):
            raise ValueError(
                'teacher_checkpoint_mode must be base_init or resume')
        if (self.pretrained_name_or_path is not None
                and teacher_checkpoint_mode is None):
            raise ValueError(
                'Teacher-enabled initialization requires explicit '
                'teacher_checkpoint_mode=base_init or resume')
        if (teacher_checkpoint_mode == 'resume'
                and self.pretrained_name_or_path is None):
            raise ValueError(
                'teacher_checkpoint_mode=resume requires '
                'pretrained_name_or_path')
        self.teacher_checkpoint_mode = teacher_checkpoint_mode
        self.teacher_extractors = dict(teacher_extractors or {})
        self.teacher_provenance = {
            name: dict(value)
            for name, value in dict(teacher_provenance or {}).items()
        }
        expected = {spec.name for spec in head.teacher_config.encoders}
        if self.teacher_extractors and set(self.teacher_extractors) != expected:
            raise ValueError(
                'teacher_extractors names must match the head encoders '
                f'exactly; expected={sorted(expected)}, '
                f'got={sorted(self.teacher_extractors)}')
        if (self.teacher_extractors
                and set(self.teacher_provenance) != set(
                    self.teacher_extractors)):
            raise ValueError(
                'teacher_provenance names must match teacher_extractors '
                f'exactly; expected={sorted(self.teacher_extractors)}, '
                f'got={sorted(self.teacher_provenance)}')
        self._teacher_provider: Optional[TeacherFeatureProvider] = None

    @property
    def teacher_provider(self) -> TeacherFeatureProvider:
        if self._teacher_provider is None:
            self._teacher_provider = TeacherFeatureProvider(
                self.vla_head.teacher_config, self.teacher_extractors,
                self.teacher_provenance)
        return self._teacher_provider

    def close_teacher_extractors(self) -> None:
        """Release owned encoder runtimes; the next extraction reloads them."""
        if self._teacher_provider is not None:
            self._teacher_provider.close()

    def _teacher_names(self) -> set[str]:
        return {spec.name for spec in self.vla_head.teacher_config.encoders}

    def _require_complete_role(self, batch: Mapping[str, Any],
                               role: str) -> bool:
        fields = ('tokens', 'valid', 'fingerprints')
        present = [
            f'teacher_{role}_{field}' in batch for field in fields]
        if any(present) and not all(present):
            missing = [
                f'teacher_{role}_{field}'
                for field, exists in zip(fields, present) if not exists]
            raise ValueError(
                f'Incomplete pre-encoded {role} teacher features; '
                f'missing={missing}')
        if all(present):
            names = set(batch[f'teacher_{role}_tokens'])
            if names != self._teacher_names():
                raise ValueError(
                    f'teacher_{role}_tokens names must match configured '
                    f'encoders exactly; expected={sorted(self._teacher_names())},'
                    f' got={sorted(names)}')
            return True
        return False

    def _merge_teacher_features(self, batch: Dict[str, Any], role: str,
                                output: Mapping[str, Any]) -> None:
        for field in ('tokens', 'valid', 'fingerprints'):
            batch[f'teacher_{role}_{field}'] = output[field]

    def _prepare_batch(
        self, images: Optional[torch.Tensor], task_description: Any,
        target: bool, kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        batch = super()._prepare_batch(
            images, task_description, target, kwargs)
        config = self.vla_head.teacher_config
        have_current = self._require_complete_role(batch, 'current')
        have_future = (
            self._require_complete_role(batch, 'future') if target else False)
        need_current = config.current_conditioning and not have_current
        need_future = (
            target and config.future_prediction and not have_future)
        if not need_current and not need_future:
            return batch
        source = dict(kwargs)
        if images is not None:
            source['images'] = images
        if not any(key in source for key in
                   ('camera_video', 'video', 'images')):
            roles = []
            if need_current:
                roles.append('current')
            if need_future:
                roles.append('future')
            raise ValueError(
                f'Missing teacher RGB for roles {roles}; provide the '
                'observation video or validated pre-encoded descriptors')
        current_frames, future_frames, _, _ = resolve_endpoint_video(
            source, target=target)
        current_layout_mapping = batch.get('current_raster_layout')
        if current_layout_mapping is None:
            raise ValueError(
                'Teacher extraction requires current_raster_layout metadata')
        current_layout = RasterLayout.from_mapping(current_layout_mapping)
        reference_valid = batch.get('reference_valid')
        if reference_valid is None:
            current_valid = torch.ones(
                current_frames.shape[0], current_layout.token_count,
                dtype=torch.bool, device=current_frames.device)
        else:
            current_valid = reference_valid[
                :, :current_layout.token_count].to(
                    device=current_frames.device, dtype=torch.bool)
        if need_current:
            output = self.teacher_provider.encode_context(
                current_frames, role='current', layout=current_layout,
                raster_valid=current_valid)
            self._merge_teacher_features(batch, 'current', output)
        if need_future:
            if future_frames is None:
                raise ValueError(
                    'Future teacher descriptors require the future endpoint')
            future_layout_mapping = batch.get('future_raster_layout')
            future_layout = (
                current_layout if future_layout_mapping is None else
                RasterLayout.from_mapping(future_layout_mapping))
            future_valid = batch.get('future_valid')
            if future_valid is None:
                future_valid = torch.ones(
                    future_frames.shape[0], future_layout.token_count,
                    dtype=torch.bool, device=future_frames.device)
            else:
                future_valid = future_valid.to(
                    device=future_frames.device, dtype=torch.bool)
            output = self.teacher_provider.encode_context(
                future_frames, role='future', layout=future_layout,
                raster_valid=future_valid)
            self._merge_teacher_features(batch, 'future', output)
        return batch

    def _require_tokens(self, batch: Dict[str, Any], target: bool) -> None:
        super()._require_tokens(batch, target)
        if (self.vla_head.teacher_config.current_conditioning
                and 'teacher_current_tokens' not in batch):
            raise ValueError(
                'Teacher current conditioning requires teacher_current_tokens')

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------
    def teacher_new_model_keys(self) -> set[str]:
        return {
            f'{HEAD_PREFIX}{name}'
            for name in self.vla_head.teacher_new_state_keys()
        }

    def _teacher_checkpoint_preflight(
        self, state: Mapping[str, Any], *,
        allow_missing_teacher: bool = False,
    ) -> bool:
        return self.vla_head._teacher_state_preflight(
            state, allow_missing_teacher=allow_missing_teacher)

    def _validate_inherited_native_state(
        self, state: Mapping[str, Any],
    ) -> None:
        names = {str(name) for name in state}
        if not any(name.startswith(('vla_head.', 'vlm_backbone.'))
                   for name in names):
            return
        required = set(self.state_dict()) - self.teacher_new_model_keys()
        missing = sorted(required - names)
        if missing:
            raise ValueError(
                'Inherited-only checkpoint is missing non-teacher model '
                f'tensors: {missing[:8]}'
                f'{"..." if len(missing) > 8 else ""}')

    def load_state_dict(self, state_dict, strict: bool = True,
                        **kwargs):
        allow_missing_teacher = self.teacher_checkpoint_mode == 'base_init'
        present = self._teacher_checkpoint_preflight(
            state_dict, allow_missing_teacher=allow_missing_teacher)
        if not present and allow_missing_teacher:
            self._validate_inherited_native_state(state_dict)
        return super().load_state_dict(
            state_dict, strict=strict, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        relative = {
            str(name)[len(prefix):] if str(name).startswith(prefix) else str(name):
            value for name, value in state_dict.items()
        }
        allow_missing_teacher = self.teacher_checkpoint_mode == 'base_init'
        present = self._teacher_checkpoint_preflight(
            relative, allow_missing_teacher=allow_missing_teacher)
        if not present and allow_missing_teacher:
            self._validate_inherited_native_state(relative)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def load_patchwam_state_dict(self, state, *, strict: bool = True,
                                 allow_missing=()):
        teacher_keys = self.teacher_new_model_keys()
        if self.teacher_checkpoint_mode == 'base_init':
            requested = set(allow_missing)
            unexpected = sorted(requested - teacher_keys)
            if unexpected:
                raise ValueError(
                    'base_init may omit only new teacher tensors; invalid '
                    f'allow_missing={unexpected[:8]}')
            self._teacher_checkpoint_preflight(
                state, allow_missing_teacher=True)
            return super().load_patchwam_state_dict(
                state, strict=strict, allow_missing=teacher_keys)
        self._teacher_checkpoint_preflight(
            state, allow_missing_teacher=False)
        # Resume/identity restores ignore caller allow-lists. Only explicit
        # base_init owns the missing-new allow-list.
        return super().load_patchwam_state_dict(
            state, strict=strict, allow_missing=())

    def from_pretrained(self) -> None:
        path = self.pretrained_name_or_path
        if path is None:
            return
        state = self._read_checkpoint_state(path)
        self._validate_pretrained_skip_prefixes()
        mode = self.teacher_checkpoint_mode
        if mode == 'resume':
            report = self.load_patchwam_state_dict(state, strict=True)
        elif mode == 'base_init':
            allowed = self.teacher_new_model_keys()
            report = self.load_patchwam_state_dict(
                state, strict=True, allow_missing=allowed)
            report.update({'new_keys': sorted(allowed)})
        else:  # pragma: no cover - guarded in __init__
            raise RuntimeError('Teacher checkpoint mode is not configured')
        report.update({'mode': mode, 'source': str(path)})
        self._teacher_initialization = report
        return report


__all__ = ['TeacherForcingPatchWAMVLA']
