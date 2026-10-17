"""Validated configuration and numerical helpers for teacher feature forcing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
_INITIALIZERS = {'zero', 'scaled_xavier_uniform'}
_PREDICTORS = {'linear', 'mlp2', 'repa_mlp'}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f'{field} must be a positive integer')
    return value


def _positive_pair(value: Any, field: str) -> tuple[int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(item) is not int or item < 1 for item in value)):
        raise ValueError(f'{field} must be a pair of positive integers')
    return int(value[0]), int(value[1])


def _probability(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f'{field} must be a finite value in [0,1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{field} must be a finite value in [0,1]')
    return value


@dataclass(frozen=True)
class TeacherEncoderSpec:
    """Static identity and native raster contract for one frozen encoder."""

    name: str
    feature_dim: int
    representation_id: str
    checkpoint_sha256: str | None
    stream: str
    preprocessing: str
    input_size: tuple[int, int]
    native_grid: tuple[int, int]
    view_context: str = 'independent_rasters'
    view_order: str = 'view_major_row_major'
    special_tokens: str = 'excluded'
    alignment: str = 'native_exact'
    loss_weight: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(
                'encoder name must start with a letter and contain only '
                'letters, digits, and underscores')
        object.__setattr__(
            self, 'feature_dim',
            _positive_int(self.feature_dim, 'feature_dim'))
        for field_name in ('representation_id', 'stream', 'preprocessing'):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{field_name} must be a non-empty string')
        if self.checkpoint_sha256 is not None:
            digest = self.checkpoint_sha256
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in '0123456789abcdef'
                           for char in digest.lower())):
                raise ValueError(
                    'checkpoint_sha256 must be a 64-character hexadecimal '
                    'digest')
            object.__setattr__(self, 'checkpoint_sha256', digest.lower())
        object.__setattr__(
            self, 'input_size',
            _positive_pair(self.input_size, 'input_size'))
        object.__setattr__(
            self, 'native_grid',
            _positive_pair(self.native_grid, 'native_grid'))
        if self.view_context != 'independent_rasters':
            raise ValueError(
                'Only independent_rasters is supported; joint-view encoders '
                'need a context-aware adapter')
        if self.view_order != 'view_major_row_major':
            raise ValueError(
                'view_order must be view_major_row_major to match the PatchWAM '
                'raster ordering')
        if self.special_tokens != 'excluded':
            raise ValueError(
                'special_tokens must be excluded from the dense spatial '
                'stream')
        if self.alignment != 'native_exact':
            raise ValueError(
                'alignment must be native_exact; extracted features are never '
                'interpolated or resized')
        if (not isinstance(self.loss_weight, (int, float))
                or isinstance(self.loss_weight, bool)
                or not math.isfinite(float(self.loss_weight))
                or float(self.loss_weight) < 0):
            raise ValueError('loss_weight must be finite and nonnegative')
        object.__setattr__(self, 'loss_weight', float(self.loss_weight))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'TeacherEncoderSpec':
        if not isinstance(value, Mapping):
            raise TypeError('encoder specification must be a mapping')
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - names)
        if unknown:
            raise ValueError(f'Unknown encoder specification keys: {unknown}')
        missing = sorted(names - set(value))
        required = {
            'name', 'feature_dim', 'representation_id', 'checkpoint_sha256',
            'stream', 'preprocessing', 'input_size', 'native_grid',
        }
        absent = sorted(name for name in required if name in missing)
        if absent:
            raise ValueError(f'Encoder specification is missing {absent}')
        return cls(**{name: value[name] for name in names if name in value})

    def identity(self) -> dict[str, Any]:
        """Canonical representation identity, excluding loss and runtime data."""
        return {
            'name': self.name,
            'feature_dim': self.feature_dim,
            'representation_id': self.representation_id,
            'checkpoint_sha256': self.checkpoint_sha256,
            'stream': self.stream,
            'preprocessing': self.preprocessing,
            'input_size': list(self.input_size),
            'native_grid': list(self.native_grid),
            'view_context': self.view_context,
            'view_order': self.view_order,
            'special_tokens': self.special_tokens,
            'alignment': self.alignment,
        }


@dataclass(frozen=True)
class TeacherForcingConfig:
    """Validated switches and architecture for teacher feature forcing."""

    encoders: tuple[TeacherEncoderSpec, ...]
    current_conditioning: bool = True
    future_prediction: bool = True
    future_feedback: bool = True
    predictor_type: str = 'repa_mlp'
    predictor_hidden_dim: int | None = 2048
    self_cond_drop_prob: float = 0.1
    true_feature_prob: float = 0.05
    current_projector_init: str = 'zero'
    feedback_projector_init: str = 'zero'
    current_projector_init_scale: float | None = None
    feedback_projector_init_scale: float | None = None
    feature_input_norm: str = 'layer_norm'
    norm_eps: float = 1e-5
    cosine_eps: float = 1e-6

    def __post_init__(self) -> None:
        if not self.encoders:
            raise ValueError('At least one teacher encoder is required')
        names = [encoder.name for encoder in self.encoders]
        if len(set(names)) != len(names):
            raise ValueError('Teacher encoder names must be unique')
        if any(not isinstance(encoder, TeacherEncoderSpec)
               for encoder in self.encoders):
            raise TypeError(
                'encoders must contain TeacherEncoderSpec instances')
        for name in ('current_conditioning', 'future_prediction',
                     'future_feedback'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if not any((self.current_conditioning, self.future_prediction,
                    self.future_feedback)):
            raise ValueError('At least one teacher feature switch must be on')
        if self.future_feedback and not self.future_prediction:
            raise ValueError(
                'future_feedback requires future_prediction')
        if self.predictor_type not in _PREDICTORS:
            raise ValueError(
                f'predictor.type must be one of {sorted(_PREDICTORS)}')
        if self.predictor_type == 'linear':
            if self.predictor_hidden_dim is not None:
                raise ValueError(
                    'predictor.hidden_dim is only valid for MLP predictors')
        else:
            object.__setattr__(
                self, 'predictor_hidden_dim',
                _positive_int(
                    2048 if self.predictor_hidden_dim is None else
                    self.predictor_hidden_dim, 'predictor.hidden_dim'))
        object.__setattr__(
            self, 'self_cond_drop_prob',
            _probability(self.self_cond_drop_prob,
                         'self_cond_drop_prob'))
        object.__setattr__(
            self, 'true_feature_prob',
            _probability(self.true_feature_prob, 'true_feature_prob'))
        for name in ('current_projector_init', 'feedback_projector_init'):
            if getattr(self, name) not in _INITIALIZERS:
                raise ValueError(
                    f'{name} must be one of {sorted(_INITIALIZERS)}')
        for name in ('current_projector_init_scale',
                     'feedback_projector_init_scale'):
            value = getattr(self, name)
            if value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool) or not math.isfinite(float(value))
                    or float(value) <= 0):
                raise ValueError(f'{name} must be finite and positive')
        if (self.current_projector_init == 'scaled_xavier_uniform'
                and self.current_projector_init_scale is None):
            raise ValueError(
                'current_projector_init_scale is required for '
                'scaled_xavier_uniform')
        if (self.feedback_projector_init == 'scaled_xavier_uniform'
                and self.feedback_projector_init_scale is None):
            raise ValueError(
                'feedback_projector_init_scale is required for '
                'scaled_xavier_uniform')
        if self.feature_input_norm != 'layer_norm':
            raise ValueError(
                'feature_input_norm must be layer_norm (non-affine)')
        for name in ('norm_eps', 'cosine_eps'):
            value = getattr(self, name)
            if (not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f'{name} must be finite and positive')
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'TeacherForcingConfig':
        if not isinstance(value, Mapping):
            raise TypeError('teacher_forcing must be a mapping')
        values = dict(value)
        allowed = {
            'encoders', 'current_conditioning', 'future_prediction',
            'future_feedback', 'predictor', 'self_cond_drop_prob',
            'true_feature_prob', 'current_projector_init',
            'feedback_projector_init', 'current_projector_init_scale',
            'feedback_projector_init_scale', 'feature_input_norm',
            'norm_eps', 'cosine_eps',
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f'Unknown teacher_forcing keys: {unknown}')
        encoders = values.pop('encoders', None)
        if not isinstance(encoders, Sequence) or isinstance(
                encoders, (str, bytes)) or not encoders:
            raise ValueError('teacher_forcing.encoders must be a non-empty list')
        resolved_encoders = []
        for encoder in encoders:
            resolved_encoders.append(
                encoder if isinstance(encoder, TeacherEncoderSpec) else
                TeacherEncoderSpec.from_mapping(encoder))
        predictor = values.pop('predictor', None)
        if predictor is None:
            predictor = dict(type='repa_mlp')
        elif isinstance(predictor, str):
            predictor = dict(type=predictor)
        if not isinstance(predictor, Mapping):
            raise TypeError('teacher_forcing.predictor must be a mapping')
        unknown_predictor = sorted(set(predictor) - {'type', 'hidden_dim'})
        if unknown_predictor:
            raise ValueError(
                f'Unknown predictor keys: {unknown_predictor}')
        predictor_type = predictor.get('type', 'repa_mlp')
        hidden_dim = predictor.get('hidden_dim')
        if predictor_type == 'linear' and hidden_dim is not None:
            raise ValueError(
                'predictor.hidden_dim is only valid for MLP predictors')
        return cls(encoders=tuple(resolved_encoders),
                   predictor_type=predictor_type,
                   predictor_hidden_dim=hidden_dim,
                   **values)

    @property
    def predictor_config(self) -> dict[str, Any]:
        result = {'type': self.predictor_type}
        if self.predictor_hidden_dim is not None:
            result['hidden_dim'] = self.predictor_hidden_dim
        return result

    def contract(self) -> dict[str, Any]:
        """Architecture and representation identity used by checkpoints."""
        return {
            'encoders': [encoder.identity() for encoder in self.encoders],
            'current_conditioning': self.current_conditioning,
            'future_prediction': self.future_prediction,
            'future_feedback': self.future_feedback,
            'predictor': self.predictor_config,
            'feature_input_norm': self.feature_input_norm,
            'norm_eps': self.norm_eps,
        }


def _initialize_predictor(module: nn.Module) -> nn.Module:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    return module


def build_feature_predictor(config: Mapping[str, Any] | str, *,
                            input_dim: int, output_dim: int) -> nn.Module:
    """Build Linear, MLP2, or the source three-linear REPA predictor."""
    input_dim = _positive_int(input_dim, 'input_dim')
    output_dim = _positive_int(output_dim, 'output_dim')
    if isinstance(config, str):
        config = {'type': config}
    if not isinstance(config, Mapping):
        raise TypeError('predictor config must be a mapping or type name')
    kind = config.get('type', 'repa_mlp')
    if kind not in _PREDICTORS:
        raise ValueError(f'Unknown predictor type {kind!r}')
    if kind == 'linear':
        if set(config) - {'type'}:
            raise ValueError('A linear predictor accepts only type')
        return _initialize_predictor(
            nn.Linear(input_dim, output_dim, bias=True))
    hidden = _positive_int(
        config.get('hidden_dim', 2048), 'predictor.hidden_dim')
    if set(config) - {'type', 'hidden_dim'}:
        raise ValueError('Unknown predictor keys')
    if kind == 'mlp2':
        module = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )
    else:
        module = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )
    return _initialize_predictor(module)


def prepare_teacher_fusion_input(tokens: Tensor, valid: Tensor, *,
                                 norm: nn.Module,
                                 allow_zero: bool = False,
                                 min_norm: float = 1e-6) -> Tensor:
    """Sanitize invalid rows and apply non-affine LayerNorm at each token."""
    if not torch.is_tensor(tokens) or tokens.ndim != 3:
        raise ValueError('teacher tokens must be [B,S,D]')
    if not torch.is_tensor(valid) or valid.shape != tokens.shape[:2]:
        raise ValueError('teacher validity must be [B,S]')
    if not torch.is_tensor(allow_zero) and type(allow_zero) is not bool:
        raise TypeError('allow_zero must be boolean')
    valid = valid.to(device=tokens.device, dtype=torch.bool)
    if valid.any():
        observed = tokens[valid]
        if not torch.isfinite(observed).all():
            raise ValueError(
                'Valid observed teacher descriptors must be finite')
        if not allow_zero:
            norms = observed.float().norm(dim=-1)
            if (norms <= float(min_norm)).any():
                raise ValueError(
                    'Valid observed teacher descriptors must be nonzero')
    safe = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
    normalized = norm(safe)
    return torch.where(valid[..., None], normalized,
                       torch.zeros_like(normalized))


def masked_teacher_cosine_loss(prediction: Tensor, target: Tensor,
                               valid: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Per-sample valid-token cosine mean, then batch mean."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('prediction and target must share [B,S,D]')
    if valid.shape != prediction.shape[:2]:
        raise ValueError('valid must be [B,S] matching prediction')
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    prediction = torch.where(valid[..., None], prediction,
                             torch.zeros_like(prediction))
    target = torch.where(valid[..., None], target.detach(),
                         torch.zeros_like(target))
    prediction_f = prediction.float()
    target_f = target.float()
    if valid.any() and not torch.isfinite(prediction_f[valid]).all():
        raise ValueError('Valid teacher predictions must be finite')
    if valid.any() and not torch.isfinite(target_f[valid]).all():
        raise ValueError('Valid teacher targets must be finite')
    cosine = (F.normalize(prediction_f, dim=-1, eps=eps)
              * F.normalize(target_f, dim=-1, eps=eps)).sum(-1)
    per_sample = torch.where(valid, 1 - cosine,
                             torch.zeros_like(cosine))
    counts = valid.sum(-1).clamp_min(1)
    return (per_sample.sum(-1) / counts).mean()


def teacher_representation_fingerprint(spec: TeacherEncoderSpec) -> bytes:
    """SHA-256 of the canonical encoder representation identity."""
    return hashlib.sha256(_canonical_json(spec.identity())).digest()


def teacher_policy_digest(config: TeacherForcingConfig) -> bytes:
    """SHA-256 of the architecture and representation compatibility contract."""
    return hashlib.sha256(_canonical_json(config.contract())).digest()


@dataclass(frozen=True)
class RasterLayout:
    """View and token geometry shared by VAE latents and teacher descriptors."""

    views: int
    video_size: tuple[int, int]
    grid: tuple[int, int]
    view_order: str = 'view_major_row_major'

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'views', _positive_int(self.views, 'views'))
        object.__setattr__(
            self, 'video_size',
            _positive_pair(self.video_size, 'video_size'))
        object.__setattr__(
            self, 'grid', _positive_pair(self.grid, 'grid'))
        if self.view_order != 'view_major_row_major':
            raise ValueError(
                'view_order must be view_major_row_major')

    @property
    def tokens_per_view(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def token_count(self) -> int:
        return self.views * self.tokens_per_view

    def to_mapping(self) -> dict[str, Any]:
        return {
            'views': self.views,
            'video_size': list(self.video_size),
            'grid': list(self.grid),
            'view_order': self.view_order,
            'tokens_per_view': self.tokens_per_view,
            'token_count': self.token_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'RasterLayout':
        if not isinstance(value, Mapping):
            raise TypeError('raster layout must be a mapping')
        return cls(views=value['views'], video_size=value['video_size'],
                   grid=value['grid'],
                   view_order=value.get('view_order',
                                        'view_major_row_major'))
