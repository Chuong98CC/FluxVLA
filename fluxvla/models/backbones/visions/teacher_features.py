"""Configurable frozen teacher-feature extraction for PatchWAM."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fluxvla.engines import build_vision_backbone_from_cfg
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.models.third_party_models.patchwam.teacher_forcing.contracts import (
    RasterLayout,
    TeacherForcingConfig,
    teacher_representation_fingerprint,
)

_PROVENANCE_KEYS = {
    'representation_id', 'artifacts', 'representation_artifacts',
    'test_contract', 'base_dir',
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _representation_digest(actual: Mapping[str, str]) -> str:
    if len(actual) == 1:
        return next(iter(actual.values()))
    payload = json.dumps(
        sorted(actual.items()), separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _balanced_chunk_sizes(total: int, minimum: int, maximum: int) -> list[int]:
    if minimum < 1 or maximum < minimum:
        raise ValueError('Invalid runtime batch bounds')
    if total < minimum:
        raise ValueError(
            f'The runtime requires at least {minimum} valid rasters per call; '
            f'got {total}. Add valid views or use a runtime with min_batch=1')
    chunks = max(1, math.ceil(total / maximum))
    if total < chunks * minimum:
        raise ValueError(
            f'Cannot split {total} rasters into chunks within '
            f'[{minimum},{maximum}]')
    base, remainder = divmod(total, chunks)
    sizes = [base + 1] * remainder + [base] * (chunks - remainder)
    if any(size < minimum or size > maximum for size in sizes):
        raise ValueError(
            f'Cannot split {total} rasters into chunks within '
            f'[{minimum},{maximum}]')
    return sizes


class TeacherFeatureProvider:
    """Own frozen spatial encoders and extract role-specific descriptors.

    The provider is deliberately outside the registered trainable tree. It
    accepts injected modules for tests and ``type``-keyed vision-backbone
    configs for production, so RADIO PyTorch, RADIO TensorRT, and future
    adapters satisfying the spatial-feature contract are interchangeable.
    """

    def __init__(self, config: TeacherForcingConfig,
                 extractor_configs: Mapping[str, Any],
                 provenance: Mapping[str, Mapping[str, Any]]) -> None:
        if not isinstance(config, TeacherForcingConfig):
            raise TypeError('config must be a TeacherForcingConfig')
        if not isinstance(extractor_configs, Mapping):
            raise TypeError('teacher_extractors must be a mapping')
        if not isinstance(provenance, Mapping):
            raise TypeError('teacher_provenance must be a mapping')
        self.config = config
        self._extractor_configs = dict(extractor_configs)
        self._provenance = dict(provenance)
        self._extractors: dict[str, nn.Module] | None = None
        self._owned: set[str] = set()
        self._verified_provenance: set[str] = set()
        self._artifact_hash_cache: dict[str, str] = {}
        self._validate_runtime_names()
        self._validate_provenance_envelopes()

    @property
    def specs(self):
        return {spec.name: spec for spec in self.config.encoders}

    @property
    def extractors(self) -> dict[str, nn.Module]:
        return self._ensure_extractors()

    def _validate_runtime_names(self) -> None:
        expected = set(self.specs)
        configured = set(self._extractor_configs)
        if not configured:
            return
        missing = sorted(expected - configured)
        extra = sorted(configured - expected)
        if missing or extra:
            raise ValueError(
                f'Teacher extractor definitions must match configured '
                f'encoders exactly; missing={missing}, extra={extra}')

    def _validate_provenance_envelopes(self) -> None:
        expected = set(self.specs)
        if set(self._provenance) != expected:
            raise ValueError(
                'teacher_provenance names must match configured encoders '
                f'exactly; missing={sorted(expected - set(self._provenance))}, '
                f'extra={sorted(set(self._provenance) - expected)}')
        for spec in self.config.encoders:
            envelope = self._provenance[spec.name]
            if not isinstance(envelope, Mapping):
                raise TypeError(
                    f'Provenance for {spec.name!r} must be a mapping')
            unknown = sorted(set(envelope) - _PROVENANCE_KEYS)
            if unknown:
                raise ValueError(
                    f'Unknown provenance keys for {spec.name!r}: {unknown}')
            if envelope.get('representation_id') != spec.representation_id:
                raise ValueError(
                    f'Runtime provenance for {spec.name!r} does not match its '
                    'representation_id')
            test_contract = envelope.get('test_contract')
            if test_contract is not None:
                if not isinstance(test_contract, str) or not test_contract:
                    raise ValueError(
                        'test_contract must be a non-empty explicit string')
                continue
            artifacts = envelope.get('artifacts')
            representation = envelope.get('representation_artifacts')
            if (not isinstance(artifacts, Mapping) or not artifacts
                    or not isinstance(representation, (list, tuple))
                    or not representation):
                raise ValueError(
                    f'Provenance for {spec.name!r} requires non-empty '
                    'artifacts and representation_artifacts')
            if set(representation) - set(artifacts):
                raise ValueError(
                    f'representation_artifacts for {spec.name!r} must be '
                    'artifact keys')
            if spec.checkpoint_sha256 is None:
                raise ValueError(
                    f'Production provenance for {spec.name!r} requires '
                    'checkpoint_sha256 in the encoder specification')

    def _verify_provenance(self, name: str,
                           extractor_config: Any) -> None:
        if name in self._verified_provenance:
            return
        spec = self.specs[name]
        envelope = self._provenance[name]
        representation_id = envelope.get('representation_id')
        if representation_id != spec.representation_id:
            raise ValueError(
                f'Runtime provenance for {name!r} does not match its '
                'representation identity')
        if envelope.get('test_contract') is not None:
            if not isinstance(extractor_config, nn.Module):
                raise ValueError(
                    'test_contract provenance is reserved for an explicitly '
                    'injected fake encoder; registry production configs must '
                    'provide artifacts and representation_artifacts')
        else:
            artifacts = dict(envelope['artifacts'])
            representation = tuple(envelope['representation_artifacts'])
            base_dir = envelope.get('base_dir')
            base_path = None if base_dir is None else Path(base_dir).expanduser()
            actual = {}
            for key, expected in artifacts.items():
                if (not isinstance(key, str) or not key
                        or not isinstance(expected, str)
                        or len(expected) != 64
                        or any(char not in '0123456789abcdef'
                               for char in expected.lower())):
                    raise ValueError(
                        f'Invalid artifact hash declaration for {name!r}')
                path = Path(key).expanduser()
                if not path.is_absolute():
                    if base_path is None:
                        raise ValueError(
                            f'Relative provenance artifact {key!r} for '
                            f'{name!r} requires base_dir')
                    path = base_path / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(
                        f'Provenance artifact for {name!r} is missing: {path}')
                cache_key = str(path)
                if cache_key not in self._artifact_hash_cache:
                    self._artifact_hash_cache[cache_key] = _sha256_file(path)
                actual_hash = self._artifact_hash_cache[cache_key]
                if actual_hash != expected.lower():
                    raise ValueError(
                        f'Provenance hash mismatch for {name!r} artifact '
                        f'{path}: expected {expected.lower()}, got '
                        f'{actual_hash}')
                actual[key] = actual_hash
            representation_actual = {
                key: actual[key] for key in representation}
            digest = _representation_digest(representation_actual)
            if digest != spec.checkpoint_sha256:
                raise ValueError(
                    f'Representation artifact set for {name!r} has digest '
                    f'{digest}, but the encoder specification declares '
                    f'{spec.checkpoint_sha256}')
        self._verified_provenance.add(name)

    def _ensure_extractors(self) -> dict[str, nn.Module]:
        if self._extractors is not None:
            return self._extractors
        missing = sorted(set(self.specs) - set(self._extractor_configs))
        if missing:
            raise ValueError(
                f'Missing teacher extractor definitions for {missing}; '
                'supply runtime configs or validated pre-encoded features')
        extractors: dict[str, nn.Module] = {}
        owned: set[str] = set()
        for name, value in self._extractor_configs.items():
            if isinstance(value, nn.Module):
                extractor = value
            elif isinstance(value, Mapping):
                import fluxvla.models.backbones.visions  # noqa: F401
                options = copy.deepcopy(value)
                if isinstance(options.get('dtype'), str):
                    dtype_name = options['dtype'].lower()
                    options['dtype'] = (
                        torch.bfloat16 if dtype_name == 'bfloat16' else
                        str_to_dtype(dtype_name))
                extractor = build_vision_backbone_from_cfg(options)
                if not isinstance(extractor, nn.Module):
                    raise TypeError(
                        f'Teacher extractor {name!r} did not build an nn.Module')
                owned.add(name)
            else:
                raise TypeError(
                    f'Teacher extractor {name!r} must be a module or config')
            # C-RADIO's upstream input conditioner stores a plain ``dtype``
            # attribute that is not updated by ``Module.to()``. Keep it in the
            # encoder's selected precision so half-precision exports run
            # without changing the frozen backbone implementation.
            conditioner = getattr(
                getattr(getattr(extractor, 'model', None), 'radio_model', None),
                'input_conditioner', None)
            if (conditioner is not None and hasattr(conditioner, 'dtype')
                    and getattr(extractor, 'dtype', None) is not None):
                conditioner.dtype = extractor.dtype
            extractor.requires_grad_(False)
            extractor.eval()
            extractors[name] = extractor
        self._extractors = extractors
        self._owned = owned
        return extractors

    def validate_static_contracts(self) -> dict[str, Any]:
        """Validate identity, width, and native raster without running tensors."""
        extractors = self._ensure_extractors()
        report = {}
        for spec in self.config.encoders:
            extractor = extractors[spec.name]
            width = getattr(extractor, 'feature_dim', None)
            if width is None or int(width) != spec.feature_dim:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} provides {width!r}-wide '
                    f'features but feature_dim={spec.feature_dim}')
            feature_grid = getattr(extractor, 'feature_grid', None)
            if not callable(feature_grid):
                raise ValueError(
                    f'Teacher encoder {spec.name!r} has no feature_grid()')
            try:
                resolved = tuple(feature_grid(*spec.input_size))
            except ValueError as exc:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} cannot accept its '
                    f'configured {spec.input_size[0]}x{spec.input_size[1]} '
                    f'input: {exc}') from exc
            if tuple(int(value) for value in resolved) != spec.native_grid:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} produces native grid '
                    f'{resolved} but the configured PatchWAM native grid is '
                    f'{spec.native_grid}; implicit feature resizing is '
                    'unsupported')
            report[spec.name] = {
                'feature_dim': spec.feature_dim,
                'input_size': list(spec.input_size),
                'native_grid': list(spec.native_grid),
                'representation_id': spec.representation_id,
            }
        return report

    @staticmethod
    def _encoder_batch_bounds(extractor: nn.Module) -> tuple[int, int] | None:
        value = None
        for name in ('encoder_batch_bounds', 'batch_bounds'):
            if hasattr(extractor, name):
                value = getattr(extractor, name)
                if callable(value):
                    value = value()
                break
        if value is None:
            metadata = getattr(extractor, 'metadata', None)
            if (metadata is not None
                    and hasattr(metadata, 'min_batch')
                    and hasattr(metadata, 'max_batch')):
                value = (metadata.min_batch, metadata.max_batch)
        if value is None:
            return None
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or any(type(item) is not int for item in value)):
            raise ValueError(
                'Encoder batch bounds must be a (min_batch,max_batch) pair')
        minimum, maximum = value
        _balanced_chunk_sizes(minimum, minimum, maximum)
        return int(minimum), int(maximum)

    def _encode_images(self, extractor: nn.Module, images: torch.Tensor, *,
                       spec: TeacherEncoderSpec, layout: RasterLayout,
                       device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
        bounds = self._encoder_batch_bounds(extractor)
        sizes = ([images.shape[0]] if bounds is None else
                 _balanced_chunk_sizes(images.shape[0], *bounds))
        parts = []
        offset = 0
        for size in sizes:
            chunk = images[offset:offset + size].contiguous()
            with torch.no_grad():
                output = extractor.encode_features(chunk)
            encoded = getattr(output, 'tokens', None)
            grid = getattr(output, 'grid', None)
            if encoded is None or grid is None:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} must return '
                    'SpatialFeatureOutput-compatible tokens and grid')
            grid = tuple(int(value) for value in grid)
            if grid != layout.grid:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} produced grid {grid}, '
                    f'expected {layout.grid}; extracted features are not '
                    'interpolated or resized')
            if (not torch.is_tensor(encoded) or encoded.ndim != 3
                    or encoded.shape[0] != chunk.shape[0]
                    or encoded.shape[1] != layout.tokens_per_view
                    or encoded.shape[-1] != spec.feature_dim):
                raise ValueError(
                    f'Teacher encoder {spec.name!r} must return '
                    f'[N,{layout.tokens_per_view},{spec.feature_dim}]')
            encoded = encoded.detach().clone().to(
                device=device, dtype=dtype)
            if not torch.isfinite(encoded).all():
                raise ValueError(
                    f'Teacher encoder {spec.name!r} produced non-finite '
                    'spatial features')
            parts.append(encoded)
            offset += size
        if offset != images.shape[0] or not parts:
            raise RuntimeError('Teacher encoder batch chunking lost rasters')
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)

    def encode_context(self, frames: torch.Tensor, *, role: str,
                       layout: RasterLayout,
                       raster_valid: torch.Tensor) -> dict[str, Any]:
        """Encode one temporal context without ever mixing current and future."""
        if role not in {'current', 'future'}:
            raise ValueError('role must be current or future')
        if not isinstance(layout, RasterLayout):
            raise TypeError('layout must be a RasterLayout')
        if (not torch.is_tensor(frames) or frames.ndim != 5
                or frames.shape[2] != 3):
            raise ValueError('frames must be [B,V,3,H,W]')
        batch, views, _, height, width = frames.shape
        if views != layout.views:
            raise ValueError(
                f'frames provide {views} views but layout declares '
                f'{layout.views}')
        if (height, width) != layout.video_size:
            raise ValueError(
                f'frames are {height}x{width} but layout declares '
                f'{layout.video_size}')
        expected_valid = (batch, layout.token_count)
        if not torch.is_tensor(raster_valid) or raster_valid.shape != expected_valid:
            raise ValueError(
                f'raster_valid must be [B,{layout.token_count}]')
        raster_valid = raster_valid.to(device=frames.device, dtype=torch.bool)
        token_valid = raster_valid.reshape(
            batch, layout.views, layout.tokens_per_view)
        raster_validity = token_valid.any(-1)
        indices = raster_validity.nonzero(as_tuple=False)
        for spec in self.config.encoders:
            if spec.input_size != layout.video_size:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} requires input '
                    f'{spec.input_size[0]}x{spec.input_size[1]}, but the '
                    f'current raster is {layout.video_size[0]}x'
                    f'{layout.video_size[1]}')
            if spec.native_grid != layout.grid:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} declares native grid '
                    f'{spec.native_grid}, but PatchWAM uses {layout.grid}; '
                    'features are not interpolated')

        tokens: dict[str, torch.Tensor] = {}
        validity: dict[str, torch.Tensor] = {}
        fingerprints: dict[str, torch.Tensor] = {}
        if indices.numel() == 0:
            for spec in self.config.encoders:
                tokens[spec.name] = torch.zeros(
                    batch, layout.token_count, spec.feature_dim,
                    device=frames.device, dtype=frames.dtype)
                validity[spec.name] = raster_valid
                fingerprint = teacher_representation_fingerprint(spec)
                fingerprints[spec.name] = torch.tensor(
                    list(fingerprint), dtype=torch.uint8,
                    device=frames.device).expand(batch, -1).clone()
            return {
                'role': role,
                'tokens': tokens,
                'valid': validity,
                'fingerprints': fingerprints,
                'layout': layout.to_mapping(),
            }

        extractors = self._ensure_extractors()
        self.validate_static_contracts()
        for spec in self.config.encoders:
            parameter = next(extractors[spec.name].parameters(), None)
            device = frames.device if parameter is None else parameter.device
            dtype = frames.dtype if parameter is None else parameter.dtype
            output_tokens = torch.zeros(
                batch, layout.token_count, spec.feature_dim,
                device=device, dtype=dtype)
            sample_index = indices[:, 0]
            view_index = indices[:, 1]
            images = frames[sample_index, view_index]
            if not torch.isfinite(images).all():
                raise ValueError(
                    f'Teacher encoder {spec.name!r} received non-finite '
                    'pixels for a valid raster')
            extractor = extractors[spec.name]
            if hasattr(extractor, 'interpolate') and extractor.interpolate:
                raise ValueError(
                    f'Teacher encoder {spec.name!r} cannot be used while '
                    'interpolating extracted features')
            self._verify_provenance(
                spec.name, self._extractor_configs[spec.name])
            encoded = self._encode_images(
                extractor, images, spec=spec, layout=layout,
                device=device, dtype=dtype)
            for row in range(encoded.shape[0]):
                sample = int(sample_index[row])
                view = int(view_index[row])
                start = view * layout.tokens_per_view
                end = start + layout.tokens_per_view
                mask = token_valid[sample, view]
                if mask.any():
                    output_tokens[sample, start:end][mask] = encoded[
                        row][mask]
            tokens[spec.name] = output_tokens
            validity[spec.name] = raster_valid
            fingerprint = teacher_representation_fingerprint(spec)
            fingerprints[spec.name] = torch.tensor(
                list(fingerprint), dtype=torch.uint8,
                device=device).expand(batch, -1).clone()
        return {
            'role': role,
            'tokens': tokens,
            'valid': validity,
            'fingerprints': fingerprints,
            'layout': layout.to_mapping(),
        }

    def close(self) -> None:
        """Release resources owned by this provider and permit a later reload."""
        if self._extractors is None:
            return
        for name in self._owned:
            extractor = self._extractors.get(name)
            close = getattr(extractor, 'close', None)
            if callable(close):
                close()
        self._extractors = None
        self._owned = set()


__all__ = ['TeacherFeatureProvider']
