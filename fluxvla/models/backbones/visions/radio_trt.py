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
"""C-RADIOv4 spatial features served by a serialized TensorRT engine.

The accelerated sibling of :class:`RadioViTBackbone`. It presents the same
frozen ``[B, N, D]`` token contract through ``FrozenVisionFeatures``, so a
consumer that already accepts the generic vision interface can switch between
the two for a matching compiled geometry.

This class owns the C-RADIO side of the boundary: the ``[-1, 1]`` to
``[0, 1]`` conversion, the fixed-geometry check, the variant/feature-width
cross-check and the normalization of the engine's spatial layout into
``[B, N, D]``. Execution is delegated to the model-independent
``base_trt.TensorRTEngine``, which holds no C-RADIO knowledge.
"""

from typing import Any, Mapping, Optional, Tuple

import torch

from fluxvla.engines import VISION_BACKBONES

from .base_trt import TensorRTEngine, torch_dtype_for
from .base_trt.radio_metadata import (DEFAULT_INPUT_NAME, EngineMetadata,
                                      default_metadata_path)
from .frozen_vision import (FrozenVisionFeatures, SpatialFeatureOutput,
                            to_unit_range)
from .radio_vit import RADIO_V4_FEATURE_DIMS, normalize_spatial_tokens


def _as_metadata(metadata: Any, *, field: str = 'metadata') -> EngineMetadata:
    if isinstance(metadata, EngineMetadata):
        return metadata
    if isinstance(metadata, Mapping):
        return EngineMetadata.from_dict(metadata)
    raise ValueError(f'{field} must be an EngineMetadata or a mapping; got '
                     f'{type(metadata).__name__}')


@VISION_BACKBONES.register_module()
class RadioViTBackboneTensorRT(FrozenVisionFeatures):
    """Frozen C-RADIOv4 tokens from a serialized TensorRT engine.

    Args:
        vision_backbone_id: Registered variant, e.g. ``c-radio_v4-h``.
            Defaults to the variant recorded in the engine metadata; when both
            are given they must agree.
        engine: A ready execution handle exposing ``run(tensor)``. Mutually
            exclusive with ``engine_path``.
        engine_path: Serialized ``.engine`` file. Loaded lazily on the first
            encode, so configuration and construction need no TensorRT.
        metadata: C-RADIO engine metadata, as a record or a mapping.
        metadata_path: Sidecar path. Defaults to ``<engine_path>.json``.
        device: CUDA device index or ``torch.device`` for the engine.
        logger_severity: TensorRT logger severity for the lazy load.
        default_image_resolution: ``(C, H, W)`` reported by ``num_patches``.
            Must equal the engine's fixed geometry, which is the only size the
            engine accepts.
    """

    def __init__(
        self,
        vision_backbone_id: Optional[str] = None,
        *,
        engine: Optional[Any] = None,
        engine_path: Optional[str] = None,
        metadata: Optional[Any] = None,
        metadata_path: Optional[str] = None,
        device: Optional[Any] = None,
        logger_severity: str = 'WARNING',
        default_image_resolution: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if (engine is None) == (engine_path is None):
            raise ValueError('Provide exactly one of engine or engine_path')
        if metadata is not None and metadata_path is not None:
            raise ValueError(
                'Provide at most one of metadata or metadata_path')
        resolved = self._resolve_metadata(
            engine=engine,
            engine_path=engine_path,
            metadata=metadata,
            metadata_path=metadata_path)
        variant = vision_backbone_id or resolved.variant
        if variant not in RADIO_V4_FEATURE_DIMS:
            raise ValueError(
                f'Unsupported C-RADIO version {variant!r}; expected one of '
                f'{sorted(RADIO_V4_FEATURE_DIMS)}')
        if variant != resolved.variant:
            raise ValueError(
                f'vision_backbone_id {variant!r} does not match the engine '
                f'metadata variant {resolved.variant!r}')
        expected_dim = RADIO_V4_FEATURE_DIMS[variant]
        if int(resolved.feature_dim) != int(expected_dim):
            raise ValueError(
                f'The engine metadata declares {resolved.feature_dim}-wide '
                f'features but {variant!r} produces {expected_dim}; rebuild '
                'the engine from the matching C-RADIO export')
        geometry = default_image_resolution or (3, resolved.input_height,
                                                resolved.input_width)
        if tuple(int(value)
                 for value in geometry) != (3, resolved.input_height,
                                            resolved.input_width):
            raise ValueError(
                'default_image_resolution '
                f'{tuple(geometry)} does not match the engine geometry '
                f'(3, {resolved.input_height}, {resolved.input_width}); '
                'num_patches is derived from it, so it must be the compiled '
                'input size')
        if engine is not None:
            self._check_injected_engine(engine, resolved)
        super().__init__(
            variant,
            feature_dim=int(resolved.feature_dim),
            resolution_step=int(resolved.resolution_step),
            default_image_resolution=geometry)
        self.metadata = resolved
        self.engine_path = None if engine_path is None else str(engine_path)
        self._device = device
        self._logger_severity = logger_severity
        self._engine = engine
        self._owns_engine = engine is None
        self.freeze()

    @staticmethod
    def _check_injected_engine(engine: Any, metadata: EngineMetadata) -> None:
        contract = getattr(engine, 'contract', None)
        if contract is None:
            return
        expected = metadata.contract()
        if contract.to_dict() != expected.to_dict():
            raise ValueError(
                'The injected engine contract does not match the C-RADIO '
                'metadata; pass the engine built for this sidecar')

    @staticmethod
    def _resolve_metadata(*, engine: Optional[Any], engine_path: Optional[str],
                          metadata: Optional[Any],
                          metadata_path: Optional[str]) -> EngineMetadata:
        explicit: Optional[EngineMetadata] = None
        if metadata is not None:
            explicit = _as_metadata(metadata)
        elif metadata_path is not None:
            explicit = EngineMetadata.load(metadata_path)
        if explicit is not None:
            return explicit
        if engine is not None:
            raise ValueError(
                'An injected engine carries no C-RADIO metadata; pass '
                'metadata or metadata_path alongside it')
        assert engine_path is not None
        return EngineMetadata.load(default_metadata_path(engine_path))

    @classmethod
    def from_engine(cls, engine_path: str,
                    **kwargs) -> 'RadioViTBackboneTensorRT':
        """Build from a serialized engine and its sidecar metadata."""
        kwargs.setdefault('engine_path', engine_path)
        return cls(**kwargs)

    # -- lazy runtime -----------------------------------------------------
    @property
    def engine(self) -> Any:
        """The execution handle, loading TensorRT on first access."""
        return self._ensure_engine()

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            if self.engine_path is None:  # pragma: no cover - guarded
                raise RuntimeError('No TensorRT engine is configured')
            self._engine = TensorRTEngine(
                self.engine_path,
                self.metadata.contract(),
                device=self._device,
                expected_tensorrt_version=self.metadata.tensorrt_version,
                logger_severity=self._logger_severity)
        return self._engine

    def close(self) -> None:
        """Release the engine this backbone owns (never an injected one)."""
        if self._engine is not None and self._owns_engine:
            self._engine.close()
            self._engine = None

    def describe(self) -> dict:
        """Metadata plus the engine report when one is loaded."""
        report = {'metadata': self.metadata.to_dict()}
        if self._engine is not None:
            report['engine'] = self._engine.describe()
        return report

    # -- geometry ---------------------------------------------------------
    def feature_grid(self, height: int, width: int) -> Tuple[int, int]:
        """Grid of the engine's single compiled geometry."""
        engine_h, engine_w = self.metadata.image_size
        if (int(height), int(width)) != (engine_h, engine_w):
            raise ValueError(
                'This C-RADIO TensorRT engine is compiled for a fixed '
                f'{engine_h}x{engine_w} input (batch is the only dynamic '
                f'axis); got {int(height)}x{int(width)}. Export and build an '
                'engine for that geometry instead of relying on an implicit '
                'resize.')
        return self.metadata.grid

    # -- encoding ---------------------------------------------------------
    @torch.no_grad()
    def encode_features(self, images: torch.Tensor) -> SpatialFeatureOutput:
        """Encode ``[B, 3, H, W]`` RGB in ``[-1, 1]`` to ``[B, N, D]``.

        Runs under ``torch.no_grad`` (frozen weights) but not
        ``torch.inference_mode``: the returned tokens are ordinary tensors, so
        a downstream trainable projector can still save them for backward.

        The engine enqueue does not synchronize the host, but the ``[-1, 1]``
        validation inherited from ``to_unit_range`` reads tensor values and
        therefore does synchronize on CUDA. That cost is real and is not
        hidden.
        """
        unit = to_unit_range(images, 'C-RADIO TensorRT')
        grid = self.feature_grid(int(unit.shape[-2]), int(unit.shape[-1]))
        inputs = unit
        if self._device is not None:
            inputs = inputs.to(device=self._device)
        inputs = inputs.to(dtype=torch_dtype_for(self.metadata.input_dtype))
        inputs = inputs.contiguous()
        raw = self._ensure_engine().run(inputs)
        tokens = normalize_spatial_tokens(raw, self.feature_dim,
                                          self.metadata.output_layout)
        if tokens.ndim != 3 or tokens.shape[0] != inputs.shape[0]:
            raise ValueError('C-RADIO TensorRT must return [B, N, D] tokens')
        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f'The TensorRT engine produced {tokens.shape[-1]}-wide '
                f'features; {self.feature_dim} was configured')
        return SpatialFeatureOutput(tokens=tokens, grid=grid)

    @property
    def precision(self) -> str:
        return self.metadata.precision

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Preferred half precision for this engine's build.

        ``float16`` for an FP16 build, ``bfloat16`` for a BF16 build, and the
        framework ``bfloat16`` default for an FP32 one. This is a preference
        for a consumer that wants a half-precision variant; the engine's real
        boundary dtypes are ``metadata.input_dtype``/``output_dtype``.
        """
        if self.metadata.precision == 'fp16':
            return torch.float16
        return torch.bfloat16


def build_radio_trt_backbone(engine_path: str,
                             **kwargs) -> RadioViTBackboneTensorRT:
    """Convenience factory for a serialized C-RADIO TensorRT engine."""
    return RadioViTBackboneTensorRT.from_engine(engine_path, **kwargs)


__all__ = [
    'DEFAULT_INPUT_NAME',
    'RadioViTBackboneTensorRT',
    'build_radio_trt_backbone',
]
