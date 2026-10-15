
"""C-RADIO engine sidecar and its conversion to an engine contract.

One record binds a serialized engine to the C-RADIO graph it was built from:
variant and feature width, patch/grid geometry, the fixed image resolution,
the input/output tensor names and dtypes, the builder precision, the batch
profile and the TensorRT version that produced it. ``precision`` is a builder
choice and is independent of ``input_dtype``/``output_dtype``, which describe
the graph boundary.

:meth:`EngineMetadata.contract` converts the record into the model-independent
:class:`~.metadata.EngineContract` the runtime executes against, so the
sidecar schema stays RADIO-specific while execution stays generic.

This module imports the standard library only.
"""

import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .metadata import (DYNAMIC_DIM, DTYPE_NAMES, EngineContract,
                       OptimizationProfile, TensorSpec)

#: Bumped when the on-disk record changes shape incompatibly.
SCHEMA_VERSION = 1

#: Default tensor names written by the C-RADIO export.
DEFAULT_INPUT_NAME = 'pixel_values'
DEFAULT_OUTPUT_NAME = 'tokens'

#: Builder precisions. BF16 is a tactic selection like FP16; INT8 is not
#: calibrated by this pipeline.
PRECISIONS: Tuple[str, ...] = ('fp32', 'fp16', 'bf16')

#: Spatial-output layouts the C-RADIO wrapper normalizes.
OUTPUT_LAYOUTS: Tuple[str, ...] = ('NLD', 'NCHW', 'NHWC')

#: ``[B, C, H, W]`` C-RADIO input always carries three RGB channels.
IMAGE_CHANNELS = 3


def _positive_int(field_name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('C-RADIO engine metadata '
                         f'{field_name} must be a positive integer; got '
                         f'{value!r}')
    return int(value)


def _non_empty_str(field_name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('C-RADIO engine metadata '
                         f'{field_name} must be a non-empty string; got '
                         f'{value!r}')
    return str(value)


def tensorrt_major(version: str) -> int:
    """Major component of a TensorRT version string."""
    text = _non_empty_str('tensorrt_version', version).strip()
    head = text.split('.', 1)[0]
    if not head.isdigit():
        raise ValueError(
            f'The TensorRT version {version!r} does not start with a major '
            'number')
    return int(head)


@dataclass(frozen=True)
class EngineMetadata:
    """Sidecar record describing one serialized C-RADIO TensorRT engine.

    Geometry is fixed: ``input_height``/``input_width`` are the only image
    sizes the engine accepts and they must be patch-aligned. The batch axis is
    the single dynamic axis, bounded by
    ``min_batch``/``opt_batch``/``max_batch``.
    """

    # -- identity and geometry (required) --------------------------------
    variant: str
    feature_dim: int
    patch_size: int
    resolution_step: int
    input_height: int
    input_width: int
    grid_height: int
    grid_width: int
    tensorrt_version: str
    # -- build and IO contract -------------------------------------------
    precision: str = 'fp32'
    input_dtype: str = 'float32'
    output_dtype: str = 'float32'
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 1
    input_name: str = DEFAULT_INPUT_NAME
    output_name: str = DEFAULT_OUTPUT_NAME
    output_layout: str = 'NLD'
    # -- per-layer precision policy (optional) ---------------------------
    #: Precision constraints mode the layers below were enforced with
    #: (``obey``/``prefer``), or ``None`` when no layer was constrained.
    precision_constraints: Optional[str] = None
    #: Human-readable summary of the requested constraints and how many layers
    #: each matched. These describe the build, not the graph boundary.
    layer_precisions: Optional[str] = None
    # -- provenance (optional) -------------------------------------------
    source_onnx: Optional[str] = None
    onnx_sha256: Optional[str] = None
    external_data: Optional[str] = None
    external_data_sha256: Optional[str] = None
    workspace_bytes: Optional[int] = None
    created_utc: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_str('variant', self.variant)
        _positive_int('feature_dim', self.feature_dim)
        _positive_int('patch_size', self.patch_size)
        _positive_int('resolution_step', self.resolution_step)
        input_height = _positive_int('input_height', self.input_height)
        input_width = _positive_int('input_width', self.input_width)
        grid_height = _positive_int('grid_height', self.grid_height)
        grid_width = _positive_int('grid_width', self.grid_width)
        _non_empty_str('tensorrt_version', self.tensorrt_version)
        tensorrt_major(self.tensorrt_version)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError('Unsupported C-RADIO engine metadata '
                             f'schema_version {self.schema_version!r}; this '
                             f'runtime reads {SCHEMA_VERSION}')
        if int(self.resolution_step) % int(self.patch_size):
            raise ValueError(
                'C-RADIO engine metadata resolution_step must be a patch-size '
                f'multiple; got step {self.resolution_step} for patch '
                f'{self.patch_size}')
        if input_height % int(self.resolution_step) or (
                input_width % int(self.resolution_step)):
            raise ValueError(
                'C-RADIO engine metadata geometry must be aligned to the '
                f'resolution step: {input_height}x{input_width} is not a '
                f'multiple of {self.resolution_step}')
        expected_grid = (input_height // int(self.patch_size),
                         input_width // int(self.patch_size))
        if (grid_height, grid_width) != expected_grid:
            raise ValueError(
                'C-RADIO engine metadata grid does not match the geometry: '
                f'{grid_height}x{grid_width} declared but '
                f'{expected_grid[0]}x{expected_grid[1]} follows from '
                f'{input_height}x{input_width} at patch {self.patch_size}')
        if self.precision not in PRECISIONS:
            raise ValueError(
                'C-RADIO engine metadata precision must be one of '
                f'{PRECISIONS}; got {self.precision!r}')
        for dtype_field in ('input_dtype', 'output_dtype'):
            value = getattr(self, dtype_field)
            if value not in DTYPE_NAMES:
                raise ValueError(
                    f'C-RADIO engine metadata {dtype_field} must be one of '
                    f'{DTYPE_NAMES}; got {value!r}. Tensor dtypes describe '
                    'the graph and are not derived from the builder '
                    'precision.')
        if self.output_layout not in OUTPUT_LAYOUTS:
            raise ValueError(
                'C-RADIO engine metadata output_layout must be one of '
                f'{OUTPUT_LAYOUTS}; got {self.output_layout!r}')
        for policy_field in ('precision_constraints', 'layer_precisions'):
            value = getattr(self, policy_field)
            if value is not None:
                _non_empty_str(policy_field, value)
        low = _positive_int('min_batch', self.min_batch)
        opt = _positive_int('opt_batch', self.opt_batch)
        high = _positive_int('max_batch', self.max_batch)
        if not low <= opt <= high:
            raise ValueError(
                'C-RADIO engine metadata batch bounds must satisfy '
                f'min <= opt <= max; got {low}, {opt}, {high}')
        _non_empty_str('input_name', self.input_name)
        _non_empty_str('output_name', self.output_name)
        if self.input_name == self.output_name:
            raise ValueError(
                'C-RADIO engine metadata input_name and output_name must '
                'differ; the runtime addresses them independently')

    # -- derived geometry -------------------------------------------------
    @property
    def grid(self) -> Tuple[int, int]:
        return (int(self.grid_height), int(self.grid_width))

    @property
    def image_size(self) -> Tuple[int, int]:
        return (int(self.input_height), int(self.input_width))

    def input_shape(self, batch: int) -> Tuple[int, int, int, int]:
        """``[B, 3, H, W]`` engine input for ``batch`` images."""
        return (int(batch), IMAGE_CHANNELS, int(self.input_height),
                int(self.input_width))

    def output_shape(self, batch: int) -> Tuple[int, ...]:
        """Engine output for ``batch`` images in the declared layout."""
        grid_height, grid_width = self.grid
        tokens = grid_height * grid_width
        if self.output_layout == 'NLD':
            return (int(batch), tokens, int(self.feature_dim))
        if self.output_layout == 'NCHW':
            return (int(batch), int(self.feature_dim), grid_height, grid_width)
        return (int(batch), grid_height, grid_width, int(self.feature_dim))

    # -- generic contract -------------------------------------------------
    def contract(self) -> EngineContract:
        """The model-independent engine boundary this record describes."""
        inputs = (TensorSpec(
            name=self.input_name,
            dtype=self.input_dtype,
            shape=self.input_shape(DYNAMIC_DIM)), )
        outputs = (TensorSpec(
            name=self.output_name,
            dtype=self.output_dtype,
            shape=self.output_shape(DYNAMIC_DIM)), )
        profile = OptimizationProfile(
            tensor=self.input_name,
            min_shape=self.input_shape(self.min_batch),
            opt_shape=self.input_shape(self.opt_batch),
            max_shape=self.input_shape(self.max_batch))
        return EngineContract(
            inputs=inputs, outputs=outputs, profiles=(profile, ))

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Full record, including ``None`` provenance, so it round-trips."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'EngineMetadata':
        """Strictly parse a record, rejecting unknown keys and bad types."""
        if not isinstance(data, Mapping):
            raise ValueError('C-RADIO engine metadata must be a mapping; got '
                             f'{type(data).__name__}')
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f'C-RADIO engine metadata has unknown field(s) {unknown}; '
                f'expected only {sorted(known)}')
        missing = sorted(
            field.name for field in fields(cls) if field.default is MISSING
            and field.default_factory is MISSING and field.name not in data)
        if missing:
            raise ValueError(
                f'C-RADIO engine metadata is missing required field(s) '
                f'{missing}')
        return cls(**dict(data))

    @classmethod
    def load(cls, path: Any) -> 'EngineMetadata':
        """Read a JSON sidecar from disk."""
        metadata_path = Path(path).expanduser()
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f'C-RADIO engine metadata sidecar not found: {metadata_path}')
        try:
            data = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(
                f'C-RADIO engine metadata {metadata_path} is not valid JSON: '
                f'{error}') from error
        return cls.from_dict(data)

    def save(self, path: Any) -> Path:
        """Write the record as JSON, creating parent directories."""
        metadata_path = Path(path).expanduser()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n')
        return metadata_path


def default_metadata_path(engine_path: Any) -> Path:
    """Sidecar path the runtime looks for next to ``foo.engine``."""
    engine_path = Path(engine_path).expanduser()
    return Path(f'{engine_path}.json')


__all__ = [
    'DEFAULT_INPUT_NAME',
    'DEFAULT_OUTPUT_NAME',
    'IMAGE_CHANNELS',
    'OUTPUT_LAYOUTS',
    'PRECISIONS',
    'SCHEMA_VERSION',
    'EngineMetadata',
    'default_metadata_path',
    'tensorrt_major',
]
