#!/usr/bin/env python3
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
"""Generic offline ONNX export and TensorRT build helpers.

These helpers work on a model-independent
:class:`~fluxvla.models.backbones.visions.base_trt.metadata.EngineContract`:
they trace a module to ONNX at the contract's boundary, optionally
externalize the weights, verify the graph against the contract and its
recorded hashes, and compile it to a serialized engine. ``onnx`` and
``tensorrt`` are imported only inside the functions that need them.

A build can also constrain the precision of individual network layers
(``layer_precisions``), which is how an FP16 build keeps selected layers, for
example every ``LayerNormalization``, in FP32. The constraint is only honoured
when it is requested strictly (``precision_constraints='obey'``), and a pattern
that matches no layer fails the build instead of leaving it silently
unconstrained.

Model-specific orchestration (which module to trace, how to preprocess) lives
in the caller, for example :mod:`tools.c_radio_export`.
"""

import contextlib
import fnmatch
import hashlib
import logging
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from fluxvla.models.backbones.visions.base_trt import (
    DYNAMIC_DIM, EngineContract, TensorSpec, require_tensorrt, trt_dtype_name,
    validate_engine_contract)
from fluxvla.models.backbones.visions.base_trt.engine import TensorRTEngine

logger = logging.getLogger(__name__)

#: Default external-data sidecar suffix.
DEFAULT_EXTERNAL_DATA_SUFFIX = '.data'

#: ONNX element type name to contract dtype name.
_ONNX_DTYPE_NAMES: Dict[str, str] = {
    'FLOAT': 'float32',
    'FLOAT16': 'float16',
    'BFLOAT16': 'bfloat16',
    'INT32': 'int32',
    'INT8': 'int8',
    'BOOL': 'bool',
}

#: Builder precisions this helper can request. FP32 and FP16 are universally
#: available; BF16 depends on the TensorRT build and the device.
_PRECISIONS: Tuple[str, ...] = ('fp32', 'fp16', 'bf16')

#: BF16 kernels need Ampere or newer (CUDA compute capability 8.0+).
_BF16_MIN_COMPUTE_CAPABILITY: Tuple[int, int] = (8, 0)

#: How a per-layer precision constraint is enforced. ``obey`` turns a
#: constraint into a requirement (TensorRT fails the build rather than
#: overriding it), ``prefer`` lets TensorRT fall back to the fastest available
#: implementation with a warning, and ``none`` sets neither builder flag.
PRECISION_CONSTRAINTS: Tuple[str, ...] = ('none', 'prefer', 'obey')

#: Profiling verbosity levels accepted by :func:`build_engine`.
PROFILING_LEVELS: Tuple[str, ...] = ('none', 'layer_names_only', 'detailed')

#: Precision labels accepted in a ``PATTERN:DTYPE`` layer override, mapped to
#: the short label recorded in the artifacts.
_LAYER_PRECISION_LABELS: Dict[str, str] = {
    'fp32': 'fp32',
    'float32': 'fp32',
    'fp16': 'fp16',
    'float16': 'fp16',
    'bf16': 'bf16',
    'bfloat16': 'bf16',
}

#: Canonical precision label to the contract dtype name.
_LAYER_PRECISION_DTYPES: Dict[str, str] = {
    'fp32': 'float32',
    'fp16': 'float16',
    'bf16': 'bfloat16',
}

#: Contract dtype to the ``trt.DataType`` member that names it.
_LAYER_DTYPE_MEMBERS: Dict[str, str] = {
    'float32': 'FLOAT',
    'float16': 'HALF',
    'bfloat16': 'BF16',
}

#: Layer names a build record keeps per requested pattern.
_LAYER_NAME_RECORD_LIMIT = 64


@dataclass(frozen=True)
class LayerPrecision:
    """One ``PATTERN:DTYPE`` per-layer precision constraint.

    ``pattern`` is matched against the *name* of each layer in the parsed
    network, with ``*`` and ``?`` wildcards and case-sensitive comparison, the
    same way ``trtexec --layerPrecisions`` matches. ``precision`` is the
    canonical label (``fp32``) and ``dtype`` the contract dtype name.
    """

    pattern: str
    precision: str
    dtype: str

    @property
    def spec(self) -> str:
        """The constraint as the user wrote it, normalized."""
        return f'{self.pattern}:{self.precision}'


def parse_layer_precision(spec: Any) -> LayerPrecision:
    """Parse ``PATTERN:DTYPE`` into a :class:`LayerPrecision`.

    ``*/LayerNormalization:fp32`` requests FP32 for every layer whose name ends
    in ``LayerNormalization``. The dtype is the last colon-separated field, so
    a pattern may contain colons. Only floating-point precisions are accepted:
    INT8 needs calibration this pipeline does not implement.
    """
    if isinstance(spec, LayerPrecision):
        return spec
    text = str(spec).strip()
    pattern, separator, label = text.rpartition(':')
    pattern = pattern.strip()
    label = label.strip().lower()
    if not separator or not pattern:
        raise ValueError(
            f'Layer precision {spec!r} must be written PATTERN:DTYPE, for '
            'example "*/LayerNormalization:fp32": the dtype follows the last '
            'colon and the pattern must not be empty')
    if label not in _LAYER_PRECISION_LABELS:
        raise ValueError(
            f'Layer precision dtype {label!r} is not supported; expected one '
            f'of {sorted(_LAYER_PRECISION_LABELS)}. INT8 needs calibration '
            'that this pipeline does not implement')
    canonical = _LAYER_PRECISION_LABELS[label]
    return LayerPrecision(
        pattern=pattern,
        precision=canonical,
        dtype=_LAYER_PRECISION_DTYPES[canonical])


def _layer_type_name(layer: Any) -> str:
    """A layer's TensorRT type without the ``LayerType.`` prefix."""
    return str(getattr(layer, 'type', 'unknown')).split('.')[-1]


def _layer_dtype(trt: Any, dtype: str) -> Any:
    """The ``trt.DataType`` member for a contract dtype name."""
    member = _LAYER_DTYPE_MEMBERS[dtype]
    found = getattr(getattr(trt, 'DataType', None), member, None)
    if found is None:
        raise RuntimeError(
            f'TensorRT {getattr(trt, "__version__", "unknown")} exposes no '
            f'trt.DataType.{member}, so a layer cannot be constrained to '
            f'{dtype}')
    return found


def apply_layer_precisions(
        trt: Any, network: Any,
        overrides: Sequence[LayerPrecision]) -> List[Dict[str, Any]]:
    """Constrain every layer whose name matches a requested pattern.

    Returns one entry per requested pattern, in request order, describing what
    was constrained. A pattern that matches no layer raises: the whole point of
    the feature is that a constraint is applied or the build stops, so an
    unmatched pattern must not quietly produce an unconstrained engine. When
    two patterns match one layer the later request wins, and both entries still
    list that layer.

    The constraint is applied through ``ILayer::setPrecision``, which matches
    ``trtexec --layerPrecisions``. That API is deprecated from TensorRT 10.12
    in favour of strong typing, so the deprecation warning it raises is
    suppressed here; the build is still weak typing and the constraint still
    takes effect (see ``docs/c-radio-v4-tensorrt.md``).
    """
    overrides = list(overrides)
    if not overrides:
        return []
    layers = [
        network.get_layer(index) for index in range(int(network.num_layers))
    ]
    if not layers:
        requested = [override.spec for override in overrides]
        raise RuntimeError(
            'The parsed network has no layers, so no layer precision '
            f'constraint can be applied; requested {requested}')
    applied: List[Dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        for override in overrides:
            matched = [
                layer for layer in layers
                if fnmatch.fnmatchcase(str(layer.name), override.pattern)
            ]
            if not matched:
                raise ValueError(
                    f'The layer precision pattern {override.pattern!r} of '
                    f'{override.spec!r} matched none of the {len(layers)} '
                    'layers in the parsed network, so it would silently leave '
                    'this build unconstrained. Layer names come from the ONNX '
                    'nodes: list them, and count a pattern, with '
                    'tools/trt_layer_report.py before building.')
            dtype = _layer_dtype(trt, override.dtype)
            for layer in matched:
                layer.precision = dtype
            names = sorted(str(layer.name) for layer in matched)
            applied.append({
                'pattern':
                override.pattern,
                'precision':
                override.precision,
                'dtype':
                override.dtype,
                'matched_layers':
                len(matched),
                'layer_types':
                sorted({_layer_type_name(l)
                        for l in matched}),
                'layer_names':
                names[:_LAYER_NAME_RECORD_LIMIT],
                'truncated':
                len(names) > _LAYER_NAME_RECORD_LIMIT,
            })
    return applied


def summarize_layer_precisions(applied: Sequence[Dict[str, Any]],
                               network_layers: int) -> str:
    """A one-line, sidecar-sized summary of applied layer constraints."""
    return '; '.join(f'{entry["pattern"]}:{entry["precision"]} matched '
                     f'{entry["matched_layers"]} of {network_layers} layers'
                     for entry in applied)


@contextlib.contextmanager
def capture_std_streams(path: Any) -> Iterator[Path]:
    """Duplicate stdout and stderr into ``path`` for the duration of the block.

    TensorRT's C++ logger writes to the process streams rather than to a Python
    logger, so builder diagnostics can only be kept by redirecting the file
    descriptors. The text is appended to any existing file and the original
    descriptors are restored on exit, including on an exception.
    """
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        handle = open(target, 'ab')
        try:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
            try:
                yield target
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
        finally:
            handle.close()
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def sha256_file(path: Any) -> str:
    """SHA-256 of a file, used to prove which ONNX an engine came from."""
    digest = hashlib.sha256()
    with open(Path(path).expanduser(), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    """Timestamp recorded in build provenance."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _require_onnx(what: str) -> Any:
    try:
        import onnx  # noqa: PLC0415 - intentionally lazy
    except ImportError as error:
        raise RuntimeError(
            f'The {what} step needs the optional ``onnx`` package, which is '
            'not installed. Install onnx in the offline build environment; '
            'the inference runtime does not require it. Original error: '
            f'{error!r}') from error
    return onnx


def convert_onnx_to_external_data(onnx_path: Any,
                                  *,
                                  location: Optional[str] = None,
                                  size_threshold: int = 1024) -> Optional[str]:
    """Move large ONNX initializers into a sidecar data file.

    Returns the data file name when one was written, else ``None``. Tensors at
    or below ``size_threshold`` bytes stay inline.

    The graph and its data file are built in a temporary directory next to the
    target and only moved into place once both are complete.
    ``onnx.save_model`` *appends* to an existing external-data file, so writing
    in place would leave the previous data as an unused prefix and double the
    artifact on every repeat conversion. Staging failures leave the original
    files untouched. The final two replacements require exclusive offline use.
    """
    onnx = _require_onnx('ONNX external-data')
    path = Path(onnx_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'ONNX model not found: {path}')
    data_location = location or f'{path.name}{DEFAULT_EXTERNAL_DATA_SUFFIX}'
    model = onnx.load(str(path), load_external_data=True)
    with tempfile.TemporaryDirectory(dir=str(path.parent)) as staging:
        staged_graph = Path(staging) / path.name
        try:
            onnx.save_model(
                model,
                str(staged_graph),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_location,
                size_threshold=int(size_threshold),
                convert_attribute=False)
        except TypeError:  # older onnx without convert_attribute
            onnx.save_model(
                model,
                str(staged_graph),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_location,
                size_threshold=int(size_threshold))
        staged_data = Path(staging) / data_location
        data_path = path.parent / data_location
        if staged_data.is_file():
            os.replace(str(staged_data), str(data_path))
        os.replace(str(staged_graph), str(path))
    return data_location if data_path.is_file() else None


def verify_onnx_identity(
    onnx_path: Any,
    *,
    onnx_sha256: Optional[str] = None,
    external_data: Optional[str] = None,
    external_data_sha256: Optional[str] = None,
) -> None:
    """Fail when the ONNX files are not the ones a record describes.

    An integrity check for a stale graph, not a signature: it only runs for
    the hashes a caller supplies.
    """
    path = Path(onnx_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'ONNX model not found: {path}')
    if onnx_sha256:
        actual = sha256_file(path)
        if actual != onnx_sha256:
            raise ValueError(
                f'The ONNX graph {path} does not match the recorded sha256 '
                f'({actual} != {onnx_sha256}); it was re-exported or edited '
                'after the record was written.')
    if external_data:
        data_path = path.parent / external_data
        if not data_path.is_file():
            raise FileNotFoundError(
                f'The record names external data {external_data!r} but '
                f'{data_path} does not exist')
        if external_data_sha256:
            actual = sha256_file(data_path)
            if actual != external_data_sha256:
                raise ValueError(
                    f'The ONNX external data {data_path} does not match the '
                    'recorded sha256; it was rewritten after the record was '
                    'written')
    elif external_data_sha256:
        raise ValueError('external_data_sha256 was recorded without '
                         'external_data; the record is inconsistent')


def _onnx_specs(onnx: Any, values: Sequence[Any]) -> List[TensorSpec]:
    specs: List[TensorSpec] = []
    for value in values:
        kind = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
        dtype = _ONNX_DTYPE_NAMES.get(str(kind))
        if dtype is None:
            raise ValueError(
                f'ONNX tensor {value.name!r} uses element type {kind!r}; this '
                f'pipeline supports only {sorted(_ONNX_DTYPE_NAMES)}')
        dims: List[int] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.HasField('dim_value') and int(dim.dim_value) > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(DYNAMIC_DIM)
        specs.append(
            TensorSpec(name=str(value.name), dtype=dtype, shape=tuple(dims)))
    return specs


def read_onnx_graph(
    onnx_path: Any, ) -> Tuple[List[TensorSpec], List[TensorSpec], List[str]]:
    """Read an ONNX graph's boundary and external-data references."""
    onnx = _require_onnx('ONNX verification')
    path = Path(onnx_path).expanduser()
    model = onnx.load(str(path), load_external_data=False)
    try:
        # Best effort: shape inference resolves axes the tracer could not
        # (for example a token count that follows from a static raster).
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # noqa: BLE001 - inference is optional
        logger.debug('ONNX shape inference failed for %s', path)
    inputs = _onnx_specs(onnx, list(model.graph.input))
    outputs = _onnx_specs(onnx, list(model.graph.output))
    references = sorted({
        entry.value
        for tensor in list(model.graph.initializer)
        for entry in tensor.external_data if entry.key == 'location'
    })
    return inputs, outputs, references


def _ordered_by_contract(specs: Sequence[TensorSpec],
                         declared: Sequence[TensorSpec],
                         kind: str) -> List[TensorSpec]:
    by_name = {spec.name: spec for spec in specs}
    missing = [spec.name for spec in declared if spec.name not in by_name]
    if missing:
        raise ValueError(f'The ONNX graph has no {kind} named {missing}; the '
                         f'contract declares {[s.name for s in declared]}')
    return [by_name[spec.name] for spec in declared]


def verify_onnx_graph(
    onnx_path: Any,
    contract: EngineContract,
    *,
    onnx_sha256: Optional[str] = None,
    external_data: Optional[str] = None,
    external_data_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Check that a written ONNX graph matches ``contract`` and its hashes.

    Validates the recorded hashes, then the boundary: the declared inputs and
    outputs must exist with matching names, dtypes, ranks and axes (static
    axes exact, dynamic axes inside the profile). Extra or missing tensors
    fail, as does an external-data reference the graph does not use.
    """
    verify_onnx_identity(
        onnx_path,
        onnx_sha256=onnx_sha256,
        external_data=external_data,
        external_data_sha256=external_data_sha256)
    inputs, outputs, references = read_onnx_graph(onnx_path)
    if len(inputs) != len(contract.inputs) or len(outputs) != len(
            contract.outputs):
        raise ValueError(
            'The ONNX graph has '
            f'{len(inputs)} inputs and {len(outputs)} outputs; the contract '
            f'declares {contract.input_names} and {contract.output_names}')
    ordered_inputs = _ordered_by_contract(inputs, contract.inputs, 'input')
    ordered_outputs = _ordered_by_contract(outputs, contract.outputs, 'output')
    unresolved = []
    for kind, declared, actual in (('input', contract.inputs, ordered_inputs),
                                   ('output', contract.outputs,
                                    ordered_outputs)):
        for spec, found in zip(declared, actual):
            if spec.dtype != found.dtype:
                raise ValueError(
                    f'ONNX {kind} {spec.name!r} has dtype {found.dtype!r} but '
                    f'the contract declares {spec.dtype!r}')
            if spec.rank != found.rank:
                raise ValueError(
                    f'ONNX {kind} {spec.name!r} has rank {found.rank} but the '
                    f'contract declares {spec.rank}')
            profile = contract.profile_for(spec.name)
            for axis, dim in enumerate(spec.shape):
                present = found.shape[axis]
                if present == DYNAMIC_DIM:
                    # Shape inference could not resolve this axis. The
                    # deserialized engine is checked against the contract, and
                    # the runtime validates the resolved output shape, so an
                    # unresolved axis is recorded rather than trusted.
                    if dim != DYNAMIC_DIM:
                        unresolved.append(f'{spec.name}[{axis}]')
                    continue
                low, _, high = (
                    profile.bounds(axis) if profile is not None else
                    (dim, dim, dim))
                if dim == DYNAMIC_DIM:
                    if not low <= present <= high:
                        raise ValueError(
                            f'ONNX {kind} {spec.name!r} axis {axis} is '
                            f'{present}, outside the declared profile bounds '
                            f'[{low}, {high}]')
                elif present != dim:
                    raise ValueError(
                        f'ONNX {kind} {spec.name!r} axis {axis} is {present} '
                        f'but the contract declares the static size {dim}')
    if external_data and external_data not in references:
        raise ValueError(
            f'The record names external data {external_data!r} but the ONNX '
            f'graph references {references}')
    return {
        'inputs': [spec.name for spec in inputs],
        'outputs': [spec.name for spec in outputs],
        'external_data': references,
        'unresolved_axes': unresolved,
    }


def dummy_inputs(contract: EngineContract,
                 *,
                 device: Any = 'cpu') -> List[torch.Tensor]:
    """Representative inputs at the contract's optimum shape."""
    tensors: List[torch.Tensor] = []
    for spec in contract.inputs:
        profile = contract.profile_for(spec.name)
        shape = profile.opt_shape if profile else spec.resolve({})
        dtype = _torch_dtype(spec.dtype)
        if dtype.is_floating_point:
            tensors.append(torch.rand(shape, dtype=dtype, device=device))
        else:
            tensors.append(torch.zeros(shape, dtype=dtype, device=device))
    return tensors


def _torch_dtype(name: str) -> torch.dtype:
    from fluxvla.models.backbones.visions.base_trt import torch_dtype_for
    return torch_dtype_for(name)


def export_onnx(
    module: Any,
    contract: EngineContract,
    out_path: Any,
    *,
    opset_version: Optional[int] = None,
    external_data: bool = True,
    external_data_min_bytes: int = 1024,
    device: Any = 'cpu',
) -> Path:
    """Trace ``module`` to ONNX at the contract's boundary.

    The graph exposes every declared input and output by name with the
    contract's dynamic axes. Only the batch-style dynamic axes are exported as
    dynamic; static axes stay fixed.
    """
    _require_onnx('ONNX export')
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    traced = module.to(device) if device is not None else module
    traced = traced.eval()
    dummy = dummy_inputs(contract, device=device or 'cpu')
    dynamic_axes: Dict[str, Dict[int, str]] = {}
    for spec in list(contract.inputs) + list(contract.outputs):
        if spec.dynamic_axes:
            dynamic_axes[spec.name] = {
                axis: f'{spec.name}_axis{axis}'
                for axis in spec.dynamic_axes
            }
    torch.onnx.export(
        traced,
        tuple(dummy),
        str(out),
        input_names=list(contract.input_names),
        output_names=list(contract.output_names),
        dynamic_axes=dynamic_axes or None,
        opset_version=opset_version,
        do_constant_folding=True,
        external_data=bool(external_data),
        dynamo=False)
    if external_data:
        convert_onnx_to_external_data(
            out, size_threshold=int(external_data_min_bytes))
    logger.info('Exported ONNX graph %s', out)
    return out


def network_io(trt: Any,
               network: Any) -> Tuple[List[TensorSpec], List[TensorSpec]]:
    """Read a parsed network's declared IO.

    These dtypes are authoritative: the FP16 builder flag selects internal
    tactics and does not retype a float32 graph's inputs or outputs.
    """
    inputs: List[TensorSpec] = []
    outputs: List[TensorSpec] = []
    for index in range(int(network.num_inputs)):
        inputs.append(_network_spec(network.get_input(index), 'input'))
    for index in range(int(network.num_outputs)):
        outputs.append(_network_spec(network.get_output(index), 'output'))
    return inputs, outputs


def _network_spec(tensor: Any, kind: str) -> TensorSpec:
    dtype = trt_dtype_name(tensor.dtype)
    if dtype is None:
        raise ValueError(f'Network {kind} {tensor.name!r} uses TensorRT dtype '
                         f'{tensor.dtype!r}, which has no contract dtype name')
    shape = tensor.shape
    return TensorSpec(
        name=str(tensor.name),
        dtype=dtype,
        shape=tuple(int(shape[i]) for i in range(len(shape))))


def create_network(builder: Any, trt: Any) -> Any:
    """Create an explicit-batch network, falling back for newer TensorRT."""
    try:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        return builder.create_network(flags)
    except Exception:  # noqa: BLE001 - flag removed in newer 10.x builds
        return builder.create_network()


def _parse_onnx(parser: Any, path: Path, external_data: Optional[str]) -> bool:
    if external_data:
        parse_from_file = getattr(parser, 'parse_from_file', None)
        if parse_from_file is None:
            raise RuntimeError(
                'The ONNX graph stores weights in external data '
                f'{external_data!r} but this TensorRT OnnxParser has no '
                'parse_from_file; use a TensorRT build that supports external '
                'data instead of inlining the file')
        return bool(parse_from_file(str(path)))
    return bool(parser.parse(path.read_bytes()))


def _enable_bf16_tactics(trt: Any, builder: Any, config: Any) -> None:
    """Request ``BuilderFlag.BF16``, or fail with an actionable reason.

    BF16 is a preference rather than a promise: TensorRT applies it to the
    layers that have BF16 kernels and leaves the rest at the precision the
    graph declares, so a BF16 engine may be mixed precision. The one outcome
    that must never happen is a silent switch to FP16 tactics, so no failure
    path here falls back to ``BuilderFlag.FP16``.
    """
    version = str(getattr(trt, '__version__', 'unknown'))
    flag = getattr(trt.BuilderFlag, 'BF16', None)
    if flag is None:
        raise RuntimeError(
            f'TensorRT {version} does not expose BuilderFlag.BF16, so BF16 '
            'tactics cannot be requested. Build with a TensorRT 10.x release '
            'that provides the flag, or build fp32 or fp16 for this device '
            'instead.')
    platform = getattr(builder, 'platform_has_fast_bf16', None)
    if platform is False:
        raise RuntimeError(
            'This platform does not support fast BF16 execution: TensorRT '
            'reports platform_has_fast_bf16=False, so a BF16 build would be '
            'rejected or silently downgraded. Build fp32 or fp16 for this '
            'device instead.')
    if platform is None and torch.cuda.is_available():
        capability = tuple(
            int(value) for value in torch.cuda.get_device_capability())
        if capability < _BF16_MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                'BF16 tactics need CUDA compute capability '
                f'{_BF16_MIN_COMPUTE_CAPABILITY[0]}.'
                f'{_BF16_MIN_COMPUTE_CAPABILITY[1]} or newer; '
                f'{torch.cuda.get_device_name()} reports '
                f'{capability[0]}.{capability[1]}. Build fp32 or fp16 for '
                'this device instead.')
    config.set_flag(flag)


def build_engine(
    onnx_path: Any,
    engine_path: Any,
    contract: EngineContract,
    *,
    precision: str = 'fp32',
    allow_tf32: bool = False,
    workspace_bytes: int = 1 << 30,
    logger_severity: str = 'WARNING',
    verify: bool = True,
    onnx_sha256: Optional[str] = None,
    external_data: Optional[str] = None,
    external_data_sha256: Optional[str] = None,
    layer_precisions: Optional[Sequence[Any]] = None,
    precision_constraints: str = 'none',
    profiling_verbosity: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile an ONNX graph to a serialized TensorRT engine.

    The graph is verified against ``contract`` and its recorded hashes before
    anything is built, and the parsed network's own IO is validated against
    the same contract. ``precision='fp16'`` sets ``BuilderFlag.FP16`` and
    ``precision='bf16'`` sets ``BuilderFlag.BF16``; both select internal
    tactics only, so the network keeps the dtypes the graph declares and
    those are what the returned record reports. A BF16 build is allowed to be
    mixed precision - layers without BF16 kernels stay at the graph's
    precision - and it fails with an actionable error when the TensorRT build
    or the device cannot run BF16 rather than falling back to FP16. There is
    no INT8 calibration. ``verify`` deserializes the fresh engine and
    re-validates it.

    An FP32 build is strict by default: TF32 tactics would silently reduce
    precision, so ``allow_tf32=False`` clears them and asks TensorRT to obey
    the declared precisions. Pass ``allow_tf32=True`` to opt back in.

    ``layer_precisions`` is an opt-in list of ``PATTERN:DTYPE`` constraints
    (or parsed :class:`LayerPrecision` objects) applied to every layer whose
    name matches, which is the API equivalent of ``trtexec
    --layerPrecisions=*/LayerNormalization:fp32``. `precision_constraints`
    decides what a constraint is worth: ``'obey'`` makes it a requirement and
    fails the build rather than overriding it, ``'prefer'`` asks TensorRT to
    honour it while still permitting its documented fallback to the fastest
    implementation, and ``'none'`` leaves the precision a hint TensorRT may
    ignore - which is why an override with ``'none'`` is rejected here rather
    than trusted to hold. A pattern that matches no layer always fails. The
    record reports the requested policies and the matched layer names. An
    explicit ``'prefer'`` also takes precedence over the constraints flag a
    strict FP32 build sets on its own, which would otherwise contradict it.

    ``profiling_verbosity`` selects how much per-layer detail the engine keeps
    for the engine inspector; ``'detailed'`` is what makes a per-layer
    datatype inspection of the built engine possible. The default leaves
    TensorRT's own default (``layer_names_only``) in place.
    """
    if precision not in _PRECISIONS:
        raise ValueError(
            f'Unsupported build precision {precision!r}; expected '
            f'one of {_PRECISIONS}')
    if precision_constraints not in PRECISION_CONSTRAINTS:
        raise ValueError('Unsupported precision constraints mode '
                         f'{precision_constraints!r}; '
                         f'expected one of {PRECISION_CONSTRAINTS}')
    overrides = [
        parse_layer_precision(spec) for spec in (layer_precisions or [])
    ]
    if overrides and precision_constraints == 'none':
        raise ValueError(
            f'{len(overrides)} layer precision override(s) were requested '
            'with '
            "precision_constraints='none'. TensorRT treats a layer precision "
            'as a hint it may ignore unless the constraints flag is set, so '
            "this build could silently drop them. Pass "
            "precision_constraints='obey' to require them, or 'prefer' to "
            'ask for them while still allowing the documented fallback.')
    if precision_constraints != 'none' and not overrides:
        raise ValueError(
            f"precision_constraints={precision_constraints!r} was requested "
            'without any layer_precisions. TensorRT ignores the precision '
            'constraint flags when no layer carries a constraint, so this '
            'build would silently do nothing: pass at least one '
            'layer_precisions entry, or leave the mode at None/none.')
    verbosity = None
    if profiling_verbosity is not None:
        verbosity = str(profiling_verbosity).strip().lower()
        if verbosity not in PROFILING_LEVELS:
            raise ValueError(
                f'Unsupported profiling verbosity {profiling_verbosity!r}; '
                f'expected one of {PROFILING_LEVELS}')
    source = Path(onnx_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f'ONNX model not found: {source}')
    # Mandatory: a graph that no longer matches the contract must not become a
    # new engine.
    verify_onnx_graph(
        source,
        contract,
        onnx_sha256=onnx_sha256,
        external_data=external_data,
        external_data_sha256=external_data_sha256)
    trt = require_tensorrt()
    engine_out = Path(engine_path).expanduser()
    engine_out.parent.mkdir(parents=True, exist_ok=True)
    severity = getattr(trt.Logger,
                       str(logger_severity).upper(), trt.Logger.WARNING)
    trt_logger = trt.Logger(severity)
    builder = trt.Builder(trt_logger)
    network = create_network(builder, trt)
    parser = trt.OnnxParser(network, trt_logger)
    if not _parse_onnx(parser, source, external_data):
        errors = [
            str(parser.get_error(index))
            for index in range(int(getattr(parser, 'num_errors', 0)))
        ]
        raise RuntimeError(f'TensorRT could not parse {source}: ' + (
            '; '.join(errors) if errors else 'no parser error was reported'))
    inputs, outputs = network_io(trt, network)
    validate_engine_contract(
        contract, _ordered_by_contract(inputs, contract.inputs, 'input'),
        _ordered_by_contract(outputs, contract.outputs, 'output'))
    applied = apply_layer_precisions(trt, network, overrides)
    config = builder.create_builder_config()
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     int(workspace_bytes))
    precision_flags: List[str] = []

    def _record_flag(name: str) -> None:
        if name not in precision_flags:
            precision_flags.append(name)

    if verbosity is not None:
        target = getattr(
            getattr(trt, 'ProfilingVerbosity', None), verbosity.upper(), None)
        if target is None:
            raise ValueError(
                f'This TensorRT build exposes no ProfilingVerbosity.'
                f'{verbosity.upper()}; expected one of '
                f'{PROFILING_LEVELS}')
        config.profiling_verbosity = target
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        _record_flag('FP16')
    elif precision == 'bf16':
        _enable_bf16_tactics(trt, builder, config)
        _record_flag('BF16')
    if precision == 'fp32' and not allow_tf32:
        if hasattr(trt.BuilderFlag, 'TF32'):
            config.clear_flag(trt.BuilderFlag.TF32)
            _record_flag('NO_TF32')
        # An explicit 'prefer' would contradict this flag, so it wins; the
        # strict FP32 default is what an unrequested build keeps.
        strict = (
            precision_constraints != 'prefer'
            and hasattr(trt.BuilderFlag, 'OBEY_PRECISION_CONSTRAINTS'))
        if strict:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            _record_flag('OBEY_PRECISION_CONSTRAINTS')
    if precision_constraints != 'none':
        flag_name = ('OBEY_PRECISION_CONSTRAINTS' if precision_constraints
                     == 'obey' else 'PREFER_PRECISION_CONSTRAINTS')
        flag = getattr(trt.BuilderFlag, flag_name, None)
        if flag is None:
            raise RuntimeError(
                f'TensorRT {trt.__version__} does not expose '
                f'BuilderFlag.{flag_name}, so the requested layer precision '
                f'constraints could not be enforced. Build with a TensorRT '
                '10.x release that provides the flag, or drop the layer '
                'precision overrides.')
        config.set_flag(flag)
        _record_flag(flag_name)
    profile = builder.create_optimization_profile()
    declared = False
    for spec in contract.inputs:
        bounds = contract.profile_for(spec.name)
        if bounds is None:
            continue
        profile.set_shape(spec.name, bounds.min_shape, bounds.opt_shape,
                          bounds.max_shape)
        declared = True
    if declared:
        config.add_optimization_profile(profile)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError(
            f'TensorRT failed to build an engine from {source}; the graph, '
            'the selected precision, or the dynamic profile was rejected. '
            'Re-run with a more verbose logger severity for the builder '
            'diagnostics.')
    engine_out.write_bytes(bytes(serialized))
    data_sha = (
        sha256_file(source.parent / external_data) if external_data else None)
    record = {
        'engine_path': str(engine_out),
        'tensorrt_version': str(trt.__version__),
        'precision': precision,
        'precision_flags': precision_flags,
        'tf32_allowed': bool(allow_tf32 and precision == 'fp32'),
        'workspace_bytes': int(workspace_bytes),
        'input_names': [spec.name for spec in inputs],
        'output_names': [spec.name for spec in outputs],
        'input_dtypes': [spec.dtype for spec in inputs],
        'output_dtypes': [spec.dtype for spec in outputs],
        'onnx_sha256': sha256_file(source),
        'external_data': external_data,
        'external_data_sha256': data_sha,
        'created_utc': utc_now(),
        'build_seconds': round(elapsed, 3),
        'engine_bytes': engine_out.stat().st_size,
    }
    if applied:
        record['precision_constraints'] = precision_constraints
        record['layer_precisions'] = applied
        record['network_layers'] = int(network.num_layers)
    if verbosity is not None:
        record['profiling_verbosity'] = verbosity
    logger.info('Built TensorRT engine %s in %.1fs', engine_out, elapsed)
    if verify:
        record['validated'] = load_and_validate(
            engine_out,
            contract,
            expected_tensorrt_version=str(trt.__version__))
    return record


def load_and_validate(
        engine_path: Any,
        contract: EngineContract,
        *,
        device: Any = None,
        expected_tensorrt_version: Optional[str] = None) -> Dict[str, Any]:
    """Deserialize an engine and validate its IO against ``contract``."""
    engine = TensorRTEngine(
        engine_path,
        contract,
        device=device,
        expected_tensorrt_version=(expected_tensorrt_version))
    try:
        engine.load()
        return engine.describe()
    finally:
        engine.close()
