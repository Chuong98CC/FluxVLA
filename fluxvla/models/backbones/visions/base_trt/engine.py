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
"""Reusable TensorRT 10.x executor for serialized engines.

The engine is described by an :class:`EngineContract` and executes named
tensors; it knows nothing about images, channels, feature widths, rasters or
any particular model. ``tensorrt`` is imported only when an engine is loaded.

Execution is name-addressed (``set_tensor_address``/``execute_async_v3``) and
asynchronous: inputs and outputs are allocated through PyTorch on the caller's
current stream and the caller orders dependent work by using that stream. One
host lock serializes load, context creation, teardown and the shape/address
mutation window, so concurrent callers cannot interleave; device work on
different streams still overlaps. Buffers are recorded on the execution stream
for the caching allocator and kept referenced until the next enqueue on that
stream. ``close`` explicitly synchronizes the host: it waits for every
recorded execution stream before releasing contexts, buffers, the engine and
the TensorRT logger/runtime handles, which must outlive the engine.
TensorRT may also synchronize internally when enqueueing on CUDA's default
stream. Use a non-default current stream to avoid that default-stream cost.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from .metadata import (DYNAMIC_DIM, EngineContract, OptimizationProfile,
                       TensorSpec, axis_allowed, tensorrt_major,
                       validate_engine_contract, validate_engine_profiles)

logger = logging.getLogger(__name__)

#: TensorRT major versions whose Python API this engine is written against.
SUPPORTED_TENSORRT_MAJORS: Tuple[int, ...] = (10, )

_TRT_DTYPE_NAMES: Dict[str, str] = {
    'FLOAT': 'float32',
    'HALF': 'float16',
    'BF16': 'bfloat16',
    'INT32': 'int32',
    'INT8': 'int8',
    'BOOL': 'bool',
}

_TORCH_DTYPES: Dict[str, torch.dtype] = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'int32': torch.int32,
    'int8': torch.int8,
    'bool': torch.bool,
}


def torch_dtype_for(name: str) -> torch.dtype:
    """Torch dtype for a contract dtype name."""
    try:
        return _TORCH_DTYPES[name]
    except KeyError as error:  # pragma: no cover - guarded by metadata
        raise ValueError(f'Unsupported tensor dtype {name!r}') from error


def trt_dtype_name(dtype: Any) -> Optional[str]:
    """Contract dtype name for a ``trt.DataType`` member, or ``None``."""
    return _TRT_DTYPE_NAMES.get(str(getattr(dtype, 'name', dtype)))


def record_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Pin ``tensor``'s storage to ``stream`` for the caching allocator.

    PyTorch returns a freed block to the pool of the stream it was allocated
    on. The engine reads its inputs and writes its outputs on the execution
    stream, which may differ from the caller's allocation stream, so those
    buffers are recorded on it.
    """
    tensor.record_stream(stream)


def wait_for_stream(stream: torch.cuda.Stream) -> None:
    """Block the host until work already enqueued on ``stream`` completes.

    Used by :meth:`TensorRTEngine.close` before anything is destroyed. It is
    the only host synchronization in this module; execution never calls it.
    """
    event = torch.cuda.Event()
    event.record(stream)
    event.synchronize()


def require_tensorrt() -> Any:
    """Import TensorRT lazily and reject an unsupported major version."""
    try:
        import tensorrt as trt  # noqa: PLC0415 - intentionally lazy
    except ImportError as error:
        raise RuntimeError(
            'TensorRT Python bindings are required to run a serialized '
            'engine. Install TensorRT 10.x and the matching hardware '
            f'libraries. Original error: {error!r}') from error
    version = str(getattr(trt, '__version__', '') or '')
    if not version:
        raise RuntimeError('The TensorRT module exposes no __version__; '
                           'cannot verify it is a supported 10.x build')
    head = version.split('.', 1)[0]
    if not head.isdigit():
        raise RuntimeError(
            'The TensorRT module reports an unparseable version '
            f'{version!r}; cannot verify it is a supported 10.x build')
    major = int(head)
    if major not in SUPPORTED_TENSORRT_MAJORS:
        raise RuntimeError(
            f'TensorRT {version} is not supported; this engine targets the '
            f'TensorRT {SUPPORTED_TENSORRT_MAJORS[0]}.x named-IO and '
            'execute_async_v3 API. TensorRT 11 changed strong typing and '
            'precision semantics, so a 11.x build cannot be treated as 10.x. '
            'Install a TensorRT 10.x runtime or rebuild for the installed '
            'major version.')
    return trt


def engine_profile_shapes(trt: Any, engine: Any,
                          name: str) -> Optional[Tuple[Tuple[int, ...], ...]]:
    """``(min, opt, max)`` shapes of the engine's profile 0 for ``name``.

    Only profile index 0 is read: this executor adds exactly one optimization
    profile, so an engine with several profiles needs a selector it does not
    implement.
    """
    getter = getattr(engine, 'get_tensor_profile_shape', None)
    if getter is None:
        getter = getattr(engine, 'get_profile_shape', None)
        if getter is None:
            return None
        shapes = getter(0, name)
    else:
        shapes = getter(name, 0)
    if shapes is None or len(shapes) != 3:
        return None
    return tuple(tuple(int(value) for value in shape) for shape in shapes)


def engine_profiles(trt: Any, engine: Any,
                    inputs: Sequence[TensorSpec]) -> List[OptimizationProfile]:
    """The optimization profile bounds the engine actually reports."""
    profiles: List[OptimizationProfile] = []
    for spec in inputs:
        if not spec.dynamic_axes:
            continue
        shapes = engine_profile_shapes(trt, engine, spec.name)
        if shapes is None:
            raise ValueError(
                f'Engine input {spec.name!r} is dynamic but the engine '
                'exposes no optimization profile for it')
        profiles.append(
            OptimizationProfile(
                tensor=spec.name,
                min_shape=shapes[0],
                opt_shape=shapes[1],
                max_shape=shapes[2]))
    return profiles


def tensor_specs(trt: Any, engine: Any, kind: str) -> List[TensorSpec]:
    """Read the engine's reported tensors of one kind as specs.

    Only contiguous device ``LINEAR`` tensors are supported: this executor
    allocates plain PyTorch buffers and binds them by address, so a vectorized
    or host-bound tensor would need a layout conversion it does not implement.
    Such a tensor is rejected here instead of being bound incorrectly.
    """
    specs: List[TensorSpec] = []
    mode = getattr(trt.TensorIOMode, kind)
    for index in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(index))
        if engine.get_tensor_mode(name) != mode:
            continue
        _check_engine_tensor_layout(trt, engine, name, kind)
        raw_dtype = engine.get_tensor_dtype(name)
        dtype = trt_dtype_name(raw_dtype)
        if dtype is None:
            raise ValueError(
                f'Engine tensor {name!r} uses TensorRT dtype {raw_dtype!r}, '
                'which has no contract dtype name')
        shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
        specs.append(TensorSpec(name=name, dtype=dtype, shape=shape))
    return specs


def _check_engine_tensor_layout(trt: Any, engine: Any, name: str,
                                kind: str) -> None:
    """Reject IO this executor cannot bind correctly."""
    get_format = getattr(engine, 'get_tensor_format', None)
    if get_format is not None:
        found = get_format(name)
        if found != trt.TensorFormat.LINEAR:
            raise ValueError(
                f'Engine {kind} {name!r} uses TensorRT format {found!r}; '
                'this executor binds contiguous LINEAR tensors only')
    get_location = getattr(engine, 'get_tensor_location', None)
    if get_location is not None:
        found = get_location(name)
        if found != trt.TensorLocation.DEVICE:
            raise ValueError(
                f'Engine {kind} {name!r} lives at {found!r}; this executor '
                'binds device tensors only')
    is_shape_inference = getattr(engine, 'is_shape_inference_io', None)
    if is_shape_inference is not None and is_shape_inference(name):
        raise ValueError(
            f'Engine {kind} {name!r} is a shape-inference tensor, which this '
            'executor does not implement')


class TensorRTEngine:
    """Named-tensor executor for a serialized TensorRT 10.x engine.

    Args:
        engine_path: Serialized ``.engine`` file, read on first use.
        contract: The boundary the engine must satisfy. ``None`` derives one
            from the engine itself.
        device: CUDA device index or ``torch.device`` for execution.
        logger_severity: TensorRT logger severity name.
    """

    def __init__(self,
                 engine_path: Any,
                 contract: Optional[EngineContract] = None,
                 *,
                 device: Optional[Any] = None,
                 expected_tensorrt_version: Optional[str] = None,
                 logger_severity: str = 'WARNING') -> None:
        self.engine_path = Path(engine_path).expanduser()
        if contract is not None and not isinstance(contract, EngineContract):
            raise ValueError('contract must be an EngineContract or None')
        self.contract = contract
        self.expected_tensorrt_version = expected_tensorrt_version
        self._requested_device = device
        self._device_index: Optional[int] = None
        self._logger_severity = logger_severity
        self._trt: Optional[Any] = None
        # Retained for the engine's lifetime: the engine holds pointers into
        # the logger, and the runtime must outlive every engine it produced.
        self._trt_logger: Optional[Any] = None
        self._runtime: Optional[Any] = None
        self._engine: Optional[Any] = None
        self._inputs: Tuple[TensorSpec, ...] = ()
        self._outputs: Tuple[TensorSpec, ...] = ()
        self._contexts: Dict[int, Any] = {}
        # stream handle -> (stream, {name: tensor}) for the work in flight.
        self._in_flight: Dict[int, Tuple[torch.cuda.Stream,
                                         Dict[str, torch.Tensor]]] = {}
        # Serializes load, context creation, teardown and the shape/address
        # mutation window. Enqueue stays asynchronous.
        self._lock = threading.RLock()
        self._closed = False

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def device_index(self) -> Optional[int]:
        return self._device_index

    def _resolve_device_index(self) -> int:
        device = self._requested_device
        if device is None:
            return int(torch.cuda.current_device())
        if isinstance(device, torch.device):
            if device.type != 'cuda':
                raise ValueError(
                    f'The TensorRT engine must run on CUDA; got {device}')
            index = device.index
            return int(
                index if index is not None else torch.cuda.current_device())
        if isinstance(device, int) and not isinstance(device, bool):
            return int(device)
        raise ValueError(
            'device must be None, an int index, or a torch.device;'
            f' got {device!r}')

    def load(self) -> 'TensorRTEngine':
        """Import TensorRT, deserialize the engine, and validate its IO."""
        with self._lock:
            self._load_locked()
            return self

    def _load_locked(self) -> None:
        if self._closed:
            raise RuntimeError('This TensorRT engine has been closed')
        if self._engine is not None:
            return
        if not self.engine_path.is_file():
            raise FileNotFoundError(
                f'Serialized TensorRT engine not found: {self.engine_path}')
        trt = require_tensorrt()
        if not torch.cuda.is_available():
            raise RuntimeError(
                'A CUDA device is required to deserialize a TensorRT engine')
        device_index = self._resolve_device_index()
        severity = getattr(trt.Logger,
                           str(self._logger_severity).upper(),
                           trt.Logger.WARNING)
        trt_logger = trt.Logger(severity)
        try:
            serialized = self.engine_path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f'Unable to read the TensorRT engine {self.engine_path}: '
                f'{error!r}') from error
        with torch.cuda.device(device_index):
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError(
                'TensorRT failed to deserialize '
                f'{self.engine_path}. The file may be truncated, built for a '
                'different TensorRT major version, or built for different '
                'hardware than this device.')
        self._check_expected_version(str(trt.__version__))
        inputs = tuple(tensor_specs(trt, engine, 'INPUT'))
        outputs = tuple(tensor_specs(trt, engine, 'OUTPUT'))
        profiles = tuple(engine_profiles(trt, engine, inputs))
        if self.contract is None:
            self.contract = EngineContract(
                inputs=inputs, outputs=outputs, profiles=profiles)
        else:
            validate_engine_contract(self.contract, inputs, outputs)
            validate_engine_profiles(self.contract, profiles)
        self._trt = trt
        self._trt_logger = trt_logger
        self._runtime = runtime
        self._engine = engine
        self._device_index = device_index
        self._inputs = inputs
        self._outputs = outputs
        logger.debug('Loaded TensorRT engine %s: inputs=%s outputs=%s',
                     self.engine_path, [spec.shape for spec in inputs],
                     [spec.shape for spec in outputs])

    def _check_expected_version(self, runtime_version: str) -> None:
        """Check the recorded build version against the installed runtime."""
        recorded = self.expected_tensorrt_version
        if not recorded:
            return
        if tensorrt_major(recorded) != tensorrt_major(runtime_version):
            raise RuntimeError(
                f'The engine records a TensorRT {recorded} build but the '
                f'runtime is {runtime_version}; engines are not portable '
                'across major versions and this executor targets TensorRT '
                f'{SUPPORTED_TENSORRT_MAJORS[0]}.x. Rebuild the engine for '
                'the installed major version.')
        if str(recorded) != str(runtime_version):
            logger.warning(
                'The engine records a TensorRT %s build and is running on %s; '
                'engines are not portable across builds and the kernel '
                'selection or numerics may differ', recorded, runtime_version)

    def _ensure_loaded(self) -> Any:
        if self._engine is None:
            self.load()
        return self._engine

    # -- execution --------------------------------------------------------
    def context_for(self, stream: torch.cuda.Stream) -> Any:
        """Execution context bound to ``stream``, created once per stream."""
        with self._lock:
            engine = self._ensure_loaded()
            key = int(stream.cuda_stream)
            context = self._contexts.get(key)
            if context is None:
                with torch.cuda.device(int(self._device_index)):
                    context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError(
                        'TensorRT failed to create an execution context for '
                        f'{self.engine_path}')
                self._contexts[key] = context
            return context

    def _check_tensor(self, spec: TensorSpec, tensor: Any) -> None:
        if not torch.is_tensor(tensor):
            raise ValueError(
                f'Input {spec.name!r} must be a torch tensor; got '
                f'{type(tensor).__name__}')
        if not tensor.is_cuda:
            raise ValueError(f'Input {spec.name!r} must be a CUDA tensor; got '
                             f'{tensor.device}')
        if self._device_index is not None and (tensor.device.index
                                               != self._device_index):
            raise ValueError(
                f'Input {spec.name!r} is on device {tensor.device.index} but '
                f'the engine runs on device {self._device_index}')
        profile = self.contract.profile_for(spec.name)
        shape = tuple(int(dim) for dim in tensor.shape)
        if len(shape) != spec.rank or any(
                not axis_allowed(spec.shape[axis], shape[axis], profile, axis)
                for axis in range(spec.rank)):
            raise ValueError(
                f'Input {spec.name!r} has shape {shape}; the engine expects '
                f'{spec.shape}' +
                (f' within profile {profile.max_shape}' if profile else ''))
        expected = torch_dtype_for(spec.dtype)
        if tensor.dtype != expected:
            raise ValueError(
                f'Input {spec.name!r} dtype is {tensor.dtype} but '
                f'the engine input is {expected} ({spec.dtype!r})')
        if not tensor.is_contiguous():
            raise ValueError(f'Input {spec.name!r} must be contiguous')

    def _check_inputs(self, inputs: Mapping[str, Any]) -> None:
        if not isinstance(inputs, Mapping):
            raise ValueError('Engine inputs must be a mapping of name to '
                             f'tensor; got {type(inputs).__name__}')
        missing = sorted(set(self.contract.input_names) - set(inputs))
        if missing:
            raise ValueError(f'Missing engine input(s) {missing}; the engine '
                             f'needs {self.contract.input_names}')
        unknown = sorted(set(inputs) - set(self.contract.input_names))
        if unknown:
            raise ValueError(f'Unknown engine input(s) {unknown}; the engine '
                             f'accepts {self.contract.input_names}')
        for spec in self.contract.inputs:
            self._check_tensor(spec, inputs[spec.name])

    def execute(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Enqueue one inference and return the resolved output tensors.

        Held under the host lock for the shape/address mutation window, so two
        callers (including two on the same CUDA stream) cannot interleave.
        No explicit host wait is issued here; :meth:`close` waits. TensorRT
        itself may synchronize when the current stream is CUDA's default.
        """
        with self._lock:
            self._ensure_loaded()
            self._check_inputs(inputs)
            device_index = int(self._device_index)
            with torch.cuda.device(device_index):
                stream = torch.cuda.current_stream(device_index)
                context = self.context_for(stream)
                for spec in self.contract.inputs:
                    shape = tuple(int(dim) for dim in inputs[spec.name].shape)
                    try:
                        accepted = context.set_input_shape(spec.name, shape)
                    except Exception as error:  # noqa: BLE001 - report
                        raise RuntimeError(
                            f'TensorRT rejected the input shape {shape} for '
                            f'{spec.name!r}: {error!r}') from error
                    if accepted is False:
                        raise RuntimeError(
                            f'TensorRT refused the input shape {shape} for '
                            f'{spec.name!r} (set_input_shape returned False)')
                outputs: Dict[str, torch.Tensor] = {}
                for spec in self.contract.outputs:
                    shape = tuple(
                        int(dim)
                        for dim in context.get_tensor_shape(spec.name))
                    if any(dim < 1 for dim in shape):
                        raise RuntimeError(
                            'TensorRT reported an unresolved output shape '
                            f'{shape} for {spec.name!r}')
                    if len(shape) != spec.rank:
                        raise RuntimeError(
                            f'TensorRT resolved output {spec.name!r} to rank '
                            f'{len(shape)} but the contract declares rank '
                            f'{spec.rank}')
                    for axis, dim in enumerate(spec.shape):
                        if dim != DYNAMIC_DIM and shape[axis] != dim:
                            raise RuntimeError(
                                f'TensorRT resolved output {spec.name!r} axis '
                                f'{axis} to {shape[axis]} but the contract '
                                f'declares the static size {dim}')
                    outputs[spec.name] = torch.empty(
                        shape,
                        dtype=torch_dtype_for(spec.dtype),
                        device=inputs[self.contract.inputs[0].name].device)
                bound = []
                for spec in self.contract.inputs:
                    bound.append(
                        context.set_tensor_address(
                            spec.name, int(inputs[spec.name].data_ptr())))
                for spec in self.contract.outputs:
                    bound.append(
                        context.set_tensor_address(
                            spec.name, int(outputs[spec.name].data_ptr())))
                names = self.contract.input_names + self.contract.output_names
                if any(value is False for value in bound):
                    raise RuntimeError(
                        'TensorRT refused a named tensor address for '
                        f'{names} (set_tensor_address returned False)')
                enqueue = getattr(context, 'execute_async_v3', None)
                if enqueue is None:
                    raise RuntimeError(
                        f'TensorRT {self._trt.__version__} execution context '
                        'has no execute_async_v3; this engine targets the '
                        'TensorRT 10.x named-IO API')
                key = int(stream.cuda_stream)
                held = dict(inputs)
                held.update(outputs)
                # Recorded before the enqueue: a call that fails part-way
                # through still has outstanding work, so close must wait for
                # this stream and the allocator must not recycle the buffers.
                self._in_flight[key] = (stream, held)
                for tensor in held.values():
                    record_stream(tensor, stream)
                try:
                    enqueued = enqueue(key)
                except Exception as error:  # noqa: BLE001 - report
                    raise RuntimeError(
                        'TensorRT execute_async_v3 failed for '
                        f'{self.engine_path}: {error!r}') from error
                if enqueued is False:
                    raise RuntimeError(
                        'TensorRT execute_async_v3 reported failure for '
                        f'{self.engine_path}')
            return outputs

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        """Single-input/single-output convenience over :meth:`execute`."""
        self._ensure_loaded()
        if len(self.contract.inputs) != 1 or len(self.contract.outputs) != 1:
            raise ValueError(
                'run() needs a single-input, single-output contract; this '
                f'engine has {self.contract.input_names} -> '
                f'{self.contract.output_names}')
        return self.execute({self.contract.inputs[0].name:
                             tensor})[self.contract.outputs[0].name]

    # -- lifetime ---------------------------------------------------------
    def close(self) -> None:
        """Wait for outstanding work, then release contexts and the engine.

        The one place this engine synchronizes the host: every recorded
        execution stream is waited on before any context, buffer, the engine
        or the TensorRT handles are destroyed, so nothing is released while
        the GPU may still be reading it.
        """
        with self._lock:
            if self._closed:
                return
            self._synchronize_in_flight()
            self._release()

    def _synchronize_in_flight(self) -> None:
        """Block the host until every recorded execution stream is idle."""
        for stream, _ in self._in_flight.values():
            wait_for_stream(stream)

    def _release(self) -> None:
        """Drop contexts, buffers, the engine and the TensorRT handles."""
        with self._lock:
            self._contexts.clear()
            self._in_flight.clear()
            self._inputs = ()
            self._outputs = ()
            self._engine = None
            self._runtime = None
            self._trt_logger = None
            self._trt = None
            self._closed = True

    def describe(self) -> Dict[str, Any]:
        """Contract, engine IO and lifecycle state, for inspection."""
        return {
            'engine_path':
            str(self.engine_path),
            'loaded':
            self.loaded,
            'device_index':
            self._device_index,
            'expected_tensorrt_version':
            self.expected_tensorrt_version,
            'tensorrt_handles_retained':
            bool(self._runtime is not None and self._trt_logger is not None),
            'contract':
            (None if self.contract is None else self.contract.to_dict()),
            'inputs': [spec.to_dict() for spec in self._inputs],
            'outputs': [spec.to_dict() for spec in self._outputs],
        }

    def __enter__(self) -> 'TensorRTEngine':
        return self.load()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


__all__ = [
    'SUPPORTED_TENSORRT_MAJORS',
    'TensorRTEngine',
    'record_stream',
    'require_tensorrt',
    'tensor_specs',
    'torch_dtype_for',
    'trt_dtype_name',
    'wait_for_stream',
]
