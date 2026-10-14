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
"""Model-independent descriptions of a TensorRT engine's boundary.

A :class:`TensorSpec` describes one named tensor, an
:class:`OptimizationProfile` bounds the dynamic axes of one input, and an
:class:`EngineContract` collects them into the shape, dtype and profile rules
an engine must satisfy. Nothing here knows about images, channels, features
or any particular model.

This module imports the standard library only, so a contract can be read,
written and validated without torch, TensorRT or the model package.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: Tensor dtype names this contract understands.
DTYPE_NAMES: Tuple[str, ...] = ('float32', 'float16', 'bfloat16', 'int32',
                                'int8', 'bool')

#: TensorRT reports a dynamic axis as -1.
DYNAMIC_DIM = -1


def _positive_int(field_name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{field_name} must be a positive integer; got '
                         f'{value!r}')
    return int(value)


def _name(value: Any, field_name: str = 'name') -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty string; got '
                         f'{value!r}')
    return str(value)


def _dtype(value: Any) -> str:
    if value not in DTYPE_NAMES:
        raise ValueError(
            f'Unsupported tensor dtype {value!r}; expected one of '
            f'{DTYPE_NAMES}')
    return str(value)


def tensorrt_major(version: str) -> int:
    """Major component of a TensorRT version string."""
    text = _name(version, 'tensorrt_version').strip()
    head = text.split('.', 1)[0]
    if not head.isdigit():
        raise ValueError(
            f'The TensorRT version {version!r} does not start with a major '
            'number')
    return int(head)


def _shape(raw: Any, *, allow_dynamic: bool = True) -> Tuple[int, ...]:
    shape = tuple(raw)
    if not shape:
        raise ValueError('A tensor shape must have at least one axis')
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f'Tensor shapes must contain integers; got '
                             f'{shape!r}')
        if dim < 1 and not (allow_dynamic and dim == DYNAMIC_DIM):
            raise ValueError(f'Invalid tensor axis {dim!r} in {shape!r}; use '
                             f'{DYNAMIC_DIM} for a dynamic axis')
    return tuple(int(dim) for dim in shape)


@dataclass(frozen=True)
class TensorSpec:
    """One named engine tensor: dtype plus declared shape.

    Declared axes are positive or :data:`DYNAMIC_DIM`. A declared axis of
    :data:`DYNAMIC_DIM` must be dynamic in the engine and, for an input,
    covered by an :class:`OptimizationProfile`.
    """

    name: str
    dtype: str
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'name', _name(self.name, 'tensor name'))
        object.__setattr__(self, 'dtype', _dtype(self.dtype))
        object.__setattr__(self, 'shape', _shape(self.shape))

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def dynamic_axes(self) -> Tuple[int, ...]:
        return tuple(index for index, dim in enumerate(self.shape)
                     if dim == DYNAMIC_DIM)

    def resolve(self, dynamic: Mapping[int, int]) -> Tuple[int, ...]:
        """Concrete shape, substituting the given dynamic axes."""
        resolved = []
        for index, dim in enumerate(self.shape):
            if dim == DYNAMIC_DIM:
                value = dynamic.get(index)
                if value is None:
                    raise ValueError(
                        f'Tensor {self.name!r} axis {index} is dynamic; a '
                        'concrete value is required')
                resolved.append(
                    _positive_int(f'{self.name} axis {index}', value))
            else:
                resolved.append(dim)
        return tuple(resolved)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'dtype': self.dtype,
            'shape': list(self.shape)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'TensorSpec':
        return cls(
            name=data['name'], dtype=data['dtype'], shape=tuple(data['shape']))


@dataclass(frozen=True)
class OptimizationProfile:
    """Dynamic-axis bounds for one engine input."""

    tensor: str
    min_shape: Tuple[int, ...]
    opt_shape: Tuple[int, ...]
    max_shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'tensor', _name(self.tensor,
                                                 'profile tensor'))
        minimum = _shape(self.min_shape, allow_dynamic=False)
        optimum = _shape(self.opt_shape, allow_dynamic=False)
        maximum = _shape(self.max_shape, allow_dynamic=False)
        if not len(minimum) == len(optimum) == len(maximum):
            raise ValueError('Profile bounds must have the same rank; got '
                             f'{minimum}, {optimum}, {maximum}')
        for axis, (low, mid,
                   high) in enumerate(zip(minimum, optimum, maximum)):
            if not low <= mid <= high:
                raise ValueError(
                    f'Profile axis {axis} must satisfy min <= opt <= max; got '
                    f'{low}, {mid}, {high}')
        object.__setattr__(self, 'min_shape', minimum)
        object.__setattr__(self, 'opt_shape', optimum)
        object.__setattr__(self, 'max_shape', maximum)

    @property
    def rank(self) -> int:
        return len(self.min_shape)

    def bounds(self, axis: int) -> Tuple[int, int, int]:
        return (self.min_shape[axis], self.opt_shape[axis],
                self.max_shape[axis])

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tensor': self.tensor,
            'min_shape': list(self.min_shape),
            'opt_shape': list(self.opt_shape),
            'max_shape': list(self.max_shape)
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'OptimizationProfile':
        return cls(
            tensor=data['tensor'],
            min_shape=tuple(data['min_shape']),
            opt_shape=tuple(data['opt_shape']),
            max_shape=tuple(data['max_shape']))


@dataclass(frozen=True)
class EngineContract:
    """The named, typed, shaped and profiled boundary an engine must match."""

    inputs: Tuple[TensorSpec, ...]
    outputs: Tuple[TensorSpec, ...]
    profiles: Tuple[OptimizationProfile, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if not inputs:
            raise ValueError('An engine contract needs at least one input')
        if not outputs:
            raise ValueError('An engine contract needs at least one output')
        names = [spec.name for spec in inputs + outputs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f'Engine contract tensor names must be unique; '
                             f'duplicated {duplicates}')
        by_name = {spec.name: spec for spec in inputs}
        profiled = [profile.tensor for profile in self.profiles]
        duplicated_profiles = sorted(
            {name
             for name in profiled if profiled.count(name) > 1})
        if duplicated_profiles:
            raise ValueError(
                'Engine contract declares more than one optimization '
                f'profile for {duplicated_profiles}; the executor uses one '
                'profile per input')
        for profile in self.profiles:
            if profile.tensor not in by_name:
                raise ValueError(
                    f'Optimization profile targets {profile.tensor!r}, which '
                    'is not an engine input')
            spec = by_name[profile.tensor]
            if profile.rank != spec.rank:
                raise ValueError(
                    f'Profile for {profile.tensor!r} has rank {profile.rank}; '
                    f'the input declares rank {spec.rank}')
            for axis, dim in enumerate(spec.shape):
                if dim != DYNAMIC_DIM and (
                        profile.min_shape[axis], profile.opt_shape[axis],
                        profile.max_shape[axis]) != (dim, dim, dim):
                    raise ValueError(
                        f'Profile for {profile.tensor!r} must fix static axis '
                        f'{axis} at {dim}')
        for spec in inputs:
            if spec.dynamic_axes and not self.profile_for(spec.name):
                raise ValueError(
                    f'Dynamic input {spec.name!r} needs an optimization '
                    'profile')
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'outputs', outputs)
        object.__setattr__(self, 'profiles', tuple(self.profiles))

    @property
    def input_names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self.inputs)

    @property
    def output_names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self.outputs)

    def spec(self, name: str) -> TensorSpec:
        for candidate in self.inputs + self.outputs:
            if candidate.name == name:
                return candidate
        raise KeyError(f'Engine contract has no tensor named {name!r}')

    def profile_for(self, name: str) -> Optional[OptimizationProfile]:
        for profile in self.profiles:
            if profile.tensor == name:
                return profile
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'inputs': [spec.to_dict() for spec in self.inputs],
            'outputs': [spec.to_dict() for spec in self.outputs],
            'profiles': [profile.to_dict() for profile in self.profiles],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'EngineContract':
        return cls(
            inputs=tuple(
                TensorSpec.from_dict(item) for item in data['inputs']),
            outputs=tuple(
                TensorSpec.from_dict(item) for item in data['outputs']),
            profiles=tuple(
                OptimizationProfile.from_dict(item)
                for item in data.get('profiles', ())),
        )

    def save(self, path: Any) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n')
        return target

    @classmethod
    def load(cls, path: Any) -> 'EngineContract':
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f'Engine contract not found: {source}')
        try:
            return cls.from_dict(json.loads(source.read_text()))
        except json.JSONDecodeError as error:
            raise ValueError(f'Engine contract {source} is not valid JSON: '
                             f'{error}') from error


def _check_tensor(declared: TensorSpec, actual: TensorSpec,
                  profile: Optional[OptimizationProfile], kind: str) -> None:
    if actual.name != declared.name:
        raise ValueError(f'Engine {kind} is named {actual.name!r} but the '
                         f'contract declares {declared.name!r}')
    if actual.dtype != declared.dtype:
        raise ValueError(
            f'Engine {kind} {declared.name!r} has dtype {actual.dtype!r} but '
            f'the contract declares {declared.dtype!r}')
    if actual.rank != declared.rank:
        raise ValueError(
            f'Engine {kind} {declared.name!r} has rank {actual.rank} but the '
            f'contract declares {declared.rank}')
    for axis, dim in enumerate(declared.shape):
        found = actual.shape[axis]
        if not axis_allowed(dim, found, profile, axis):
            if dim == DYNAMIC_DIM:
                raise ValueError(
                    f'Engine {kind} {declared.name!r} axis {axis} is declared '
                    f'dynamic but the engine reports {found}; the profile is '
                    f'{profile}')
            raise ValueError(
                f'Engine {kind} {declared.name!r} axis {axis} is {found} but '
                f'the contract declares the static size {dim}')


def axis_allowed(declared: int, actual: int,
                 profile: Optional[OptimizationProfile], axis: int) -> bool:
    """Whether a concrete axis value satisfies a declared axis and profile."""
    if declared != DYNAMIC_DIM:
        return actual == declared
    if actual == DYNAMIC_DIM:
        return True
    if profile is not None:
        low, _, high = profile.bounds(axis)
        return low <= actual <= high
    return False


def validate_engine_contract(contract: EngineContract,
                             engine_inputs: Sequence[TensorSpec],
                             engine_outputs: Sequence[TensorSpec]) -> None:
    """Validate a live engine's reported tensors against ``contract``.

    Tensors are matched by name, not by position, so an engine that reports its
    IO in a different order still validates. Names, dtypes, static axes and
    dynamic axes (against the declared optimization bounds) must all agree, so
    an engine built for a different boundary fails at load instead of producing
    wrong results.
    """
    if len(engine_inputs) != len(contract.inputs):
        raise ValueError(
            f'The engine has {len(engine_inputs)} inputs but the contract '
            f'declares {contract.input_names}')
    if len(engine_outputs) != len(contract.outputs):
        raise ValueError(
            f'The engine has {len(engine_outputs)} outputs but the contract '
            f'declares {contract.output_names}')
    _match_by_name('input', contract.inputs, engine_inputs, contract)
    _match_by_name('output', contract.outputs, engine_outputs, contract)


def _match_by_name(kind: str, declared: Sequence[TensorSpec],
                   actual: Sequence[TensorSpec],
                   contract: EngineContract) -> None:
    found = {spec.name: spec for spec in actual}
    missing = sorted(spec.name for spec in declared if spec.name not in found)
    if missing:
        raise ValueError(f'The engine reports no {kind} named {missing}; it '
                         f'reports {sorted(found)}')
    extra = sorted(name for name in found
                   if name not in {spec.name
                                   for spec in declared})
    if extra:
        raise ValueError(f'The engine reports unexpected {kind}(s) {extra}')
    for spec in declared:
        _check_tensor(spec, found[spec.name], contract.profile_for(spec.name),
                      kind)


def validate_engine_profiles(
        contract: EngineContract,
        engine_profiles: Sequence[OptimizationProfile]) -> None:
    """Validate declared bounds against the bounds the engine actually has.

    The contract is the caller's expectation; the engine reports what it was
    built with. A mismatch means the engine cannot be driven by the declared
    profiles, so it is rejected rather than silently clamped.
    """
    actual = {profile.tensor: profile for profile in engine_profiles}
    for declared in contract.profiles:
        reported = actual.get(declared.tensor)
        if reported is None:
            raise ValueError(f'The engine reports no optimization profile for '
                             f'{declared.tensor!r}; the contract declares '
                             f'{declared.min_shape}/{declared.opt_shape}/'
                             f'{declared.max_shape}')
        if (declared.min_shape, declared.opt_shape,
                declared.max_shape) != (reported.min_shape, reported.opt_shape,
                                        reported.max_shape):
            raise ValueError(
                f'Optimization profile for {declared.tensor!r} does not match '
                f'the engine: declared {declared.min_shape}/'
                f'{declared.opt_shape}/{declared.max_shape} but the engine '
                f'reports {reported.min_shape}/{reported.opt_shape}/'
                f'{reported.max_shape}')
