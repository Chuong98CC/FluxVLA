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
"""Reusable TensorRT execution support for vision backbones.

``metadata`` and ``engine`` are model-independent: they describe and execute a
named, typed, shaped engine boundary. Model-specific sidecars live beside
them as adapters (``radio_metadata`` for C-RADIO) and are not exported here,
so importing the generic layer never pulls in a model package.
"""

from .engine import (SUPPORTED_TENSORRT_MAJORS, TensorRTEngine, record_stream,
                     require_tensorrt, tensor_specs, torch_dtype_for,
                     trt_dtype_name, wait_for_stream)  # noqa: F401
from .metadata import (DTYPE_NAMES, DYNAMIC_DIM, EngineContract,
                       OptimizationProfile, TensorSpec,
                       validate_engine_contract)  # noqa: F401

__all__ = [
    'DTYPE_NAMES',
    'DYNAMIC_DIM',
    'EngineContract',
    'OptimizationProfile',
    'SUPPORTED_TENSORRT_MAJORS',
    'TensorRTEngine',
    'TensorSpec',
    'record_stream',
    'require_tensorrt',
    'tensor_specs',
    'torch_dtype_for',
    'trt_dtype_name',
    'validate_engine_contract',
    'wait_for_stream',
]
