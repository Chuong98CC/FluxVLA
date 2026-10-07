# Copyright 2026 PatchWAM contributors.
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
"""Vendored PatchWAM numerical core (parameter-free codec, flow, policy).

This package contains the vendor-copied PatchWAM components required by the
FluxVLA ``PatchWAMVLA`` / ``PatchWAMHead`` integration. The optional FLUX.2
asset adapter and specialized Qwen3-VL encoder are added in a later phase; they
are intentionally not imported here so the core stays import-light and works
without model assets.
"""

from .codec import RepeatedActionCodec
from .flow import ShiftedFlow
from .geometry import (flatten_image_latents, joint_visibility,
                       raster_coordinates, restore_image_latents,
                       sequence_coordinates)
from .policy import PatchFlowPolicy, make_tiny_policy
from .tiny import ConditionalTokenBlock, SmallJointTransformer

__all__ = [
    'ConditionalTokenBlock',
    'PatchFlowPolicy',
    'RepeatedActionCodec',
    'ShiftedFlow',
    'SmallJointTransformer',
    'flatten_image_latents',
    'joint_visibility',
    'make_tiny_policy',
    'raster_coordinates',
    'restore_image_latents',
    'sequence_coordinates',
]
