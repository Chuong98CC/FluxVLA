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
"""Component reference for the frozen C-RADIO visual backbones.

This is a component example, not a runnable recipe: copy the dictionary you
need into the ``vision_backbone`` (or ``inference_model.vision_backbone``)
slot of a VLA/model config. Both entries register with ``VISION_BACKBONES``
and return the same ``[B, N, D]`` tokens and ``(h, w)`` grid.

The TensorRT entry repeats no geometry or dtype: the sidecar next to the
engine is the contract and is validated against the real engine at load.
Build the artifacts with ``tools/c_radio_tensorrt.py`` (see
``docs/c-radio-v4-tensorrt.md``).
"""

import torch

# PyTorch path: a complete local official C-RADIO export, loaded with
# ``local_files_only=True``. Nothing is downloaded and no model is
# substituted.
vision_backbone = dict(
    type='RadioViTBackbone',
    vision_backbone_id='c-radio_v4-h',
    model_path='/path/to/C-RADIOv4-H',
    patch_size=16,
    dtype=torch.bfloat16,
    default_image_resolution=(3, 256, 256),
)

# TensorRT path: the same tokens served by a serialized engine. ``engine_path``
# and its sidecar are produced offline; the engine is loaded lazily, so a
# config can be built and inspected where TensorRT is not installed.
inference_vision_backbone = dict(
    type='RadioViTBackboneTensorRT',
    vision_backbone_id='c-radio_v4-h',
    engine_path='/path/to/c_radio_v4_h_256.engine',
    metadata_path='/path/to/c_radio_v4_h_256.engine.json',
    device=0,
)
