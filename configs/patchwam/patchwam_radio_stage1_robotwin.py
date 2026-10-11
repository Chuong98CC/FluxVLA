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
#
# PatchWAM + frozen C-RADIOv4, Stage 1: visual-projection warmup, every task.
#
# `pretrained_name_or_path` is the released (or a native) pretrained PatchWAM
# checkpoint. Only the new `D <-> hidden` visual input/output projections
# train; the denoising transformer, the frozen visual backbone, the text
# proprio projection and the pretrained action maps stay frozen. `visual_stage`
# makes the recipe refuse to start without that checkpoint
# (`from_pretrained` raises), and `visual_stage`/`vla_head.stage` must agree.
#
# Launch (single node, eight processes, from the FluxVLA root):
#   NPROC_PER_NODE=8 bash scripts/train.sh \
#     configs/patchwam/patchwam_radio_stage1_robotwin.py \
#     work_dirs/patchwam_radio_stage1_robotwin
#
# Stage 2 inherits the checkpoint this run writes. Point
# `PATCHWAM_STAGE1_CHECKPOINT` at a concrete
# `.../checkpoints/step-<step>-epoch-<epoch>[-loss=...].pt` file (or the
# `latest-checkpoint.pt` symlink); see the stage 2 recipe.

_base_ = ['./patchwam_radio_robotwin_base.py']

import os

# Released or native pretrained PatchWAM weights (same env var as the
# original PatchWAM recipes; the base recipe declares the same default).
_checkpoint = os.environ.get(
    'PATCHWAM_CHECKPOINT',
    '/data/Pretrained_models/patchwam-robotwin/step_140000.pt')

model = dict(
    visual_stage='stage1',
    pretrained_name_or_path=_checkpoint,
    vla_head=dict(
        stage='stage1',
        lora=None,
    ),
)
