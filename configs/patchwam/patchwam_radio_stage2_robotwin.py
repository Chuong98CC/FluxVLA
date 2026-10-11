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
# PatchWAM + frozen C-RADIOv4, Stage 2: DiT LoRA adaptation, every task.
#
# `stage1_checkpoint` is the stage 1 run's checkpoint (a concrete
# `.../checkpoints/step-<step>-epoch-<epoch>[-loss=...].pt` file or the
# `latest-checkpoint.pt` symlink). `pretrained_name_or_path` is deliberately
# not set: a staged load always uses its own stage source, so a leftover base
# path is ignored. Only the visual
# projections and the LoRA adapters on the double/single block attention and
# feed-forward linears train; everything else stays frozen, and the native
# optimizer selects parameters by `requires_grad`, so it sees exactly those two
# groups. The adapters are zero-initialized, so stage 2 starts from the stage 1
# predictions exactly.
#
# Launch (single node, eight processes, from the FluxVLA root), with the
# stage 1 checkpoint written by that run:
#   export PATCHWAM_STAGE1_CHECKPOINT=<stage1>/checkpoints/latest-checkpoint.pt
#   NPROC_PER_NODE=8 bash scripts/train.sh \
#     configs/patchwam/patchwam_radio_stage2_robotwin.py \
#     work_dirs/patchwam_radio_stage2_robotwin

_base_ = ['./patchwam_radio_robotwin_base.py']

import os

_stage1_checkpoint = os.environ.get('PATCHWAM_STAGE1_CHECKPOINT', '')

model = dict(
    visual_stage='stage2',
    stage1_checkpoint=(_stage1_checkpoint or None),
    vla_head=dict(
        stage='stage2',
        lora=dict(rank=32, alpha=64, dropout=0.0),
    ),
)
