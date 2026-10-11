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
# Stage 2 (projections + DiT LoRA) on the default five-task RoboTwin subset.
#
# The `task_selection` block is identical to the stage 1 ablation recipe, so
# the two stages resolve the same tasks and episodes.
#
# Launch (single node, eight processes, from the FluxVLA root):
#   PATCHWAM_STAGE1_CHECKPOINT=<stage1>/checkpoints/latest-checkpoint.pt \
#   NPROC_PER_NODE=8 bash scripts/train.sh \
#     configs/patchwam/patchwam_radio_ablation_default5_stage2_robotwin.py \
#     work_dirs/patchwam_radio_ablation_default5_stage2_robotwin

_base_ = ['./patchwam_radio_robotwin_base.py']

import os

_stage1_checkpoint = os.environ.get('PATCHWAM_STAGE1_CHECKPOINT', '')
_task_selection = dict(num_tasks=5, episodes_per_task=20, seed=42)

model = dict(
    visual_stage='stage2',
    stage1_checkpoint=(_stage1_checkpoint or None),
    vla_head=dict(
        stage='stage2',
        lora=dict(rank=32, alpha=64, dropout=0.0),
    ),
)

train_dataloader = dict(
    dataset=dict(datasets=dict(task_selection=dict(_task_selection))))
val_dataloader = dict(
    dataset=dict(datasets=dict(task_selection=dict(_task_selection))))

# Bounded ablation budget: a step budget instead of an epoch budget.
runner = dict(max_epochs=None, max_steps=2000, save_iter_interval=500)
