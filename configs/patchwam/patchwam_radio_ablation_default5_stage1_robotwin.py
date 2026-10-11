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
# Stage 1 on the default five-task RoboTwin ablation subset.
#
# The subset is resolved from the per-root metadata before sample indexes and
# statistics: five distinct tasks (seed-controlled) with a bounded number of
# episodes per task, identified by task *name* because task indices are
# root-local. The stage 1 and stage 2 ablation recipes declare the same
# `task_selection`, so both stages train on identical episodes; the resolved
# selection (task names and concrete root/episode ids) is recorded in the run
# metadata.
#
# Launch (single node, eight processes, from the FluxVLA root):
#   NPROC_PER_NODE=8 bash scripts/train.sh \
#     configs/patchwam/patchwam_radio_ablation_default5_stage1_robotwin.py \
#     work_dirs/patchwam_radio_ablation_default5_stage1_robotwin

_base_ = ['./patchwam_radio_robotwin_base.py']

import os

_checkpoint = os.environ.get(
    'PATCHWAM_CHECKPOINT',
    '/data/Pretrained_models/patchwam-robotwin/step_140000.pt')
_task_selection = dict(num_tasks=5, episodes_per_task=20, seed=42)

model = dict(
    visual_stage='stage1',
    pretrained_name_or_path=_checkpoint,
    vla_head=dict(
        stage='stage1',
        lora=None,
    ),
)

train_dataloader = dict(
    dataset=dict(datasets=dict(task_selection=dict(_task_selection))))
val_dataloader = dict(
    dataset=dict(datasets=dict(task_selection=dict(_task_selection))))

# Bounded ablation budget: a step budget instead of an epoch budget.
runner = dict(max_epochs=None, max_steps=2000, save_iter_interval=500)
