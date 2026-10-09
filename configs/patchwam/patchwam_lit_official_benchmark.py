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
# Official PatchWAM + LIT RoboTwin 2.0 benchmark protocol.
#
# Mirrors `patchwam_robotwin_official_benchmark.py` (the released PatchWAM
# reference): seed 0, action horizon 16, replan 16, 10 denoising steps, unseen
# instructions, expert seed validation, 100 episodes per task per condition.
# The 50 released task names and the clean/randomized conditions are consumed
# from the shared `patchwam_robotwin_official_benchmark_tasks.json` manifest by
# `tools/patchwam_official_benchmark.py`; running the whole suite is the
# launcher's job, this config only fixes the per-case protocol.
#
# The only LIT-specific difference from the released reference is
# normalization: `dataset_stats_path` points at the Stage-2 training
# statistics (`dataset_statistics.json`) instead of the released PatchWAM
# asset, so the trained LIT weights are evaluated with the statistics they were
# trained under.
#
#   python tools/patchwam_official_benchmark.py \
#     --config configs/patchwam/patchwam_lit_official_benchmark.py \
#     --ckpt-path work_dirs/patchwam_lit_stage2_robotwin/checkpoints/latest-checkpoint.pt
# `--dry-run` lists the planned cases without loading a model or simulator.

_base_ = './patchwam_lit_eval_robotwin.py'

seed = 0

eval = dict(
    seed=seed,
    episodes_per_task=100,
    instruction_selection='official',
    instruction_pool_size=100,
    output_dir='runs/evaluation/patchwam_lit_robotwin_official',
)
